import torch
import torch.nn as nn
import numpy as np
import cupy as cp
import qutip as qt
from typing import Optional, Union, List, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset
from strawberryfields.ops import Dgate, Rgate, Sgate, BSgate
from ..utils import setup_logger
from ..utils.quantum_utils import QuantumUtils
import torch.cuda as cuda

logger = setup_logger()

class DiscreteVariableQNN(nn.Module):
    """
    GPU-accelerated Discrete-Variable QNN using PyTorch and CuPy.
    Implements quantum-enhanced anomaly detection as described in the paper.
    """
    
    def __init__(self, n_qubits: int, n_layers: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the DV-QNN with GPU support."""
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        
        # Track memory usage
        self.memory_tracker = {
            'allocated': 0,
            'cached': 0
        }
        
        # Enable cuDNN benchmarking for optimal performance
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Initialize quantum parameters with requires_grad=True
        self.params = nn.Parameter(
            torch.randn(n_layers, n_qubits * 3, device=device, requires_grad=True)
        )
        
        # Initialize quantum operators on GPU using CuPy if available
        if self.device == 'cuda':
            self.xp = cp
        else:
            self.xp = np
            
        self.initialize_quantum_operators()
        
        # Initialize state tracking
        self.current_state = None
        
        # Move model to specified device
        self.to(device)
        
        # Add adaptive threshold parameters
        self.anomaly_threshold = nn.Parameter(
            torch.tensor(0.5, device=self.device)
        )
        self.threshold_history = []
        
        self.quantum_utils = QuantumUtils()
        
    def initialize_quantum_operators(self):
        """Initialize quantum operators using CuPy/NumPy."""
        # Basic quantum gates as defined in paper
        if self.device == 'cuda':
            self.I = cp.eye(2, dtype=complex)
            self.X = cp.array([[0, 1], [1, 0]], dtype=complex)
            self.Y = cp.array([[0, -1j], [1j, 0]], dtype=complex)
            self.Z = cp.array([[1, 0], [0, -1]], dtype=complex)
            self.H = cp.array([[1, 1], [1, -1]], dtype=complex) / cp.sqrt(2)
            self.CNOT = cp.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=complex)
        else:
            self.I = np.eye(2, dtype=complex)
            self.X = np.array([[0, 1], [1, 0]], dtype=complex)
            self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
            self.H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            self.CNOT = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=complex)
        
    def rotation_matrix(self, angle: torch.Tensor) -> Union[np.ndarray, cp.ndarray]:
        """Generate rotation matrix for quantum gates."""
        try:
            # Move angle to CPU for calculation
            angle_cpu = angle.detach().cpu()
            
            if self.device == 'cuda':
                return cp.array([
                    [cp.cos(angle_cpu/2), -1j*cp.sin(angle_cpu/2)],
                    [-1j*cp.sin(angle_cpu/2), cp.cos(angle_cpu/2)]
                ], dtype=complex)
            else:
                return np.array([
                    [np.cos(angle_cpu/2), -1j*np.sin(angle_cpu/2)],
                    [-1j*np.sin(angle_cpu/2), np.cos(angle_cpu/2)]
                ], dtype=complex)
        except Exception as e:
            logger.error(f"Error in rotation_matrix: {str(e)}, "
                        f"angle type: {type(angle)}, "
                        f"device: {angle.device if torch.is_tensor(angle) else 'unknown'}")
            raise
        
    def apply_gate(self, state: torch.Tensor, gate: Union[np.ndarray, cp.ndarray], 
                   qubit: int) -> torch.Tensor:
        """Apply quantum gate to specific qubit."""
        try:
            state_cp = None  # Initialize state_cp
            
            # Convert state to appropriate array type
            if self.device == 'cuda':
                if isinstance(state, torch.Tensor):
                    state_np = state.detach().cpu().numpy()
                    state_cp = cp.asarray(state_np)
                else:
                    state_cp = cp.asarray(state)
                gate_cp = cp.asarray(gate)
            else:
                if isinstance(state, torch.Tensor):
                    state_np = state.detach().cpu().numpy()
                    state_cp = np.asarray(state_np)
                else:
                    state_cp = np.asarray(state)
                gate_cp = np.asarray(gate)
            
            if state_cp is None:
                raise ValueError("Failed to convert state to array")
            
            # Ensure state has correct shape
            if state_cp.ndim == 1:
                n_qubits = int(np.log2(len(state_cp)))
                state_cp = state_cp.reshape([2] * n_qubits)
            
            # Apply gate
            if self.device == 'cuda':
                # Prepare indices for the operation
                perm = list(range(state_cp.ndim))
                perm[0], perm[qubit] = perm[qubit], perm[0]
                
                # Reshape and transpose
                state_cp = state_cp.transpose(perm)
                state_cp = state_cp.reshape(2, -1)
                
                # Apply gate
                result = cp.dot(gate_cp, state_cp)
                
                # Reshape back
                result = result.reshape([2] * n_qubits)
                result = result.transpose(perm)
                
                # Convert back to flat array
                result = result.ravel()
                result_np = cp.asnumpy(result)
            else:
                # Same operations for CPU
                perm = list(range(state_cp.ndim))
                perm[0], perm[qubit] = perm[qubit], perm[0]
                
                state_cp = state_cp.transpose(perm)
                state_cp = state_cp.reshape(2, -1)
                
                result = np.dot(gate_cp, state_cp)
                
                result = result.reshape([2] * n_qubits)
                result = result.transpose(perm)
                result_np = result.ravel()
            
            # Convert to torch tensor and move to appropriate device
            result = torch.from_numpy(result_np).to(self.device)
            result.requires_grad_(True)
            return result
        
        except Exception as e:
            logger.error(f"Error in apply_gate: {str(e)}, "
                        f"gate type: {type(gate)}, "
                        f"state type: {type(state)}, "
                        f"state_cp type: {type(state_cp) if state_cp is not None else 'None'}, "
                        f"state shape: {state.shape if hasattr(state, 'shape') else 'unknown'}, "
                        f"requires_grad: {state.requires_grad if torch.is_tensor(state) else 'N/A'}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for batch processing."""
        try:
            batch_size = x.shape[0]
            anomaly_scores = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            
            # Process samples in smaller sub-batches if needed
            sub_batch_size = min(16, batch_size)  # Maximum size that our quantum circuit can handle
            num_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size
            
            for i in range(num_sub_batches):
                start_idx = i * sub_batch_size
                end_idx = min((i + 1) * sub_batch_size, batch_size)
                
                # Process current sub-batch
                sub_batch = x[start_idx:end_idx]
                sub_scores = []
                
                for j in range(len(sub_batch)):
                    sample = sub_batch[j]
                    score = self.forward_pass(sample)
                    sub_scores.append(score.real)
                
                # Convert sub-batch scores to tensor
                sub_scores = torch.tensor(sub_scores, device=self.device, dtype=torch.float32)
                anomaly_scores[start_idx:end_idx] = sub_scores
            
            # Enable gradients
            anomaly_scores.requires_grad_(True)
            
            # Log shape information
            logger.debug(f"Forward pass output shape: {anomaly_scores.shape}, "
                        f"Expected batch size: {batch_size}")
            
            return anomaly_scores
        
        except Exception as e:
            logger.error(f"Error in forward: {str(e)}, "
                        f"input shape: {x.shape}, "
                        f"output shape: {anomaly_scores.shape if 'anomaly_scores' in locals() else 'N/A'}")
            raise

    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with state tracking."""
        try:
            # Encode input into quantum state
            state = self.encode_features(inputs)
            
            # Ensure state requires grad and is on correct device
            state = state.to(self.device)
            state.requires_grad_(True)
            
            # Store initial state
            self.current_state = state.clone()
            
            # Apply variational layers
            for layer in range(self.n_layers):
                # Apply single-qubit rotations
                for qubit in range(self.n_qubits):
                    params = self.params[layer][qubit*3:(qubit+1)*3]
                    state = self.apply_gate(state, self.rotation_matrix(params[0]), qubit)
                    state = self.apply_gate(state, self.rotation_matrix(params[1]), qubit)
                    state = self.apply_gate(state, self.rotation_matrix(params[2]), qubit)
                
                # Apply entangling CNOT gates
                for q in range(self.n_qubits - 1):
                    state = self.apply_cnot(state, q, q+1)
                    state.requires_grad_(True)
                
                # Update current state after each layer
                self.current_state = state.clone()
            
            # Calculate anomaly score using quantum state overlap
            score = 1 - torch.abs(torch.sum(torch.conj(state) * state))
            score.requires_grad_(True)
            return score.real  # Return only real part
        except Exception as e:
            logger.error(f"Error in forward_pass: {str(e)}")
            raise

    def predict(self, X: Union[np.ndarray, torch.Tensor], 
               threshold: float) -> torch.Tensor:
        """GPU-accelerated prediction."""
        X = torch.as_tensor(X, device=self.device)
        scores = torch.tensor([self.forward_pass(x) for x in X])
        return (scores > threshold).float()

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode classical features into quantum states."""
        try:
            # Initialize quantum state
            state = torch.zeros(2**self.n_qubits, device='cpu', 
                              dtype=torch.complex64)
            state[0] = 1.0  # Initialize to |0⟩ state
            
            # Ensure x is on CPU for processing
            x = x.cpu()
            
            # Handle both single sample and batch
            if x.dim() == 1:
                features = x
            else:
                features = x.flatten()  # Handle batch by flattening
            
            # Apply encoding gates based on feature values
            for i, val in enumerate(features):
                if i >= self.n_qubits:  # Ensure we don't exceed n_qubits
                    break
                
                # Convert complex values to real by taking magnitude
                if torch.is_complex(val):
                    val_magnitude = torch.abs(val)
                else:
                    val_magnitude = val
                    
                # Compare real values
                if val_magnitude.item() > 0:
                    state = self.apply_gate(state, self.X, i)
                # Apply Hadamard gates for superposition
                state = self.apply_gate(state, self.H, i)
            
            return state.to(self.device)
        except Exception as e:
            logger.error(f"Error in encode_features: {str(e)}, input shape: {x.shape}")
            raise

    def apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate between control and target qubits."""
        try:
            # Detach for numpy conversion while preserving graph
            if self.device == 'cuda':
                if isinstance(state, torch.Tensor):
                    state_cp = cp.asarray(state.detach().cpu().numpy())
                else:
                    state_cp = cp.asarray(state)
            else:
                if isinstance(state, torch.Tensor):
                    state_cp = np.asarray(state.detach().cpu().numpy())
                else:
                    state_cp = np.asarray(state)
            
            # Ensure state has correct shape
            if state_cp.ndim == 1:
                n_qubits = int(np.log2(len(state_cp)))
                state_cp = state_cp.reshape([2] * n_qubits)
            
            # Apply CNOT
            if self.device == 'cuda':
                # Prepare indices for the operation
                perm = list(range(state_cp.ndim))
                perm[0], perm[control] = perm[control], perm[0]
                perm[1], perm[target] = perm[target], perm[1]
                
                # Reshape and transpose
                state_cp = state_cp.transpose(perm)
                state_cp = state_cp.reshape(4, -1)
                
                # Apply CNOT
                result = cp.dot(self.CNOT, state_cp)
                
                # Reshape back
                result = result.reshape([2] * n_qubits)
                result = result.transpose(perm)
                
                # Convert back to flat array
                result = result.ravel()
                result_np = cp.asnumpy(result)
            else:
                # Same operations for CPU
                perm = list(range(state_cp.ndim))
                perm[0], perm[control] = perm[control], perm[0]
                perm[1], perm[target] = perm[target], perm[1]
                
                state_cp = state_cp.transpose(perm)
                state_cp = state_cp.reshape(4, -1)
                
                result = np.dot(self.CNOT, state_cp)
                
                result = result.reshape([2] * n_qubits)
                result = result.transpose(perm)
                result_np = result.ravel()
            
            # Convert back to torch tensor with gradients
            result = torch.from_numpy(result_np).to(self.device)
            result.requires_grad_(True)
            
            # Create new tensor that shares gradient history
            if state.requires_grad:
                result = result + 0 * state.sum()
            
            return result
        
        except Exception as e:
            logger.error(f"Error in apply_cnot: {str(e)}, "
                        f"state type: {type(state)}, "
                        f"requires_grad: {state.requires_grad if torch.is_tensor(state) else 'N/A'}, "
                        f"device: {state.device if torch.is_tensor(state) else 'unknown'}")
            raise

    def get_quantum_state(self) -> torch.Tensor:
        """Get current quantum state for analysis."""
        return self.current_state if hasattr(self, 'current_state') else None

    def compute_cost(self, output_states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the cost/loss for training.
        
        Args:
            output_states (torch.Tensor): Model outputs (batch_size,)
            targets (torch.Tensor): Ground truth labels (batch_size,)
            
        Returns:
            torch.Tensor: Computed loss value
        """
        try:
            # Binary Cross Entropy Loss for anomaly detection
            criterion = nn.BCELoss()
            
            # Ensure output_states are probabilities (between 0 and 1)
            output_probs = torch.sigmoid(output_states)
            output_probs.requires_grad_(True)
            
            # Convert target to float and move to correct device
            targets = targets.float().to(self.device)
            
            # Get batch sizes
            output_batch_size = output_probs.shape[0]
            target_batch_size = targets.shape[0]
            
            # Log batch sizes for debugging
            logger.debug(f"Output batch size: {output_batch_size}, Target batch size: {target_batch_size}")
            
            # Ensure shapes match
            if output_batch_size != target_batch_size:
                logger.warning(f"Batch size mismatch: output {output_batch_size} != target {target_batch_size}")
                raise ValueError(f"Batch size mismatch: output {output_batch_size} != target {target_batch_size}")
            
            # Compute loss
            loss = criterion(output_probs, targets)
            loss.requires_grad_(True)
            
            return loss
        
        except Exception as e:
            logger.error(f"Error in compute_cost: {str(e)}, "
                        f"output shape: {output_states.shape}, "
                        f"target shape: {targets.shape}, "
                        f"output requires_grad: {output_states.requires_grad}, "
                        f"output device: {output_states.device}")
            raise

    def quantum_cost_function(self, output_states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Alternative name for compute_cost to maintain compatibility."""
        return self.compute_cost(output_states, target)

    def update_threshold(self, false_positive_rate: float, 
                        target_fpr: float = 0.01,
                        learning_rate: float = 0.01):
        """Dynamically adjust anomaly threshold based on FPR."""
        adjustment = learning_rate * (false_positive_rate - target_fpr)
        self.anomaly_threshold.data += adjustment
        self.threshold_history.append(self.anomaly_threshold.item())
        
    def quantum_feature_importance(self) -> torch.Tensor:
        """Calculate feature importance using quantum measurements."""
        importance = torch.zeros(self.n_qubits, device=self.device)
        for i in range(self.n_qubits):
            # Measure impact of each qubit on final state
            with torch.no_grad():
                base_state = torch.zeros(2**self.n_qubits, device=self.device,
                                      dtype=torch.complex64)
                base_state[0] = 1.0
                # Apply X gate to measure feature impact
                modified_state = self.apply_gate(base_state, self.X, i)
                importance[i] = torch.abs(
                    torch.sum(torch.conj(base_state) * modified_state)
                )
        return importance
        
    def get_circuit_depth(self) -> int:
        """Calculate quantum circuit depth."""
        # Base depth from encoding
        depth = 2  # Initial H gates + encoding
        # Add depth from variational layers
        depth += self.n_layers * (3 + self.n_qubits - 1)  # Rotations + CNOTs
        return depth
        
    def estimate_resource_requirements(self) -> Dict[str, int]:
        """Estimate quantum resources needed."""
        return {
            'n_qubits': self.n_qubits,
            'circuit_depth': self.get_circuit_depth(),
            'n_parameters': sum(p.numel() for p in self.parameters()),
            'n_cnot_gates': self.n_layers * (self.n_qubits - 1)
        }
        
    def save_quantum_state(self, path: str):
        """Save quantum state for analysis."""
        if hasattr(self, 'current_state'):
            torch.save({
                'quantum_state': self.current_state,
                'parameters': self.state_dict(),
                'threshold': self.anomaly_threshold.item(),
                'circuit_info': self.estimate_resource_requirements()
            }, path)
            
    def load_quantum_state(self, path: str):
        """Load quantum state and parameters."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.current_state = checkpoint['quantum_state']
        self.anomaly_threshold.data = torch.tensor(
            checkpoint['threshold'], device=self.device
        )

    def track_memory(self) -> Dict[str, int]:
        """Monitor GPU memory usage"""
        if self.device == 'cuda':
            self.memory_tracker.update({
                'allocated': cuda.memory_allocated(),
                'cached': cuda.memory_reserved()
            })
            return self.memory_tracker
        return {'allocated': 0, 'cached': 0}

    def clear_cache(self):
        """Clear GPU cache if memory pressure is high"""
        if self.device == 'cuda':
            cuda.empty_cache()
            logger.info("GPU cache cleared")

    def process_batch(self, batch_X: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        """Optimized batch processing for single GPU"""
        try:
            # Pin memory for faster GPU transfer
            if self.device == 'cuda' and not batch_X.is_pinned():
                batch_X = batch_X.pin_memory()
                batch_y = batch_y.pin_memory()
            
            # Process batch with automatic memory management
            with torch.cuda.stream(torch.cuda.Stream()):
                output_states = self(batch_X)
                cost = self.compute_cost(output_states, batch_y)
                
                # Monitor memory usage
                current_memory = self.track_memory()
                if current_memory['allocated'] > 0.9 * cuda.get_device_properties(0).total_memory:
                    self.clear_cache()
                    
            return output_states, cost
            
        except cuda.OutOfMemoryError:
            logger.error("GPU out of memory - attempting recovery")
            self.clear_cache()
            # Try with half precision if needed
            if batch_X.dtype != torch.float16:
                logger.info("Switching to half precision")
                batch_X = batch_X.half()
                batch_y = batch_y.half()
                return self.process_batch(batch_X, batch_y)
            raise

    def optimize_memory(self):
        """Optimize memory usage for single GPU"""
        if self.device == 'cuda':
            # Use mixed precision when beneficial
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler()
            
            # Optimize memory allocation
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(0.9)  # Reserve some memory for system
            logger.info("Memory optimization settings applied")

    def update_quantum_parameters(self):
        """Update quantum parameters with memory optimization"""
        try:
            with torch.no_grad():
                # Existing parameter update code
                self.params.data = torch.remainder(self.params.data, 2 * np.pi)
                
                if hasattr(self, 'training') and self.training:
                    noise = torch.randn_like(self.params.data) * 0.01
                    self.params.data += noise
                
                # Ensure parameters stay on GPU
                self.params.data = self.params.data.to(self.device)
                
                # Monitor memory after update
                self.track_memory()
                
        except Exception as e:
            logger.error(f"Error in update_quantum_parameters: {str(e)}")
            self.clear_cache()
            raise

    def train(self, mode: bool = True):
        """Set training mode with memory optimization"""
        super().train(mode)
        if mode and self.device == 'cuda':
            self.optimize_memory()
        return self

    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)

def create_quantum_dataloader(X: np.ndarray, y: np.ndarray, 
                            batch_size: int = 32,
                            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                            ) -> DataLoader:
    """Create a PyTorch DataLoader for quantum data with GPU support."""
    X_tensor = torch.tensor(X, device=device)
    y_tensor = torch.tensor(y, device=device)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)