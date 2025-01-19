import torch
import pennylane as qml
import numpy as np
import cupy as cp
import qutip as qt
from typing import Optional, Union
from torch.utils.data import DataLoader, TensorDataset
from strawberryfields.ops import Dgate, Rgate, Sgate, BSgate
from logger import logger

class DiscreteVariableQNN:
    """
    GPU-accelerated Discrete-Variable QNN using PyTorch and CuPy.
    """
    
    def __init__(self, n_qubits: int, n_layers: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the DV-QNN with GPU support."""
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        
        # Initialize parameters on GPU if available
        self.params = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits * 3, 
                       device=self.device, requires_grad=True)
        )
        
        # Initialize quantum operators on GPU using CuPy if available
        if self.device == 'cuda':
            self.xp = cp
        else:
            self.xp = np
            
        self.initialize_quantum_operators()
        
    def initialize_quantum_operators(self):
        """Initialize quantum operators using CuPy/NumPy."""
        # Basic quantum gates
        self.I = self.xp.eye(2, dtype=complex)
        self.X = self.xp.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = self.xp.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = self.xp.array([[1, 0], [0, -1]], dtype=complex)
        
    def rotation_matrix(self, angle: float) -> np.ndarray:
        """Generate rotation matrix."""
        return self.xp.array([
            [self.xp.cos(angle/2), -1j*self.xp.sin(angle/2)],
            [-1j*self.xp.sin(angle/2), self.xp.cos(angle/2)]
        ], dtype=complex)
        
    def apply_gate(self, state: torch.Tensor, gate: np.ndarray, 
                  qubit: int) -> torch.Tensor:
        """Apply quantum gate to specific qubit."""
        # Convert state to GPU if needed
        if self.device == 'cuda':
            state = cp.asarray(state.cpu().numpy())
            
        # Reshape for gate application
        state_shape = [2] * self.n_qubits
        state = state.reshape(state_shape)
        
        # Apply gate
        state = self.xp.tensordot(gate, state, axes=(1, qubit))
        
        # Convert back to torch tensor
        if self.device == 'cuda':
            state = torch.from_numpy(cp.asnumpy(state)).to(self.device)
        else:
            state = torch.from_numpy(state).to(self.device)
            
        return state
        
    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated forward pass."""
        # Initialize quantum state
        state = torch.zeros(2**self.n_qubits, device=self.device, 
                          dtype=torch.complex64)
        state[0] = 1.0
        
        # Apply input encoding
        for i, x in enumerate(inputs):
            if x > 0:
                state = self.apply_gate(state, self.X, i)
        
        # Apply variational layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                params = self.params[layer, qubit*3:(qubit+1)*3]
                
                # Apply rotations and quantum operations
                state = self.apply_gate(state, 
                    self.rotation_matrix(params[0].item()), qubit)
                state = self.apply_gate(state, 
                    self.rotation_matrix(params[1].item()), qubit)
                state = self.apply_gate(state, 
                    self.rotation_matrix(params[2].item()), qubit)
        
        # Calculate anomaly score
        return 1 - torch.abs(torch.sum(torch.conj(state) * state))

    def predict(self, X: Union[np.ndarray, torch.Tensor], 
               threshold: float) -> torch.Tensor:
        """GPU-accelerated prediction."""
        X = torch.as_tensor(X, device=self.device)
        scores = torch.tensor([self.forward_pass(x) for x in X])
        return (scores > threshold).float()


def create_quantum_dataloader(X: np.ndarray, y: np.ndarray, 
                            batch_size: int = 32,
                            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                            ) -> DataLoader:
    """Create a PyTorch DataLoader for quantum data with GPU support."""
    X_tensor = torch.tensor(X, device=device)
    y_tensor = torch.tensor(y, device=device)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)