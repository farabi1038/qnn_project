import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from copy import deepcopy

from src.utils import setup_logger
from src.utils.quantum_utils import QuantumUtils
import qutip as qt

logger = setup_logger()

class QNNArchitecture(torch.nn.Module):
    """
    GPU-accelerated Quantum Neural Network Architecture integrating:
    - Quantum feature encoding
    - Quantum-classical hybrid processing
    - Dynamic anomaly detection
    """
    def __init__(self, n_qubits: int, n_layers: int, device: Optional[str] = None):
        super(QNNArchitecture, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize quantum parameters on GPU
        self.quantum_params = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(3, n_qubits, device=self.device))
            for _ in range(n_layers)
        ])
        
        logger.info(f"Initialized QNN with {n_qubits} qubits, {n_layers} layers on {self.device}")

        self.quantum_utils = QuantumUtils()

    def _to_quantum_state(self, classical_data: torch.Tensor) -> torch.Tensor:
        """Convert classical data to quantum states using amplitude encoding"""
        # Normalize input
        normalized = classical_data / torch.norm(classical_data, dim=1, keepdim=True)
        # Map to quantum state space
        return normalized.to(self.device)

    def quantum_layer(self, state: torch.Tensor, layer_params: torch.Tensor) -> torch.Tensor:
        """Apply quantum operations for a single layer"""
        # ... existing code ...
        # Apply rotation gates
        rotated = self._apply_rotations(state, layer_params[0])
        # Apply entangling gates
        entangled = self._apply_entanglement(rotated, layer_params[1])
        # Apply non-linear transformation
        return self._apply_nonlinearity(entangled, layer_params[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum neural network"""
        # Convert classical input to quantum state
        quantum_state = self._to_quantum_state(x)
        
        # Apply quantum layers
        for layer_idx in range(self.n_layers):
            quantum_state = self.quantum_layer(
                quantum_state, 
                self.quantum_params[layer_idx]
            )
            
        # Compute anomaly score
        return self.compute_anomaly_score(quantum_state)

    def compute_anomaly_score(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score from quantum state"""
        # Calculate quantum state overlap
        overlap = torch.abs(torch.sum(torch.conj(quantum_state) * quantum_state, dim=1))
        return 1 - overlap

    @staticmethod
    def _to_torch(qobj):
        """Helper to convert QuTiP to PyTorch tensor"""
        return torch.tensor(qobj.full(), dtype=torch.complex64)

    @staticmethod
    def _to_qutip(tensor):
        """Helper to convert PyTorch tensor to QuTiP"""
        return qt.Qobj(tensor.cpu().numpy())

    @staticmethod
    def random_network(qnn_arch: list, num_training_pairs: int):
        """
        Create a random QNN network based on the architecture and training pairs.
        Architecture must start and end with the same number of qubits.

        Returns:
            tuple: (qnn_arch, unitaries, training_data, exact_unitary)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert qnn_arch[0] == qnn_arch[-1], "Architecture must start and end with the same number of qubits."

        # Generate the exact unitary for training data
        network_unitary = QuantumUtils.random_qubit_unitary(qnn_arch[-1])
        network_training_data = QNNArchitecture.random_training_data(network_unitary, num_training_pairs)

        # Build layer-by-layer unitaries
        network_unitaries = [[]]
        for l in range(1, len(qnn_arch)):
            num_input_qubits = qnn_arch[l - 1]
            num_output_qubits = qnn_arch[l]
            network_unitaries.append([])

            for j in range(num_output_qubits):
                unitary = QuantumUtils.random_qubit_unitary(num_input_qubits + 1)
                if num_output_qubits - 1 != 0:
                    unitary = qt.tensor(
                        QuantumUtils.random_qubit_unitary(num_input_qubits + 1),
                        QuantumUtils.tensored_id(num_output_qubits - 1)
                    )
                unitary = QuantumUtils.swapped_op(unitary, num_input_qubits, num_input_qubits + j)
                # Convert to PyTorch tensor for GPU support
                unitary_tensor = QNNArchitecture._to_torch(unitary).to(device)
                network_unitaries[l].append(unitary_tensor)

        logger.info("Random QNN network built with architecture: %s, Training Pairs: %d",
                    qnn_arch, num_training_pairs)
        return qnn_arch, network_unitaries, network_training_data, network_unitary

    @staticmethod
    def random_training_data(unitary: qt.Qobj, num_samples: int):
        """
        Generate training data using a given unitary.

        Args:
            unitary (qt.Qobj): Reference unitary.
            num_samples (int): Number of training samples.

        Returns:
            list: [(|input_state>, |output_state>), ...]
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_qubits = len(unitary.dims[0])
        training_data = []
        unitary_tensor = QNNArchitecture._to_torch(unitary).to(device)

        for _ in range(num_samples):
            input_state = QuantumUtils.random_qubit_state(num_qubits)
            input_tensor = QNNArchitecture._to_torch(input_state).to(device)
            output_tensor = torch.matmul(unitary_tensor, input_tensor)
            training_data.append([input_tensor, output_tensor])

        logger.debug("Generated %d training pairs.", num_samples)
        return training_data

    @staticmethod
    def feedforward(qnn_arch: list, unitaries: list, training_data: list):
        """
        Feedforward pass of the QNN (Sec. III.B):
        - Each sample goes layer by layer
        - Return the final output state for each sample
        """
        output_states = []
        for x, sample in enumerate(training_data):
            # Convert ket to density matrix using GPU operations
            current_state = torch.matmul(sample[0], sample[0].conj().T)
            
            # Process through each layer
            for l in range(1, len(qnn_arch)):
                current_state = QNNArchitecture.make_layer_channel(
                    qnn_arch, unitaries, l, current_state
                )
            
            output_states.append(current_state)
        
        logger.debug("Feedforward pass completed for %d samples.", len(training_data))
        return output_states

    @staticmethod
    def make_layer_channel(qnn_arch: list, unitaries: list, l: int, input_state: torch.Tensor) -> torch.Tensor:
        """
        Single layer channel (Sec. III.B) with GPU support:
        1. Tensor input with |0> states for new qubits.
        2. Apply the unitaries.
        3. Partial trace out the input qubits => output qubits only.
        """
        device = input_state.device
        num_input_qubits = qnn_arch[l - 1]
        num_output_qubits = qnn_arch[l]

        # Create zero state tensor on GPU
        zero_state = torch.zeros((2**num_output_qubits, 1), dtype=torch.complex64, device=device)
        zero_state[0] = 1.0

        state = torch.kron(input_state, torch.matmul(zero_state, zero_state.conj().T))
        
        # Apply unitaries
        layer_uni = unitaries[l][0]
        for i in range(1, num_output_qubits):
            layer_uni = torch.matmul(unitaries[l][i], layer_uni)

        result = torch.matmul(torch.matmul(layer_uni, state), layer_uni.conj().T)
        
        # Partial trace implementation for GPU tensors
        traced_dims = [2] * (num_input_qubits + num_output_qubits)
        reshaped_tensor = result.reshape(traced_dims + traced_dims)
        traced_result = torch.trace(reshaped_tensor.permute(
            list(range(num_input_qubits, num_input_qubits + num_output_qubits)) +
            list(range(num_input_qubits + num_output_qubits + num_input_qubits,
                      num_input_qubits + num_output_qubits + num_input_qubits + num_output_qubits))
        ))

        return traced_result

    @staticmethod
    def cost_function(training_data: list, output_states: list) -> float:
        """
        Compute the average cost based on fidelity with target states.
        """
        cost_sum = 0.0
        valid_samples = 0

        for i, (sample, predicted_state) in enumerate(zip(training_data, output_states)):
            target_state = torch.matmul(sample[1], sample[1].conj().T)
            
            try:
                # Calculate fidelity between density matrices using GPU operations
                sqrt_pred = torch.matrix_power(predicted_state, 0.5)
                fidelity = torch.abs(torch.trace(torch.matmul(sqrt_pred,
                                   torch.matmul(target_state, sqrt_pred))))
                cost_sum += 1 - fidelity.item()
                valid_samples += 1
            except Exception as e:
                logger.error(f"Error in fidelity calculation for sample {i}: {e}")

        avg_cost = cost_sum / valid_samples if valid_samples > 0 else float('inf')
        logger.info(f"Average cost computed: {avg_cost:.4f}")
        return avg_cost

    def create_circuit(self, num_qubits: int) -> qt.Qobj:
        """Create a quantum circuit with the specified number of qubits."""
        return self.quantum_utils.tensored_id(num_qubits)
    
    def apply_gates(self, state: qt.Qobj, num_qubits: int) -> qt.Qobj:
        """Apply quantum gates to the state."""
        return self.quantum_utils.random_qubit_unitary(num_qubits) * state
