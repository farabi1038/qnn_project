import numpy as np
import qutip as qt
import torch  # Add PyTorch for GPU support
from copy import deepcopy
from quantum_utils import QuantumUtils
from logger import logger


class QNNArchitecture:
    """
    QNN Architecture functions integrating key concepts:
    - Building networks (Sec. III)
    - Feedforward logic (Sec. III.B)
    - Cost function & training (Sec. IV)
    - Quantum anomaly detection & thresholding (Sec. IV–V)
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

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

    @staticmethod
    def qnn_training(qnn_arch: list, unitaries: list, training_data: list,
                     learning_rate: float, epochs: int):
        """
        Train the QNN using gradient-based updates with GPU acceleration.
        """
        logger.info("Starting QNN training for %d epochs.", epochs)
        best_cost = float('inf')
        best_unitaries = None
        device = training_data[0][0].device

        # Convert unitaries to PyTorch parameters for gradient computation
        trainable_unitaries = [[torch.nn.Parameter(u.clone()) for u in layer] for layer in unitaries[1:]]
        optimizer = torch.optim.Adam(sum([list(layer) for layer in trainable_unitaries], []), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            output_states = QNNArchitecture.feedforward(qnn_arch, [[], *trainable_unitaries], training_data)
            
            # Calculate cost
            current_cost = QNNArchitecture.cost_function(training_data, output_states)
            
            # Backward pass
            loss = torch.tensor(current_cost, requires_grad=True, device=device)
            loss.backward()
            optimizer.step()
            
            # Update best model if needed
            if current_cost < best_cost:
                best_cost = current_cost
                best_unitaries = deepcopy([[u.detach() for u in layer] for layer in trainable_unitaries])
            
            logger.info("Epoch %d/%d: Cost=%.6f", epoch + 1, epochs, current_cost)
            
            if current_cost < 1e-6:  # Convergence check
                logger.info("Training converged early at epoch %d", epoch + 1)
                break

        return [[], *best_unitaries] if best_unitaries else [[], *trainable_unitaries]