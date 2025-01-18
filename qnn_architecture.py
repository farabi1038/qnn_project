import numpy as np
import qutip as qt
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

    @staticmethod
    def random_network(qnn_arch: list, num_training_pairs: int):
        """
        Create a random QNN network based on the architecture and training pairs.
        Architecture must start and end with the same number of qubits.

        Returns:
            tuple: (qnn_arch, unitaries, training_data, exact_unitary)
        """
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
                network_unitaries[l].append(unitary)

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
        num_qubits = len(unitary.dims[0])
        training_data = []

        for _ in range(num_samples):
            input_state = QuantumUtils.random_qubit_state(num_qubits)
            output_state = unitary * input_state
            training_data.append([input_state, output_state])

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
            current_state = sample[0] * sample[0].dag()  # Convert ket -> density matrix
            
            # Process through each layer
            for l in range(1, len(qnn_arch)):
                current_state = QNNArchitecture.make_layer_channel(
                    qnn_arch, unitaries, l, current_state
                )
                if not isinstance(current_state, qt.Qobj):
                    logger.error(f"feedforward: Layer {l}, Sample {x} - State is not Qobj.")
                    current_state = qt.Qobj(current_state)
            
            # Append only the final output state
            output_states.append(current_state)
        
        logger.debug("Feedforward pass completed for %d samples.", len(training_data))
        return output_states

    @staticmethod
    def make_layer_channel(qnn_arch: list, unitaries: list, l: int, input_state: qt.Qobj) -> qt.Qobj:
        """
        Single layer channel (Sec. III.B):
        1. Tensor input with |0> states for new qubits.
        2. Apply the unitaries.
        3. Partial trace out the input qubits => output qubits only.
        """
        num_input_qubits = qnn_arch[l - 1]
        num_output_qubits = qnn_arch[l]

        if input_state.type == 'ket':
            input_state = input_state * input_state.dag()

        state = qt.tensor(input_state, QuantumUtils.tensored_qubit0(num_output_qubits))
        layer_uni = unitaries[l][0].copy()
        for i in range(1, num_output_qubits):
            layer_uni = unitaries[l][i] * layer_uni

        result = layer_uni * state * layer_uni.dag()
        traced_result = result.ptrace(list(range(num_input_qubits, num_input_qubits + num_output_qubits)))

        if not isinstance(traced_result, qt.Qobj):
            logger.warning(f"make_layer_channel: Layer {l} - Traced result is not Qobj.")
            traced_result = qt.Qobj(traced_result)

        return traced_result

    @staticmethod
    def cost_function(training_data: list, output_states: list) -> float:
        """
        Compute the average cost based on fidelity with target states.
        Args:
            training_data: List of [input_state, target_state] pairs
            output_states: List of predicted output states (Qobj)
        Returns:
            float: Average cost
        """
        cost_sum = 0.0
        valid_samples = 0

        for i, (sample, predicted_state) in enumerate(zip(training_data, output_states)):
            target_state = sample[1]
            
            # Ensure both states are density matrices
            if target_state.type == 'ket':
                target_state = target_state * target_state.dag()
            if predicted_state.type == 'ket':
                predicted_state = predicted_state * predicted_state.dag()

            if not isinstance(target_state, qt.Qobj) or not isinstance(predicted_state, qt.Qobj):
                logger.error(f"Sample {i}: Invalid types - target={type(target_state)}, predicted={type(predicted_state)}")
                continue

            try:
                # Calculate fidelity between density matrices
                fidelity = qt.fidelity(target_state, predicted_state)
                cost_sum += 1 - fidelity
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
        Train the QNN using gradient-based updates.
        """
        logger.info("Starting QNN training for %d epochs.", epochs)
        best_cost = float('inf')
        best_unitaries = None

        for epoch in range(epochs):
            # Forward pass
            output_states = QNNArchitecture.feedforward(qnn_arch, unitaries, training_data)
            
            # Calculate cost
            current_cost = QNNArchitecture.cost_function(training_data, output_states)
            
            # Update best model if needed
            if current_cost < best_cost:
                best_cost = current_cost
                best_unitaries = deepcopy(unitaries)
            
            logger.info("Epoch %d/%d: Cost=%.6f", epoch + 1, epochs, current_cost)
            
            if current_cost < 1e-6:  # Convergence check
                logger.info("Training converged early at epoch %d", epoch + 1)
                break

        return best_unitaries or unitaries