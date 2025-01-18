import numpy as np
import qutip as qt
from quantum_utils import QuantumUtils


class DiscreteVariableQNN:
    """
    Implements a Discrete-Variable Quantum Neural Network (DV-QNN)
    using Qutip for quantum operations.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        """
        Initialize the Discrete QNN.

        :param n_qubits: Number of qubits in the network.
        :param n_layers: Number of layers in the quantum circuit.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = np.random.uniform(-0.1, 0.1, size=(n_layers, n_qubits * 3))  # Trainable parameters

    def forward_pass(self, inputs: np.ndarray) -> float:
        """
        Perform a forward pass through the quantum network.

        :param inputs: Input state vector as a NumPy array.
        :return: Anomaly score (float).
        """
        # Initialize the input state as a tensor product of single-qubit states
        input_state = qt.tensor([qt.basis(2, int(x)) for x in inputs])

        # Apply layers of the QNN
        current_state = input_state
        for layer_idx in range(self.n_layers):
            for qubit_idx in range(self.n_qubits):
                r1, s, r2 = self.params[layer_idx, qubit_idx * 3:(qubit_idx + 1) * 3]
                # Apply rotation gates and a placeholder operation
                current_state = QuantumUtils.apply_rotation(current_state, qubit_idx, r1, r2)
                current_state = QuantumUtils.apply_placeholder_operation(current_state, qubit_idx, s)

        # Calculate anomaly score as 1 - the trace of the density matrix
        try:
            anomaly_score = 1 - abs(current_state.tr().real)
        except Exception:
            anomaly_score = 0.0

        return anomaly_score

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Predict anomaly scores for a dataset.

        :param X: Input dataset as a 2D NumPy array.
        :param threshold: Threshold for anomaly detection.
        :return: Binary anomaly predictions (1=anomaly, 0=normal).
        """
        predictions = []
        for x in X:
            score = self.forward_pass(x)
            predictions.append(1 if score > threshold else 0)
        return np.array(predictions)
