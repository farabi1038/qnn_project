import pennylane as qml
import strawberryfields as sf
from strawberryfields.ops import Dgate, Rgate, Sgate, BSgate
from logger import logger


class ContinuousVariableQNN:
    """
    Continuous Variable (CV) Quantum Neural Network using Strawberry Fields
    and PennyLane.
    """

    def __init__(self, n_qumodes: int, n_layers: int, cutoff_dim: int = 10):
        """
        Initialize the CV QNN model.

        :param n_qumodes: Number of qumodes (features).
        :param n_layers: Number of layers in the variational circuit.
        :param cutoff_dim: Cutoff dimension for Fock space.
        """
        self.n_qumodes = n_qumodes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim

        # Use a PennyLane device wrapping Strawberry Fields
        self.dev = qml.device("strawberryfields.fock", wires=n_qumodes, cutoff_dim=cutoff_dim)
        

        # Initialize parameters: Random initialization for the circuit
        self.params = self.initialize_parameters()

    def initialize_parameters(self):
        """
        Randomly initialize the parameters for the variational circuit.

        :return: NumPy array of trainable parameters.
        """
        import numpy as np
        np.random.seed(42)
        return np.random.uniform(-0.1, 0.1, size=(self.n_layers, 3 * self.n_qumodes))

    def circuit(self, inputs, parameters):
        """
        Define the CV quantum circuit using PennyLane.

        :param inputs: Input features (classical data).
        :param parameters: Trainable parameters for the circuit.
        """
        # Encode the classical inputs using displacement gates
        for i in range(self.n_qumodes):
            qml.Displacement(inputs[i], 0.0, wires=i)

        # Apply variational layers
        for l in range(self.n_layers):
            for i in range(self.n_qumodes):
                r1, s, r2 = parameters[l, 3 * i:3 * (i + 1)]
                qml.Rotation(r1, wires=i)
                qml.Squeeze(s, 0.0, wires=i)
                qml.Rotation(r2, wires=i)

            # Apply entangling gates (e.g., beam splitters)
            for i in range(self.n_qumodes - 1):
                qml.Beamsplitter(0.5, 0.0, wires=[i, i + 1])

    def forward_pass(self, inputs):
        """
        Perform a forward pass to compute the anomaly score.

        :param inputs: Input data (1D array of length `n_qumodes`).
        :return: Anomaly score (float).
        """
        @qml.qnode(self.dev)
        def qnn(inputs, parameters):
            self.circuit(inputs, parameters)
            return qml.expval(qml.NumberOperator(0))

        score = qnn(inputs, self.params)
        return 1 - score  # Map score to an anomaly likelihood [0, 1]

    def train(self, X_train, y_train, max_steps=50, learning_rate=0.01):
        """
        Train the QNN using a gradient descent optimizer.

        :param X_train: Training features.
        :param y_train: Training labels.
        :param max_steps: Number of training iterations.
        :param learning_rate: Learning rate for optimization.
        """
        opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

        def cost_fn(parameters, X, y):
            loss = 0
            for xi, yi in zip(X, y):
                prediction = self.forward_pass(xi)
                loss += (prediction - yi) ** 2
            return loss / len(X)

        for step in range(max_steps):
            self.params = opt.step(lambda p: cost_fn(p, X_train, y_train), self.params)
            if step % 5 == 0:
                current_loss = cost_fn(self.params, X_train, y_train)
                logger.info(f"Step {step}: Loss = {current_loss:.4f}")
