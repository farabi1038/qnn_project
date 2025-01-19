import torch
import pennylane as qml
import numpy as np
from typing import Union, Tuple
from torch.utils.data import DataLoader, TensorDataset
from logger import logger


class ContinuousVariableQNN(torch.nn.Module):
    """
    GPU-accelerated Continuous Variable QNN using PyTorch and PennyLane.
    """

    def __init__(self, n_qumodes: int, n_layers: int = 2, cutoff_dim: int = 10, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the CV QNN model with GPU support.
        
        Args:
            n_qumodes (int): Number of quantum modes (wires) in the circuit.
            n_layers (int): Number of layers in the variational quantum circuit.
            cutoff_dim (int): Fock space truncation dimension.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        super(ContinuousVariableQNN, self).__init__()  # Initialize nn.Module
        
        self.n_qumodes = n_qumodes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.device = device

        # Set up quantum device
        try:
            if self.device == 'cuda':
                self.dev = qml.device('lightning.gpu', wires=self.n_qumodes)
                logger.info("Using GPU-accelerated PennyLane device")
            else:
                self.dev = qml.device("strawberryfields.fock", wires=self.n_qumodes, cutoff_dim=self.cutoff_dim)
                logger.info(f"Using Strawberry Fields Fock device with cutoff_dim={self.cutoff_dim}")
        except Exception as e:
            logger.warning(f"Failed to initialize preferred device: {e}")
            logger.info("Falling back to default.gaussian device")
            self.dev = qml.device("default.gaussian", wires=self.n_qumodes)

        # Define trainable parameters
        self.params = torch.nn.Parameter(
            torch.randn(self.n_layers, 3 * self.n_qumodes, device=self.device, requires_grad=True)
        )

        # Create the quantum circuit
        self.qnode = qml.QNode(self.circuit_def, self.dev, interface="torch")

    def circuit_def(self, inputs: torch.Tensor, parameters: torch.Tensor):
        """
        Define the CV quantum circuit using PennyLane.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (n_qumodes,).
            parameters (torch.Tensor): Variational parameters of shape (n_layers, 3 * n_qumodes).
        
        Returns:
            list: Expectation values of the number operators for each mode.
        """
        # Encode inputs as displacements
        for i in range(self.n_qumodes):
            qml.Displacement(inputs[i], 0.0, wires=i)

        # Apply variational layers
        for l in range(self.n_layers):
            for i in range(self.n_qumodes):
                params = parameters[l, 3 * i:3 * (i + 1)]
                qml.Rotation(params[0], wires=i)
                qml.Squeezing(params[1], 0.0, wires=i)
                qml.Rotation(params[2], wires=i)

            # Add entangling beamsplitters
            for i in range(self.n_qumodes - 1):
                qml.Beamsplitter(np.pi / 4, 0.0, wires=[i, i + 1])

        return [qml.expval(qml.NumberOperator(i)) for i in range(self.n_qumodes)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the quantum circuit.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, n_qumodes).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_qumodes).
        """
        batch_size = inputs.shape[0]
        outputs = torch.zeros((batch_size, self.n_qumodes), device=self.device)

        for i in range(batch_size):
            outputs[i] = self.qnode(inputs[i], self.params)

        return outputs

    def compute_cost(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error loss.

        Args:
            predictions (torch.Tensor): Predicted outputs.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        return torch.nn.functional.mse_loss(predictions, targets)

    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int = 32, epochs: int = 50, learning_rate: float = 0.01) -> Tuple[list, torch.Tensor]:
        """
        Train the model using PyTorch.

        Args:
            X_train (torch.Tensor): Training data.
            y_train (torch.Tensor): Training labels.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimizer.

        Returns:
            Tuple[list, torch.Tensor]: Training loss history and final outputs.
        """
        # Create data loader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer
        optimizer = torch.optim.Adam([self.params], lr=learning_rate)

        # Train the model
        training_costs = []
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                predictions = self.forward(batch_X)

                # Compute loss
                loss = self.compute_cost(predictions, batch_y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            training_costs.append(avg_loss)

            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

        return training_costs, self.forward(X_train)
