import torch
import pennylane as qml
import numpy as np
import cupy as cp
import qutip as qt
from typing import Optional, Union, Tuple
from torch.utils.data import DataLoader, TensorDataset
from strawberryfields.ops import Dgate, Rgate, Sgate, BSgate
from logger import logger


class ContinuousVariableQNN:
    """
    GPU-accelerated Continuous Variable QNN using PyTorch, PennyLane, and CuPy.
    """
    
    def __init__(self, n_qubits=None, n_qumodes=None, n_layers=2, cutoff_dim: int = 10, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the CV QNN model with GPU support.
        """
        logger.info(f"Using device: {torch.cuda.is_available()}")
        # Handle both n_qubits and n_qumodes parameters
        if n_qubits is not None and n_qumodes is not None:
            raise ValueError("Please provide either n_qubits or n_qumodes, not both")
        
        self.n_qumodes = n_qumodes if n_qumodes is not None else n_qubits
        if self.n_qumodes is None:
            raise ValueError("Must provide either n_qubits or n_qumodes")

        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.device = device
        logger.info(f"Using device from self: {self.device}")
        
        # Use GPU-enabled device if available
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

        # Initialize parameters on GPU if available
        self.params = torch.nn.Parameter(
            torch.randn(self.n_layers, 3 * self.n_qumodes, 
                       device=self.device, requires_grad=True)
        )
        
        # Create the quantum circuit with proper decoration
        self.circuit = qml.QNode(self.circuit_def, self.dev)
        
    def to_device(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input data to torch tensor on correct device."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.device)

    def circuit_def(self, inputs, parameters):
        """Define the CV quantum circuit using PennyLane."""
        # Convert inputs to GPU tensor if needed
        inputs = self.to_device(inputs)
        
        # Encode inputs
        for i in range(self.n_qumodes):
            qml.Displacement(inputs[i].item(), 0.0, wires=i)

        # Apply variational layers
        for l in range(self.n_layers):
            for i in range(self.n_qumodes):
                params = parameters[l, 3*i:3*(i+1)]
                qml.Rotation(params[0].item(), wires=i)
                qml.Squeezing(params[1].item(), 0.0, wires=i)
                qml.Rotation(params[2].item(), wires=i)

            # Entangling operations
            for i in range(self.n_qumodes - 1):
                qml.Beamsplitter(np.pi/4, 0.0, wires=[i, i + 1])

        # Return expectations for all modes
        return [qml.expval(qml.NumberOperator(i)) for i in range(self.n_qumodes)]

    def forward_pass(self, X: Union[np.ndarray, torch.Tensor]) -> Tuple[list, torch.Tensor]:
        """
        Perform forward pass through the quantum circuit.
        
        Args:
            X: Input data
            
        Returns:
            tuple: (costs, output_states)
                - costs: List of costs for each input
                - output_states: Tensor of output states
        """
        # Convert input to tensor if needed
        X = self.to_device(X)
        
        # Calculate outputs
        costs = []
        outputs = []
        
        with torch.no_grad():
            for x in X:
                # Forward pass through circuit
                output = torch.tensor(self.circuit(x, self.params), 
                                    device=self.device)
                outputs.append(output)
                
                # Calculate cost (using L2 norm as example cost function)
                cost = torch.norm(output).item()
                costs.append(cost)
        
        output_states = torch.stack(outputs)
        
        return costs, output_states

    def train(self, X_train, y_train, batch_size=32, epochs=50, 
              learning_rate=0.01):
        """GPU-accelerated training using PyTorch."""
        # Convert data to PyTorch tensors
        X_train = self.to_device(X_train)
        y_train = self.to_device(y_train)
        
        # Create data loader for batching
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam([self.params], lr=learning_rate)
        
        training_costs = []
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = torch.stack([
                    torch.tensor(self.circuit(x, self.params), 
                               device=self.device)
                    for x in batch_X
                ])
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss/len(dataloader)
            training_costs.append(avg_loss)
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return training_costs