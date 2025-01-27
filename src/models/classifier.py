import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import logging
from typing import List, Dict, Optional

def quantum_feature_encoding(x_input, wires):
    """Simple displacement-based encoding on the given 'wires'."""
    # Ensure we don't try to encode more features than we have wires
    n_features = len(x_input)
    n_wires = len(wires)
    n_encode = min(n_features, n_wires)
    
    for i in range(n_encode):
        qml.Displacement(float(x_input[i]), 0.0, wires=wires[i])

def interferometer(params, wires):
    """Implement interferometer layer."""
    n = len(wires)
    if n <= 1:  # Skip if only one wire
        return
        
    chunk = (n - 1)
    thetas = params[0:chunk]
    phis = params[chunk:2*chunk]
    rloc = params[2*chunk:3*chunk]

    for i in range(n-1):
        qml.Beamsplitter(float(thetas[i]), float(phis[i]), wires=[wires[i], wires[i+1]])
        qml.Rotation(float(rloc[i]), wires=wires[i])

def cv_layer(params: Dict, wires: List[int]):
    """
    Implement a single CV layer with:
    - Interferometer U1
    - local Squeezing
    - Interferometer U2
    - local Displacements
    (Removed Kerr operations as they're not supported by gaussian devices)
    """
    n = len(wires)
    if n <= 0:  # Safety check
        return
        
    # 1) U1
    interferometer(params['int1'], wires)
    
    # 2) Squeezing
    for i in range(n):
        qml.Squeezing(float(params['squeezes'][i]), 0.0, wires=wires[i])
    
    # 3) U2
    interferometer(params['int2'], wires)
    
    # 4) Displacements
    for i in range(n):
        qml.Displacement(float(params['displacements'][i]), 0.0, wires=wires[i])

def make_cv_qnode(n_layers: int, layer_widths: List[int], out_classes: int = 3, cutoff: int = 6):
    """Create a QNode for 3-class classification."""
    # Ensure we have enough wires for both input features and output classes
    max_width = max(max(layer_widths), out_classes)
    
    # Use default.gaussian instead of strawberryfields.fock
    dev = qml.device("default.gaussian", wires=max_width)

    @qml.qnode(dev, interface="torch")
    def circuit(all_params, x_input):
        # 1) encode
        input_wires = range(layer_widths[0])
        quantum_feature_encoding(x_input, wires=input_wires)

        # 2) layers
        for idx in range(n_layers):
            n_qumodes = layer_widths[idx]
            layer_wires = range(n_qumodes)
            cv_layer(all_params[idx], wires=layer_wires)

        # 3) measure out_classes wires using number operator
        measurements = []
        for w in range(min(out_classes, max_width)):
            measurements.append(qml.expval(qml.NumberOperator(wires=w)))
        return measurements
    
    return circuit

class CVQNNClassifier(nn.Module):
    """Multi-class CV-QNN with trainable parameters across multiple layers."""
    
    def __init__(
        self,
        in_features: int = 5,
        n_layers: int = 2,
        layer_widths: Optional[List[int]] = None,
        out_classes: int = 3,
        cutoff: int = 6
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        if layer_widths is None:
            layer_widths = [max(in_features, out_classes)] * n_layers
        
        layer_widths[0] = max(layer_widths[0], in_features)
        
        self.in_features = in_features
        self.out_classes = out_classes
        self.n_layers = n_layers
        self.layer_widths = layer_widths
        self.cutoff = cutoff

        self.qnode = make_cv_qnode(n_layers, layer_widths, out_classes, cutoff)

        # Build trainable layer params (removed Kerr parameters)
        self.layer_params = nn.ParameterList()
        for layer_idx in range(self.n_layers):
            n = self.layer_widths[layer_idx]
            if n <= 1:
                continue
                
            int1_size = 3*(n-1)
            int2_size = 3*(n-1)
            
            p_int1 = nn.Parameter(0.1*torch.randn(int1_size))
            p_squeezes = nn.Parameter(0.1*torch.randn(n))
            p_int2 = nn.Parameter(0.1*torch.randn(int2_size))
            p_disp = nn.Parameter(0.1*torch.randn(n))
            
            self.layer_params.append(nn.ParameterList([
                p_int1, p_squeezes, p_int2, p_disp
            ]))
            
        self.logger.info(
            f"Initialized CVQNNClassifier with:\n"
            f"  Input features: {in_features}\n"
            f"  Number of layers: {n_layers}\n"
            f"  Layer widths: {layer_widths}\n"
            f"  Output classes: {out_classes}\n"
            f"  Cutoff dimension: {cutoff}"
        )

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum circuit."""
        try:
            # Prepare layer parameters
            all_params = []
            for layer_idx in range(self.n_layers):
                if layer_idx >= len(self.layer_params):
                    continue
                    
                p_list = list(self.layer_params[layer_idx])
                param_dict = {
                    'int1': p_list[0],
                    'squeezes': p_list[1],
                    'int2': p_list[2],
                    'displacements': p_list[3]
                }
                all_params.append(param_dict)

            # Process each sample in batch
            outputs = []
            for sample in x_batch:
                # Convert qnode output to tensor and ensure gradients
                meas = self.qnode(all_params, sample)
                if isinstance(meas, tuple):
                    meas = torch.stack([torch.tensor(m, requires_grad=True) for m in meas])
                else:
                    meas = torch.tensor(meas, requires_grad=True, device=x_batch.device)
                
                # Transform measurements to logits and maintain gradients
                logits = torch.log1p(meas)
                outputs.append(logits)
            
            # Stack all outputs and ensure gradients are maintained
            output_tensor = torch.stack(outputs)
            output_tensor.requires_grad_(True)
            
            return output_tensor
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise 