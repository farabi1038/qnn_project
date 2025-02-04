import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import logging
from typing import List, Dict, Optional
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

logging.basicConfig(level=logging.INFO)

def setup_ddp(rank: int, world_size: int):
    """Setup for distributed data parallel training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class CVLayer(nn.Module):
    """Represents a single CV quantum layer with trainable parameters."""
    def __init__(self, n_wires: int, device: str = "cuda"):
        super().__init__()
        if n_wires <= 1:
            return
            
        # Initialize parameters with better scaling
        scale = 0.1  # Reduced scale for better initial convergence
        self.int1 = nn.Parameter(
            torch.randn(3 * (n_wires - 1), device=device) * scale
        )
        self.squeezes = nn.Parameter(
            torch.randn(n_wires, device=device) * scale
        )
        self.int2 = nn.Parameter(
            torch.randn(3 * (n_wires - 1), device=device) * scale
        )
        self.displacements = nn.Parameter(
            torch.randn(n_wires, device=device) * scale
        )

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            'int1': self.int1,
            'squeezes': self.squeezes,
            'int2': self.int2,
            'displacements': self.displacements
        }

class CVQNNClassifier(nn.Module):
    """Multi-GPU optimized CV-QNN classifier."""
    
    def __init__(
        self,
        in_features: int = 5,
        n_layers: int = 2,
        layer_widths: Optional[List[int]] = None,
        out_classes: int = 3,
        cutoff: int = 6,
        device: str = "cuda",
        rank: int = 0
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.rank = rank
        
        if layer_widths is None:
            layer_widths = [max(in_features, out_classes)] * n_layers
        layer_widths[0] = max(layer_widths[0], in_features)
        
        self.in_features = in_features
        self.out_classes = out_classes
        self.n_layers = n_layers
        self.layer_widths = layer_widths
        self.cutoff = cutoff
        
        # Initialize quantum layers
        self.quantum_layers = nn.ModuleList([
            CVLayer(width, device=device)
            for width in self.layer_widths
        ])
        
        # Create quantum circuits for each GPU
        self.create_circuit()
        
        if rank == 0:
            self.logger.info(
                f"Initialized Multi-GPU CVQNNClassifier\n"
                f"Input features: {in_features}\n"
                f"Number of layers: {n_layers}\n"
                f"Layer widths: {layer_widths}\n"
                f"Output classes: {out_classes}\n"
                f"Running on GPU {rank}"
            )

    def create_circuit(self):
        """Creates quantum circuits optimized for current GPU."""
        max_width = max(max(self.layer_widths), self.out_classes)
        try:
            dev = qml.device(
                "default.gaussian",
                wires=max_width
            )
        except Exception as e:
            self.logger.error(f"Failed to create quantum device on GPU {self.rank}: {e}")
            raise

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(x_input, params_dict):
            # Encode features
            for i in range(self.layer_widths[0]):
                qml.Displacement(x_input[i], 0.0, wires=i)

            # Apply CV layers with better batching
            for layer_idx, layer_params in enumerate(params_dict):
                n = self.layer_widths[layer_idx]
                if n <= 1:
                    continue

                # Group operations for better parallelization
                # Interferometer 1
                for i in range(0, n-1, 2):
                    # Parallel application for non-overlapping pairs
                    qml.Beamsplitter(
                        layer_params['int1'][3*i],
                        layer_params['int1'][3*i + 1],
                        wires=[i, i + 1]
                    )
                
                for i in range(1, n-1, 2):
                    qml.Beamsplitter(
                        layer_params['int1'][3*i],
                        layer_params['int1'][3*i + 1],
                        wires=[i, i + 1]
                    )
                
                # Grouped rotations
                for i in range(n-1):
                    qml.Rotation(layer_params['int1'][3*i + 2], wires=i)

                # Parallel squeezing
                for i in range(n):
                    qml.Squeezing(layer_params['squeezes'][i], 0.0, wires=i)

                # Interferometer 2 (same parallel structure)
                for i in range(0, n-1, 2):
                    qml.Beamsplitter(
                        layer_params['int2'][3*i],
                        layer_params['int2'][3*i + 1],
                        wires=[i, i + 1]
                    )
                
                for i in range(1, n-1, 2):
                    qml.Beamsplitter(
                        layer_params['int2'][3*i],
                        layer_params['int2'][3*i + 1],
                        wires=[i, i + 1]
                    )

                # Parallel displacements
                for i in range(n):
                    qml.Displacement(layer_params['displacements'][i], 0.0, wires=i)

            # Parallel measurement
            return [qml.expval(qml.NumberOperator(wires=w)) 
                   for w in range(min(self.out_classes, max_width))]

        self.circuit = circuit

    @torch.cuda.amp.autocast()
    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Multi-GPU optimized forward pass."""
        try:
            x_batch = x_batch.to(self.device)
            
            # Get parameters
            params_dict = [
                layer.get_parameters()
                for layer in self.quantum_layers
                if hasattr(layer, 'get_parameters')
            ]
            
            # Process local batch chunk
            batch_size = x_batch.shape[0]
            outputs = []
            
            # Parallel processing of local batch
            for i in range(batch_size):
                x_sample = x_batch[i]
                result = self.circuit(x_sample, params_dict)
                result_tensor = torch.as_tensor(result, dtype=torch.float32, device=self.device)
                outputs.append(result_tensor)
            
            # Efficient stacking
            measurements = torch.stack(outputs, dim=0)
            
            return torch.log1p(F.relu(measurements))
            
        except Exception as e:
            self.logger.error(f"Error in forward pass on GPU {self.rank}: {e}")
            raise

def train_parallel(rank, world_size, model_args):
    """Training function for distributed training."""
    setup_ddp(rank, world_size)
    
    # Create model for this GPU
    model = ParallelCVQNNClassifier(**model_args, rank=rank).to(rank)
    model = DDP(model, device_ids=[rank])
    
    return model

