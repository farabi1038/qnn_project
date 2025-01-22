import os
import torch
import numpy as np
import cupy as cp
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any
from loguru import logger
from tqdm import tqdm
import yaml
import json

from src.utils import setup_logger, PlottingManager
from src.models import DiscreteVariableQNN, ContinuousVariableQNN
from src.models.training_state import TrainingState
from src.utils import plotting

logger = setup_logger()

class GPUPipeline:
    def __init__(self, config_path: str = "config.yml"):
        """Initialize GPU-accelerated pipeline."""
        try:
            # Create necessary directories first
            self.create_output_dirs()
            
            # Load configuration
            self.config = self._load_config(config_path)
            logger.info(f"Configuration loaded from {config_path}")

            # Validate required configuration
            if 'data' not in self.config:
                raise ValueError("Missing 'data' section in config")
            if 'csv_path' not in self.config['data']:
                raise ValueError("Missing 'csv_path' in data config")
                
            # Setup device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                self.xp = cp
            else:
                logger.info("GPU not available, using CPU")
                self.xp = np
            
            self.training_state = TrainingState()
            self.plotting = PlottingManager()
            self.execution_mode = self.config.get('execution_mode', 'test')
            logger.info(f"GPU Pipeline initialized with device: {self.device}")
            logger.info(f"Execution mode: {self.execution_mode}")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise

    def create_output_dirs(self):
        """Create necessary output directories."""
        try:
            dirs = ['models', 'results', 'plots', 'checkpoints']
            for dir_name in dirs:
                os.makedirs(dir_name, exist_ok=True)
                logger.debug(f"Created/verified directory: {dir_name}")
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Set default execution mode if not specified
                if 'execution_mode' not in config:
                    config['execution_mode'] = 'test'
                return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def to_device(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Transfer data to appropriate device (GPU/CPU)."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        return data.to(self.device)

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics."""
        try:
            # Ensure both tensors are on the same device
            y_true = y_true.to(self.device)
            y_pred = y_pred.to(self.device)

            # Detach tensors and move to CPU for numpy operations
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()

            # Compute metrics
            metrics = {
                'accuracy': float(np.mean(y_true_np == y_pred_np)),
                'precision': float(np.sum((y_pred_np == 1) & (y_true_np == 1)) / (np.sum(y_pred_np == 1) + 1e-8)),
                'recall': float(np.sum((y_pred_np == 1) & (y_true_np == 1)) / (np.sum(y_true_np == 1) + 1e-8))
            }
            
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-8)
            
            logger.info(f"Metrics computed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise

    def initialize_model(self) -> Union[DiscreteVariableQNN, ContinuousVariableQNN]:
        """Initialize the QNN model based on configuration."""
        try:
            if self.config["quantum"]["type"] == "discrete":
                self.model = DiscreteVariableQNN(
                    n_qubits=self.config["quantum"]["n_qubits"],
                    n_layers=self.config["quantum"]["n_layers"],
                    device=self.device
                )
            else:
                self.model = ContinuousVariableQNN(
                    n_qumodes=self.config["quantum"]["n_qumodes"],
                    n_layers=self.config["quantum"]["n_layers"],
                    cutoff_dim=self.config["quantum"]["cutoff_dim"],
                    device=self.device
                )
            
            logger.info(f"Model initialized: {self.config['quantum']['type']} QNN")
            return self.model
            
        except KeyError as e:
            logger.error(f"Missing configuration key: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def initialize_optimizer(self, model: Union[DiscreteVariableQNN, ContinuousVariableQNN]) -> torch.optim.Optimizer:
        """Initialize the optimizer for the QNN model."""
        try:
            if model is None:
                raise ValueError("Model cannot be None")
            
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config["training"]["learning_rate"]
            )
            
            logger.info(f"Optimizer initialized with learning rate: {self.config['training']['learning_rate']}")
            return self.optimizer
            
        except KeyError as e:
            logger.error(f"Missing configuration key: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing optimizer: {str(e)}")
            raise

    def train_model(self, qnn_model, optimizer, X_train, y_train) -> Tuple[List[float], torch.Tensor]:
        """GPU-accelerated model training with checkpointing and error tracking."""
        try:
            logger.info("Training model...")
            logger.info(f"Type of X_train is {type(X_train)}, type of y_train is {type(y_train)}")
            
            # Handle case where y_train is None by generating dummy labels
            if y_train is None:
                logger.info("No labels provided, generating dummy labels for unsupervised learning")
                y_train = np.zeros(len(X_train), dtype=np.float32)  # Default to all zeros
            
            X_train = self.to_device(X_train)
            y_train = self.to_device(y_train)
            
            logger.info(f"Training data transferred to {self.device}")
            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")
            
            metrics_dict = {
                'loss': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }
            
            # Handle training resumption
            start_epoch = 0
            if self.config.get("training", {}).get("resume_training", False):
                checkpoint = self.training_state.load_checkpoint('latest_checkpoint.pt')
                if checkpoint:
                    qnn_model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    metrics_dict = checkpoint['metrics']
                    logger.debug(f"Resumed training from epoch {start_epoch}")
                    logger.debug(f"Loaded checkpoint metrics: {metrics_dict}")
            
            plot_interval = self.config["visualization"]["plot_interval"]
            early_stopping_patience = self.config.get("training", {}).get("early_stopping_patience", 5)
            logger.info(f"Plot interval: {plot_interval}, Early stopping patience: {early_stopping_patience}")
            
            progress_bar = tqdm(range(start_epoch, self.config["training"]["epochs"]),
                            initial=start_epoch,
                            total=self.config["training"]["epochs"])
            logger.info(f"Starting training for {self.config['training']['epochs']} epochs")
            
            for epoch in progress_bar:
                epoch_start_time = datetime.now()
                logger.info(f"\nStarting epoch {epoch + 1}")
                
                if isinstance(qnn_model, DiscreteVariableQNN):
                    # Process data in batches for DiscreteQNN
                    batch_size = self.config["training"]["batch_size"]
                    batch_losses = []
                    total_batches = (len(X_train) + batch_size - 1) // batch_size
                    logger.info(f"Processing DiscreteQNN with batch size: {batch_size}")
                    logger.info(f"Total number of batches: {total_batches}")
                    
                    for i in range(0, len(X_train), batch_size):
                        batch_start_time = datetime.now()
                        batch_num = i//batch_size + 1
                        
                        # Get batch data
                        batch_X = X_train[i:i + batch_size]
                        batch_y = y_train[i:i + batch_size]
                        logger.info(f"\nBatch {batch_num}/{total_batches}:")
                        logger.info(f"Batch indices: {i}:{i + batch_size}")
                        logger.info(f"Batch X shape: {batch_X.shape}, y shape: {batch_y.shape}")
                        
                        # Forward pass
                        logger.info("Starting forward pass...")
                        output_states = qnn_model(batch_X)
                        logger.info(f"Forward pass completed. Output states shape: {output_states.shape}")
                        
                        # Compute cost
                        cost = qnn_model.compute_cost(output_states, batch_y)
                        batch_losses.append(cost.item())
                        logger.info(f"Batch {batch_num} loss: {cost.item():.4f}")
                        
                        # Backward pass
                        optimizer.zero_grad()
                        cost.backward()
                        optimizer.step()
                        
                        # Update quantum parameters
                        qnn_model.update_quantum_parameters()
                        logger.info("Quantum parameters updated")
                        
                        batch_time = datetime.now() - batch_start_time
                        logger.info(f"Batch {batch_num} processing time: {batch_time}")
                    
                    # Compute epoch metrics
                    cost = sum(batch_losses) / len(batch_losses)
                    logger.info(f"Epoch {epoch + 1} average loss: {cost:.4f}")
                    output_states = qnn_model(X_train)
                    
                else:
                    # Original continuous QNN forward pass
                    logger.info("Processing ContinuousQNN")
                    output_states = qnn_model(X_train)
                    cost = qnn_model.compute_cost(output_states, y_train)
                    logger.info(f"Epoch {epoch + 1} loss: {cost.item():.4f}")
                    
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                
                # Calculate metrics
                predictions = (output_states > 0.5).float()
                metrics = self.compute_metrics(y_train, predictions)
                logger.info(f"Epoch {epoch + 1} metrics: {metrics}")
                
                # Update metrics dictionary
                metrics_dict['loss'].append(cost if isinstance(cost, float) else cost.item())
                metrics_dict['accuracy'].append(metrics['accuracy'])
                metrics_dict['precision'].append(metrics['precision'])
                metrics_dict['recall'].append(metrics['recall'])
                metrics_dict['f1_score'].append(metrics['f1_score'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{cost if isinstance(cost, float) else cost.item():.4f}",
                    'accuracy': f"{metrics['accuracy']:.4f}"
                })
                
                # Check for early stopping
                cost_value = cost if isinstance(cost, float) else cost.item()
                if not self.training_state.check_improvement(cost_value, patience=early_stopping_patience):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                # Save checkpoint
                if epoch % self.config.get("training", {}).get("checkpoint_interval", 10) == 0:
                    self.training_state.save_checkpoint(
                        epoch,
                        qnn_model.state_dict(),
                        optimizer.state_dict(),
                        metrics_dict,
                        'latest_checkpoint.pt'
                    )
                    logger.info(f"Checkpoint saved at epoch {epoch + 1}")
                
                # Plot training progress
                if epoch % plot_interval == 0:
                    self.plotting.plot_training_metrics(metrics_dict, epoch)
                    self.plotting.plot_error_tracking(metrics_dict['loss'], epoch)
                    logger.info(f"Training plots updated at epoch {epoch + 1}")
                
                epoch_time = datetime.now() - epoch_start_time
                logger.info(f"Epoch {epoch + 1} processing time: {epoch_time}")
            
            logger.info("Training completed")
            logger.info(f"Final metrics: {metrics_dict}")
            return metrics_dict['loss'], output_states
            
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
            logger.info("Exception details:", exc_info=True)
            raise

    def save_model(self, model, optimizer, metrics, path=None):
        """Save the model, optimizer state, and metrics."""
        try:
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"models/model_{timestamp}.pth"
                
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
            
            # Save model summary
            summary_path = path.replace('.pth', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Model Summary\n{'-'*50}\n")
                f.write(f"Timestamp: {checkpoint['timestamp']}\n")
                f.write(f"Metrics: {metrics}\n")
                f.write(f"Config: {self.config}\n")
            
            return path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def save_results(self, metrics, history, outputs, y_true):
        """Save all results including metrics, model, and plots."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = f"results/run_{timestamp}"
            os.makedirs(base_path, exist_ok=True)
            
            # Save metrics to JSON
            metrics_path = f"{base_path}/metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Use plotting module to save plots
            plots_path = f"{base_path}/plots"
            os.makedirs(plots_path, exist_ok=True)
            
            # Generate plots using plotting module
            plotting.plot_training_loss(history['loss'], plots_path)
            plotting.plot_roc_curve(y_true, outputs, plots_path)
            plotting.plot_confusion_matrix(y_true, outputs, plots_path)
            
            # Save results summary
            summary_path = f"{base_path}/summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Results Summary\n{'-'*50}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Metrics: {metrics}\n")
                f.write(f"Plots saved to: {plots_path}\n")
            
            logger.info(f"All results saved to {base_path}")
            return base_path
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def get_execution_mode(self) -> str:
        """Get the current execution mode."""
        return self.execution_mode

    def set_execution_mode(self, mode: str) -> None:
        """Set the execution mode."""
        valid_modes = ['train', 'test', 'evaluate']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of {valid_modes}")
        self.execution_mode = mode
        logger.info(f"Execution mode set to: {mode}")

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj) 