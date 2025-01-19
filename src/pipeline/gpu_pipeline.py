import os
import torch
import numpy as np
import cupy as cp
from datetime import datetime
from typing import Dict, List, Tuple, Union
from loguru import logger
from tqdm import tqdm
import yaml

from src.utils import setup_logger, PlottingManager
from src.models import DiscreteVariableQNN, ContinuousVariableQNN
from src.models.training_state import TrainingState

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
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise

    def to_device(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Transfer data to appropriate device (GPU/CPU)."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        return data.to(self.device)

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Compute various classification metrics."""
        try:
            # Convert to binary predictions if needed
            if y_pred.dim() > 1:
                y_pred = (y_pred > 0.5).float()
            
            # Calculate metrics
            tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
            tn = torch.sum((y_true == 0) & (y_pred == 0)).float()
            fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
            fn = torch.sum((y_true == 1) & (y_pred == 0)).float()
            
            # Compute metrics with zero handling
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            return {
                'accuracy': accuracy.item(),
                'precision': precision.item(),
                'recall': recall.item(),
                'f1_score': f1_score.item()
            }
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

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
            X_train = self.to_device(X_train)
            y_train = self.to_device(y_train)
            logger.debug(f"Training data transferred to {self.device}")
            logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.debug(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")
            
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
            logger.debug(f"Plot interval: {plot_interval}, Early stopping patience: {early_stopping_patience}")
            
            progress_bar = tqdm(range(start_epoch, self.config["training"]["epochs"]),
                            initial=start_epoch,
                            total=self.config["training"]["epochs"])
            logger.debug(f"Starting training for {self.config['training']['epochs']} epochs")
            
            for epoch in progress_bar:
                epoch_start_time = datetime.now()
                logger.debug(f"\nStarting epoch {epoch + 1}")
                
                if isinstance(qnn_model, DiscreteVariableQNN):
                    # Process data in batches for DiscreteQNN
                    batch_size = self.config["training"]["batch_size"]
                    batch_losses = []
                    total_batches = (len(X_train) + batch_size - 1) // batch_size
                    logger.debug(f"Processing DiscreteQNN with batch size: {batch_size}")
                    logger.debug(f"Total number of batches: {total_batches}")
                    
                    for i in range(0, len(X_train), batch_size):
                        batch_start_time = datetime.now()
                        batch_num = i//batch_size + 1
                        
                        # Get batch data
                        batch_X = X_train[i:i + batch_size]
                        batch_y = y_train[i:i + batch_size]
                        logger.debug(f"\nBatch {batch_num}/{total_batches}:")
                        logger.debug(f"Batch indices: {i}:{i + batch_size}")
                        logger.debug(f"Batch X shape: {batch_X.shape}, y shape: {batch_y.shape}")
                        
                        # Forward pass
                        logger.debug("Starting forward pass...")
                        output_states = qnn_model(batch_X)
                        logger.debug(f"Forward pass completed. Output states shape: {output_states.shape}")
                        
                        # Compute cost
                        cost = qnn_model.compute_cost(output_states, batch_y)
                        batch_losses.append(cost.item())
                        logger.debug(f"Batch {batch_num} loss: {cost.item():.4f}")
                        
                        # Backward pass
                        optimizer.zero_grad()
                        cost.backward()
                        optimizer.step()
                        
                        # Update quantum parameters
                        qnn_model.update_quantum_parameters()
                        logger.debug("Quantum parameters updated")
                        
                        batch_time = datetime.now() - batch_start_time
                        logger.debug(f"Batch {batch_num} processing time: {batch_time}")
                    
                    # Compute epoch metrics
                    cost = sum(batch_losses) / len(batch_losses)
                    logger.debug(f"Epoch {epoch + 1} average loss: {cost:.4f}")
                    output_states = qnn_model(X_train)
                    
                else:
                    # Original continuous QNN forward pass
                    logger.debug("Processing ContinuousQNN")
                    output_states = qnn_model(X_train)
                    cost = qnn_model.compute_cost(output_states, y_train)
                    logger.debug(f"Epoch {epoch + 1} loss: {cost.item():.4f}")
                    
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                
                # Calculate metrics
                predictions = (output_states > 0.5).float()
                metrics = self.compute_metrics(y_train, predictions)
                logger.debug(f"Epoch {epoch + 1} metrics: {metrics}")
                
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
                    logger.debug(f"Early stopping triggered at epoch {epoch + 1}")
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
                    logger.debug(f"Checkpoint saved at epoch {epoch + 1}")
                
                # Plot training progress
                if epoch % plot_interval == 0:
                    self.plotting.plot_training_metrics(metrics_dict, epoch)
                    self.plotting.plot_error_tracking(metrics_dict['loss'], epoch)
                    logger.debug(f"Training plots updated at epoch {epoch + 1}")
                
                epoch_time = datetime.now() - epoch_start_time
                logger.debug(f"Epoch {epoch + 1} processing time: {epoch_time}")
            
            logger.debug("Training completed")
            logger.debug(f"Final metrics: {metrics_dict}")
            return metrics_dict['loss'], output_states
            
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
            raise

    def save_model(self, model, optimizer, metrics: Dict):
        """Save trained model and associated data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join('models', timestamp)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(save_dir, 'model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config,
                'timestamp': timestamp
            }, model_path)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def save_results(self, metrics: Dict, history: Dict, predictions: torch.Tensor, true_labels: torch.Tensor):
        """Save training results and predictions."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join('results', timestamp)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save metrics and history
            np.save(os.path.join(save_dir, 'metrics.npy'), metrics)
            np.save(os.path.join(save_dir, 'history.npy'), history)
            np.save(os.path.join(save_dir, 'predictions.npy'), predictions.cpu().numpy())
            np.save(os.path.join(save_dir, 'true_labels.npy'), true_labels.cpu().numpy())
            
            logger.info(f"Results saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise 