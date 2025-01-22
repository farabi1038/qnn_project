import torch
import numpy as np
import yaml
from datetime import datetime
from typing import Tuple, List, Dict
from src.utils import setup_logger
from .gpu_pipeline import GPUPipeline
import os
import json
from src.utils import PlottingManager as plot
from sklearn import metrics as sk_metrics

logger = setup_logger()

class TestingManager:
    """Manager class for testing the quantum anomaly detection pipeline."""
    
    def __init__(self, config: Dict = None):
        """Initialize the testing manager."""
        self.config = config or self._load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Testing Manager initialized with device: {self.device}")
    
    def _load_config(self, config_path: str = 'config.yml') -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            raise
    
    def create_test_data(self, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic test data."""
        # Generate random data
        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size
        
        # Generate normal data
        normal_data = torch.randn(train_size, self.config['quantum']['n_qubits'])
        
        # Normalize the data to [0,1] range
        normal_data = (normal_data - normal_data.min()) / (normal_data.max() - normal_data.min())
        
        # Create labels (0 for normal)
        normal_labels = torch.zeros(train_size)
        
        # Generate anomaly data with different distribution
        anomaly_data = torch.randn(test_size, self.config['quantum']['n_qubits']) * 1.5 + 2
        
        # Normalize anomaly data to [0,1] range
        anomaly_data = (anomaly_data - anomaly_data.min()) / (anomaly_data.max() - anomaly_data.min())
        
        # Create labels (1 for anomaly)
        anomaly_labels = torch.ones(test_size)
        
        # Combine data and labels
        X = torch.cat([normal_data, anomaly_data])
        y = torch.cat([normal_labels, anomaly_labels])
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        logger.info(f"Created test data: {train_size} training samples, {test_size} test samples")
        return X, y
    
    def train_epoch(self, pipeline: GPUPipeline, model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        optimizer.zero_grad()
        
        # Ensure data is on correct device
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Forward pass
        outputs = model(X)
        loss = torch.nn.functional.binary_cross_entropy(outputs, y.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean()
        
        return loss.item(), accuracy.item()
    
    def evaluate(self, pipeline: GPUPipeline, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the model."""
        model.eval()
        with torch.no_grad():
            # Ensure data is on correct device
            X = X.to(self.device)
            outputs = model(X)
            predictions = (outputs > 0.5).float()
        return predictions
        
    def test_pipeline(self, pipeline=None) -> Dict:
        """Test the quantum anomaly detection pipeline."""
        try:
            logger.info("Starting pipeline test")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create necessary directories
            test_dir = f"results/test_{timestamp}"
            model_dir = f"models/test_{timestamp}"
            plot_dir = f"{test_dir}/plots"
            
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)
            
            # Create test data
            X, y = self.create_test_data()
            
            # Use provided pipeline or create new one
            if pipeline is None:
                pipeline = GPUPipeline(mode='test')
            
            # Initialize model first
            model = pipeline.initialize_model()
            # Then initialize optimizer with the model
            optimizer = pipeline.initialize_optimizer(model)
            # Initialize criterion (loss function)
            pipeline.criterion = torch.nn.BCELoss()
            
            # Training parameters from config
            n_epochs = self.config.get('training', {}).get('n_epochs', 2)
            batch_size = self.config.get('training', {}).get('batch_size', 32)
            
            costs = []
            logger.info(f"Starting training for {n_epochs} epochs")
            
            # Save initial model state
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }, f"{model_dir}/initial_model.pth")
            
            for epoch in range(n_epochs):
                total_loss = 0
                n_batches = 0
                
                # Create batches
                permutation = torch.randperm(X.size()[0])
                for i in range(0, X.size()[0], batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_x, batch_y = X[indices], y[indices]
                    
                    # Forward pass with sigmoid activation
                    outputs = torch.sigmoid(pipeline.model(batch_x))
                    loss = pipeline.criterion(outputs, batch_y)
                    
                    # Backward pass
                    pipeline.optimizer.zero_grad()
                    loss.backward()
                    pipeline.optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                    costs.append(loss.item())
                    
                    if n_batches % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{n_epochs}, Batch {n_batches}, Loss: {loss.item():.4f}")
                
                avg_loss = total_loss / n_batches
                logger.info(f"Epoch {epoch+1}/{n_epochs} completed, Average Loss: {avg_loss:.4f}")
                
                # Save epoch checkpoint
                if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, f"{model_dir}/checkpoint_epoch_{epoch+1}.pth")
            
            # Evaluate model
            pipeline.model.eval()
            with torch.no_grad():
                test_outputs = torch.sigmoid(pipeline.model(X))
                test_loss = pipeline.criterion(test_outputs, y)
                predictions = (test_outputs > 0.5).float()
                accuracy = (predictions == y).float().mean()
            
            metrics = {
                'test_loss': test_loss.item(),
                'accuracy': accuracy.item(),
                'n_samples': len(X)
            }
            
            # Save all available plots
            logger.info("Generating and saving plots...")
            
            # Training metrics plots
            plot.plot_training_history({'loss': costs}, plot_dir)
            plot.plot_learning_curves(costs, plot_dir)
            plot.plot_loss_distribution(costs, plot_dir)
            
            # Model performance plots
            plot.plot_model_performance(y.cpu().numpy(), test_outputs.cpu().numpy(), plot_dir)
            plot.plot_roc_curve(y.cpu().numpy(), test_outputs.cpu().numpy(), plot_dir)
            plot.plot_precision_recall_curve(y.cpu().numpy(), test_outputs.cpu().numpy(), plot_dir)
            plot.plot_confusion_matrix(y.cpu().numpy(), (test_outputs > 0.5).cpu().numpy(), plot_dir)
            
            # Distribution plots
            plot.plot_prediction_distribution(test_outputs.cpu().numpy(), plot_dir)
            plot.plot_feature_importance(model, X.cpu().numpy(), plot_dir)
            
            # Error analysis
            errors = (predictions != y).cpu().numpy()
            plot.plot_error_analysis(X[errors].cpu().numpy(), y[errors].cpu().numpy(), plot_dir)
            
            # Threshold analysis
            plot.plot_threshold_impact(y.cpu().numpy(), test_outputs.cpu().numpy(), plot_dir)
            
            # Model architecture
            plot.plot_model_architecture(model, plot_dir)
            
            # Performance metrics heatmap
            metrics_dict = {
                'Accuracy': accuracy.item(),
                'Loss': test_loss.item(),
                'Precision': sk_metrics.precision_score(y.cpu().numpy(), predictions.cpu().numpy()),
                'Recall': sk_metrics.recall_score(y.cpu().numpy(), predictions.cpu().numpy()),
                'F1': sk_metrics.f1_score(y.cpu().numpy(), predictions.cpu().numpy())
            }
            plot.plot_metrics_heatmap(metrics_dict, plot_dir)
            
            logger.info(f"All plots saved to {plot_dir}")
            
            # Save final results
            results_file = f"{test_dir}/test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'config': self.config,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=4)
            
            # Save final model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }, f"{model_dir}/final_model.pth")
            
            logger.info(f"Testing completed. Results saved to {test_dir}")
            logger.info(f"Models saved to {model_dir}")
            logger.info(f"Plots saved to {plot_dir}")
            
            return metrics, costs, test_outputs.cpu(), y.cpu()
            
        except Exception as e:
            logger.error(f"Error in test pipeline: {str(e)}")
            raise

    def run_performance_test(self, num_iterations: int = 5):
            """Run performance testing with multiple iterations."""
            try:
                logger.info(f"Starting performance test with {num_iterations} iterations")
                
                metrics_history = []
                execution_times = []
                
                for i in range(num_iterations):
                    logger.info(f"Starting iteration {i+1}/{num_iterations}")
                    start_time = datetime.now()
                    
                    metrics, costs, outputs, y_true = self.test_pipeline()
                    metrics_history.append(metrics)
                    
                    execution_time = datetime.now() - start_time
                    execution_times.append(execution_time.total_seconds())
                    
                    logger.info(f"Iteration {i+1} completed in {execution_time}")
                
                # Compute average metrics
                avg_metrics = {
                    key: np.mean([m[key] for m in metrics_history])
                    for key in metrics_history[0].keys()
                }
                
                logger.info("Performance test results:")
                logger.info(f"Average execution time: {np.mean(execution_times):.2f} seconds")
                logger.info(f"Average metrics: {avg_metrics}")
                
                return avg_metrics, execution_times
                
            except Exception as e:
                logger.error(f"Error in performance test: {str(e)}")
                logger.debug("Exception details:", exc_info=True)
                raise 