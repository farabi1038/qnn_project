import torch
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from loguru import logger
from tqdm import tqdm

class TestingManager:
    def __init__(self, pipeline):
        """Initialize testing manager with pipeline reference."""
        self.pipeline = pipeline
        logger.info("Testing Manager initialized")

    def create_test_data(self, num_samples: int = 1000, num_features: int = 13) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create synthetic data for quick pipeline testing."""
        try:
            logger.debug(f"Creating synthetic data with {num_samples} samples and {num_features} features")
            
            # Generate random features
            X = torch.randn(num_samples, num_features)
            logger.debug(f"Generated feature matrix of shape: {X.shape}")
            
            # Generate labels (0 or 1) based on a simple rule
            y = (torch.sum(X, dim=1) > 0).float()
            logger.debug(f"Generated labels of shape: {y.shape}")
            
            # Split into train and test
            train_size = int(0.8 * num_samples)
            logger.debug(f"Splitting data with train_size: {train_size}")
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_test = X[train_size:]
            y_test = y[train_size:]
            
            logger.info(f"Created test data: {train_size} training samples, {num_samples - train_size} test samples")
            logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.debug(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error creating test data: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
            raise

    def evaluate_model(self, model, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        try:
            logger.debug("Starting model evaluation")
            logger.debug(f"Test data shapes - X: {X_test.shape}, y: {y_test.shape}")
            
            # Get predictions
            with torch.no_grad():
                predictions = model(X_test)
                binary_predictions = (predictions > 0.5).float()
            
            logger.debug(f"Generated predictions of shape: {predictions.shape}")
            
            # Compute metrics
            metrics = self.pipeline.compute_metrics(y_test, binary_predictions)
            logger.debug(f"Computed test metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
            raise

    def test_pipeline(self):
        """Quick test of the pipeline with synthetic data."""
        try:
            start_time = datetime.now()
            logger.info("Starting pipeline test")
            
            # Create synthetic data
            X_train, X_test, y_train, y_test = self.create_test_data(
                num_samples=1000,
                num_features=13
            )
            logger.debug(f"Created synthetic data - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
            # Initialize model and optimizer
            qnn_model = self.pipeline.initialize_model()
            optimizer = self.pipeline.initialize_optimizer(qnn_model)
            logger.debug("Model and optimizer initialized")
            
            # Modify config for quick testing
            self.pipeline.config["training"]["epochs"] = 30
            self.pipeline.config["training"]["batch_size"] = 32
            self.pipeline.config["visualization"]["plot_interval"] = 5
            logger.debug(f"Training configuration updated: {self.pipeline.config['training']}")
            
            # Train model
            logger.info("Starting model training...")
            costs, output_states = self.pipeline.train_model(qnn_model, optimizer, X_train, y_train)
            logger.debug(f"Training completed - Final cost: {costs[-1]:.4f}")
            
            # Evaluate model
            logger.info("Evaluating model performance...")
            test_metrics = self.evaluate_model(qnn_model, X_test, y_test)
            
            # Save results
            self.pipeline.save_results(
                metrics=test_metrics,
                history={'loss': costs},
                predictions=output_states,
                true_labels=y_train
            )
            
            # Save model
            self.pipeline.save_model(qnn_model, optimizer, test_metrics)
            
            # Create final plots
            logger.info("Generating final visualization plots...")
            self.pipeline.plotting.create_all_plots(
                costs=costs,
                anomaly_scores=output_states,
                metrics=test_metrics,
                risk_scores=output_states.cpu().numpy(),
                segment_predictions=y_test.cpu().numpy(),
                access_policies=['default'] * len(y_test)  # Example policies
            )
            
            execution_time = datetime.now() - start_time
            logger.info(f"Pipeline test completed in {execution_time}")
            logger.info("Test results:")
            logger.info(f"Final loss: {costs[-1]:.4f}")
            logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test precision: {test_metrics['precision']:.4f}")
            logger.info(f"Test recall: {test_metrics['recall']:.4f}")
            logger.info(f"Test F1 score: {test_metrics['f1_score']:.4f}")
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Error in test pipeline: {str(e)}")
            logger.debug(f"Exception details:", exc_info=True)
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
                
                metrics = self.test_pipeline()
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