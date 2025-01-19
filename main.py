import os
import yaml
import numpy as np
import torch
import cupy as cp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Union
from datetime import datetime
from tqdm import tqdm
import logging
import json
import pandas as pd
from loguru import logger

from data_preprocessing import load_cesnet_data
from anomaly_detection import AnomalyDetection
from qnn_architecture import QNNArchitecture
from continuous_qnn import ContinuousVariableQNN
from discrete_qnn import DiscreteVariableQNN
from micro_segmentation import MicroSegmentation
from zero_trust_framework import ZeroTrustFramework


class TrainingState:
    """Class to manage training state and checkpointing."""
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')
        self.patience_counter = 0
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, epoch: int, model_state: Dict, 
                       optimizer_state: Dict, metrics: Dict, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metrics': metrics,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
    def load_checkpoint(self, filename: str) -> Dict:
        """Load training checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.best_loss = checkpoint['best_loss']
            self.patience_counter = checkpoint['patience_counter']
            logger.info(f"Checkpoint loaded: {path}")
            return checkpoint
        return None
    
    def check_improvement(self, current_loss: float, 
                         patience: int = 5, min_delta: float = 0.001) -> bool:
        """Check if the model is improving."""
        if current_loss < (self.best_loss - min_delta):
            self.best_loss = current_loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return False
        return True


class GPUPipeline:
    def __init__(self, config_path: str = "config.yml"):
        """Initialize GPU-accelerated pipeline."""
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.xp = cp
        else:
            logger.info("GPU not available, using CPU")
            self.xp = np
        
        self.training_state = TrainingState()

    def create_output_dirs(self):
        """Create necessary output directories."""
        dirs = ['plots', 'results', 'models', 'logs']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")

    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def save_data(self, data: Union[np.ndarray, torch.Tensor], filename: str):
        """Save data with GPU support."""
        filepath = os.path.join('results', filename)
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        np.save(filepath, data)
        logger.info(f"Data saved to {filepath}")

    def plot_line_chart(self, x, y, xlabel, ylabel, title, filename):
        """Plot and save a line chart."""
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-', markersize=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        filepath = os.path.join('plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Line chart saved as {filepath}")

    def plot_histogram(self, data, xlabel, ylabel, title, filename, bins=30):
        """Plot and save a histogram."""
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        filepath = os.path.join('plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Histogram saved as {filepath}")

    def plot_precision_recall(self, precision, recall, filename):
        """Plot and save the Precision-Recall curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        filepath = os.path.join('plots', filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Precision-Recall curve saved as {filepath}")

    def plot_async(self, plot_functions: List[Dict]):
        """Asynchronously generate and save plots."""
        def execute_plot(plot_info):
            plt.switch_backend('Agg')  # Non-interactive backend for thread safety
            func = plot_info['function']
            args = plot_info['args']
            kwargs = plot_info['kwargs']
            func(*args, **kwargs)

        with ThreadPoolExecutor() as executor:
            executor.map(execute_plot, plot_functions)

    def to_device(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Transfer data to appropriate device."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        return data.to(self.device)

    def train_model(self, qnn_model, optimizer, X_train, y_train) -> Tuple[List[float], torch.Tensor]:
        """GPU-accelerated model training with checkpointing and error tracking."""
        X_train = self.to_device(X_train)
        y_train = self.to_device(y_train)
        
        metrics_dict = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        start_epoch = 0
        if self.config.get("training", {}).get("resume_training", False):
            checkpoint = self.training_state.load_checkpoint('latest_checkpoint.pt')
            if checkpoint:
                qnn_model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                metrics_dict = checkpoint['metrics']
                logger.info(f"Resuming training from epoch {start_epoch}")
        
        plot_interval = self.config["visualization"]["plot_interval"]
        early_stopping_patience = self.config.get("training", {}).get("early_stopping_patience", 5)
        
        progress_bar = tqdm(range(start_epoch, self.config["training"]["epochs"]),
                        initial=start_epoch,
                        total=self.config["training"]["epochs"])
        
        for epoch in progress_bar:
            output_states = qnn_model(X_train)
            cost = qnn_model.compute_cost(output_states, y_train)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            predictions = (output_states > 0.5).float()
            metrics = self.compute_metrics(y_train, predictions)
            
            metrics_dict['loss'].append(cost.item())
            metrics_dict['accuracy'].append(metrics['accuracy'])
            metrics_dict['precision'].append(metrics['precision'])
            metrics_dict['recall'].append(metrics['recall'])
            metrics_dict['f1_score'].append(metrics['f1_score'])
            
            progress_bar.set_postfix({
                'loss': f"{cost.item():.4f}",
                'accuracy': f"{metrics['accuracy']:.4f}"
            })
            
            if not self.training_state.check_improvement(cost.item(), 
                                                    patience=early_stopping_patience):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            if epoch % self.config.get("training", {}).get("checkpoint_interval", 10) == 0:
                self.training_state.save_checkpoint(
                    epoch,
                    qnn_model.state_dict(),
                    optimizer.state_dict(),
                    metrics_dict,
                    'latest_checkpoint.pt'
                )
            
            if epoch % plot_interval == 0:
                self.plot_training_metrics(metrics_dict, epoch)
                self.plot_error_tracking(metrics_dict['loss'], epoch)
        
        return metrics_dict['loss'], output_states

    def plot_error_tracking(self, losses: List[float], current_epoch: int):
        """Plot error tracking visualization."""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', label='Training Loss')
        plt.axhline(y=self.training_state.best_loss, color='r', linestyle='--',
                   label=f'Best Loss: {self.training_state.best_loss:.4f}')
        
        z = np.polyfit(range(len(losses)), losses, 1)
        p = np.poly1d(z)
        plt.plot(range(len(losses)), p(range(len(losses))), "r--", alpha=0.8,
                label=f'Trend (slope: {z[0]:.2e})')
        
        plt.title('Training Loss Tracking')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        filepath = os.path.join('plots', f'error_tracking_epoch_{current_epoch}.png')
        plt.savefig(filepath)
        plt.close()

    def plot_training_metrics(self, metrics_dict: Dict, current_epoch: int):
        """Plot training metrics."""
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        plt.figure(figsize=(12, 8))
        
        for metric in metrics_to_plot:
            plt.plot(metrics_dict[metric], label=metric.capitalize())
        
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        filepath = os.path.join('plots', f'training_metrics_epoch_{current_epoch}.png')
        plt.savefig(filepath)
        plt.close()

    def run_pipeline(self):
        """Execute the complete GPU-accelerated pipeline."""
        start_time = datetime.now()
        logger.info("Starting GPU-accelerated Quantum Anomaly Detection Pipeline")

        try:
            # Load and preprocess data
            logger.info("Loading and preprocessing data...")
            X_train, X_test, y_train, y_test = load_cesnet_data(
                self.config["data"]["csv_path"],
                test_size=self.config["data"]["test_size"],
                random_state=self.config["data"]["random_state"]
            )

            # Transfer data to GPU
            X_train, X_test = self.to_device(X_train), self.to_device(X_test)
            y_train, y_test = self.to_device(y_train), self.to_device(y_test)

            # Initialize QNN model and architecture
            logger.info("Initializing QNN model...")
            qnn_model = self.initialize_model()
            
            # Create random QNN architecture and training data
            qnn_arch = self.config["qnn_architecture"]["architecture"]
            num_training_pairs = self.config["qnn_architecture"]["num_training_pairs"]
            
            logger.info("Creating random QNN network and training data...")
            _, unitaries, training_data, _ = QNNArchitecture.random_network(qnn_arch, num_training_pairs)

            # Initialize optimizer
            optimizer = torch.optim.Adam(qnn_model.parameters(), 
                                       lr=self.config["training"]["learning_rate"])

            # Train model
            logger.info("Starting model training...")
            costs, output_states = self.train_model(qnn_model, optimizer, X_train, y_train)

            # Detect anomalies
            logger.info("Detecting anomalies...")
            anomaly_scores, predictions, metrics = self.detect_anomalies(qnn_model, X_test, y_test)

            # Initialize Zero Trust Framework and Micro-segmentation
            zt_framework = ZeroTrustFramework(
                risk_threshold=self.config["zero_trust"]["risk_threshold"]
            )
            micro_segmentation = MicroSegmentation(
                segment_threshold=self.config["zero_trust"]["segment_threshold"]
            )

            # Process Zero Trust and Micro-segmentation
            risk_scores, access_decisions = [], []
            for i, (x_sample, score) in enumerate(zip(X_test, anomaly_scores)):
                user_ctx = {"role": "user"}
                device_ctx = {"location": "local"}
                risk_score = zt_framework.compute_risk_score(user_ctx, device_ctx, score)
                access_decision = zt_framework.decide_access(risk_score)
                risk_scores.append(risk_score)
                access_decisions.append(access_decision)

            # Micro-segmentation analysis
            segment_predictions = [1 if score > self.config["anomaly"]["threshold"] 
                                else 0 for score in anomaly_scores]
            isolated_segments = micro_segmentation.isolate_segments(X_test, segment_predictions)

            # Save all results
            self.save_all_results(
                costs=costs,
                anomaly_scores=anomaly_scores,
                predictions=predictions,
                metrics=metrics,
                risk_scores=risk_scores,
                access_decisions=access_decisions,
                isolated_segments=isolated_segments
            )

            # Generate all plots
            self.create_all_plots(
                anomaly_scores=anomaly_scores,
                metrics=metrics,
                risk_scores=risk_scores,
                segment_predictions=segment_predictions
            )
            

            execution_time = datetime.now() - start_time
            logger.info(f"Pipeline execution completed successfully in {execution_time}")
            
            return metrics

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise

    def create_all_plots(self, costs, anomaly_scores, metrics, risk_scores, segment_predictions):
        """Generate all visualization plots."""
        plot_tasks = [
            {
                'function': self.plot_line_chart,
                'args': (np.arange(1, len(costs) + 1), costs),
                'kwargs': {
                    'xlabel': 'Epochs',
                    'ylabel': 'Cost',
                    'title': 'QNN Training Loss',
                    'filename': 'training_loss.png'
                }
            },
            {
                'function': self.plot_histogram,
                'args': (anomaly_scores.cpu().numpy(),),
                'kwargs': {
                    'xlabel': 'Anomaly Score',
                    'ylabel': 'Frequency',
                    'title': 'Anomaly Score Distribution',
                    'filename': 'anomaly_scores_distribution.png'
                }
            },
            {
                'function': self.plot_histogram,
                'args': (risk_scores,),
                'kwargs': {
                    'xlabel': 'Risk Score',
                    'ylabel': 'Frequency',
                    'title': 'Risk Score Distribution',
                    'filename': 'risk_scores_distribution.png'
                }
            },
            {
                'function': self.plot_histogram,
                'args': (segment_predictions,),
                'kwargs': {
                    'xlabel': 'Segment',
                    'ylabel': 'Flag Count',
                    'title': 'Segment Isolation Analysis',
                    'filename': 'segment_isolation_analysis.png'
                }
            }
        ]
        
        if 'precision' in metrics and 'recall' in metrics:
            plot_tasks.append({
                'function': self.plot_precision_recall,
                'args': (metrics['precision'], metrics['recall']),
                'kwargs': {
                    'filename': 'precision_recall_curve.png'
                }
            })

        self.plot_async(plot_tasks)

    def initialize_model(self):
        """Initialize the QNN model based on configuration."""
        if self.config["quantum"]["type"] == "discrete":
            return DiscreteVariableQNN(
                n_qubits=self.config["quantum"]["n_qubits"],
                n_layers=self.config["quantum"]["n_layers"],
                device=self.device
            )
        else:
            return ContinuousVariableQNN(
                n_qumodes=self.config["quantum"]["n_qumodes"],
                n_layers=self.config["quantum"]["n_layers"],
                cutoff_dim=self.config["quantum"]["cutoff_dim"],
                device=self.device
            )

    def save_all_results(self, costs, anomaly_scores, predictions, metrics,
                        risk_scores, access_decisions, isolated_segments):
        """Save all results to files."""
        # Save numerical results
        self.save_data(costs, 'training_costs.npy')
        self.save_data(anomaly_scores.cpu(), 'anomaly_scores.npy')
        self.save_data(predictions.cpu(), 'predictions.npy')
        self.save_data(np.array(risk_scores), 'risk_scores.npy')
        self.save_data(np.array(access_decisions), 'access_decisions.npy')
        
        # Save metrics and analysis results
        results = {
            'metrics': metrics,
            'isolated_segments': isolated_segments,
            'execution_timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join('results', 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info("All results saved successfully")


def main():
    """Main entry point for the pipeline."""
    try:
        # Record start time
        start_time = datetime.now()
        
        # Initialize and run pipeline
        logger.info("Initializing GPU Pipeline...")
        pipeline = GPUPipeline()
        
        logger.info("Starting pipeline execution...")
        metrics = pipeline.run_pipeline()
        
        # Log final results
        logger.info("Pipeline execution completed successfully")
        logger.info(f"Final metrics: {metrics}")
        logger.info(f"Total execution time: {datetime.now() - start_time}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()