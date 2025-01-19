import os
import yaml
import numpy as np
import torch
import cupy as cp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Union
from logger import logger
from data_preprocessing import load_cesnet_data
from anomaly_detection import AnomalyDetection
from qnn_architecture import QNNArchitecture
from continuous_qnn import ContinuousVariableQNN
from discrete_qnn import DiscreteVariableQNN
from micro_segmentation import MicroSegmentation
from zero_trust_framework import ZeroTrustFramework


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

    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def save_data(self, data: Union[np.ndarray, torch.Tensor], filename: str):
        """Save data with GPU support."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        np.save(filename, data)
        logger.info(f"Data saved to {filename}")

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

    def train_model(self, qnn_model, X_train, y_train) -> Tuple[List[float], torch.Tensor]:
        """GPU-accelerated model training."""
        X_train = self.to_device(X_train)
        y_train = self.to_device(y_train)
        
        costs = []
        for epoch in range(self.config["training"]["epochs"]):
            output_states = qnn_model.forward_pass(X_train)
            cost = qnn_model.compute_cost(output_states, y_train)
            costs.append(cost.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Cost = {cost.item():.4f}")
        
        return costs, output_states

    def run_pipeline(self):
        """Execute the complete GPU-accelerated pipeline."""
        logger.info("Starting GPU-accelerated Quantum Anomaly Detection Pipeline")

        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_cesnet_data(
            self.config["data"]["csv_path"],
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_state"]
        )

        # Transfer data to GPU
        X_train, X_test = self.to_device(X_train), self.to_device(X_test)
        y_train, y_test = self.to_device(y_train), self.to_device(y_test)

        # Initialize QNN model
        qnn_model = (DiscreteVariableQNN if self.config["quantum"]["type"] == "discrete" 
                    else ContinuousVariableQNN)(
            n_qubits=self.config["quantum"]["n_qubits"],
            n_layers=self.config["quantum"]["n_layers"],
            device=self.device
        )

        # Train model
        costs, output_states = self.train_model(qnn_model, X_train, y_train)

        # Prepare plotting tasks
        plot_tasks = [
            {
                'function': plot_line_chart,
                'args': (np.arange(1, len(costs) + 1), costs),
                'kwargs': {
                    'xlabel': 'Epochs',
                    'ylabel': 'Cost',
                    'title': 'QNN Training Loss',
                    'filename': 'training_loss.png'
                }
            }
            # Add more plotting tasks as needed
        ]

        # Execute plots asynchronously
        self.plot_async(plot_tasks)

        # Detect anomalies
        anomaly_scores = torch.tensor([
            qnn_model.forward_pass(x) for x in X_test
        ], device=self.device)

        # Optimize threshold
        best_threshold = AnomalyDetection.adjust_threshold(
            cp.asarray(anomaly_scores.cpu()) if self.device.type == 'cuda' else anomaly_scores.numpy(),
            self.config["anomaly"]["percentile"]
        )

        # Compute metrics
        predictions = (anomaly_scores > best_threshold).float()
        metrics = self.compute_metrics(y_test, predictions)

        # Save results
        self.save_results(costs, anomaly_scores, predictions, metrics)

        logger.info("Pipeline execution completed successfully")
        return metrics

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict:
        """Compute performance metrics on GPU."""
        tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
        fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
        tn = torch.sum((y_true == 0) & (y_pred == 0)).float()
        fn = torch.sum((y_true == 1) & (y_pred == 0)).float()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1_score.item(),
            'accuracy': accuracy.item()
        }

    def save_results(self, costs: List[float], anomaly_scores: torch.Tensor,
                    predictions: torch.Tensor, metrics: Dict):
        """Save all results and generate plots."""
        # Save numerical results
        self.save_data(costs, 'training_costs.npy')
        self.save_data(anomaly_scores.cpu(), 'anomaly_scores.npy')
        self.save_data(predictions.cpu(), 'predictions.npy')

        # Save metrics
        with open('metrics.yml', 'w') as f:
            yaml.dump(metrics, f)

        logger.info("Results saved successfully")


def main():
    pipeline = GPUPipeline()
    metrics = pipeline.run_pipeline()
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()