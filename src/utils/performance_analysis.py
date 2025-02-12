
################################################################################
# PERFORMANCE ANALYZER
################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from typing import Dict, List
from pathlib import Path
import logging

class PerformanceAnalyzer:
    """Analyzes and visualizes model training performance with advanced verification techniques."""
    def __init__(self, save_dir: str = "plots"):
        self.logger = logging.getLogger(__name__)
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Performance Analyzer initialized. Plots will be saved to: {self.save_dir}")

        # Metrics storage
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []
        self.gradient_norms = {}  # stores gradient norm lists for each parameter
        self.epoch_times, self.memory_usage, self.peak_memory = [], [], []
        self.layer_weights = {}
        self.predictions, self.true_labels = [], []

    def update_metrics(self, metrics: dict, epoch: int):
        """Store loss, accuracy, and resource usage after each epoch."""
        self.train_losses.append(metrics.get('train_loss', 0))
        self.val_losses.append(metrics.get('val_loss', 0))
        self.train_accuracies.append(metrics.get('train_acc', 0))
        self.val_accuracies.append(metrics.get('val_acc', 0))
        self.epoch_times.append(metrics.get('epoch_time', 0))
        if torch.cuda.is_available():
            self.memory_usage.append(torch.cuda.memory_allocated() / 1024**2)
            self.peak_memory.append(torch.cuda.max_memory_allocated() / 1024**2)

    def store_predictions(self, y_true, y_pred):
        """Store model predictions and true labels for later analysis."""
        self.true_labels.extend(y_true)
        self.predictions.extend(y_pred)

    def compute_and_plot_gradient_norms(self, model: torch.nn.Module):
        """Compute and plot gradient norms for each parameter during training."""
        try:
            # Compute gradient norms for each parameter
            for name, param in model.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm().item()
                    if name not in self.gradient_norms:
                        self.gradient_norms[name] = []
                    self.gradient_norms[name].append(norm)
                    self.logger.debug(f"Gradient norm for {name}: {norm:.4f}")

            total_records = sum(len(norms) for norms in self.gradient_norms.values())
            self.logger.info(f"Collected gradient norms for {len(self.gradient_norms)} parameters, total records: {total_records}")

            if not self.gradient_norms:
                self.logger.warning("No gradient norms available to plot.")
                return

            # Plot all gradient norms
            plt.figure(figsize=(12, 6))
            for name, norms in self.gradient_norms.items():
                plt.plot(range(1, len(norms) + 1), norms, label=name)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title('Gradient Norms During Training')
            plt.xlabel('Iterations')
            plt.ylabel('Gradient Norm')
            plt.grid(True)
            plt.tight_layout()
            plot_path = self.save_dir / 'gradient_norms.png'
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Gradient norms plot saved to: {plot_path.absolute()}")
        except Exception as e:
            self.logger.error(f"Error computing or plotting gradient norms: {str(e)}")

    def plot_training_curves(self):
        """Plot training and validation loss/accuracy curves."""
        plt.figure(figsize=(14, 6))
        epochs = range(1, len(self.train_losses) + 1)
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', marker='o')
        plt.plot(epochs, self.val_losses, label='Validation Loss', marker='s')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy', marker='o')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy', marker='s')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        training_curve_path = self.save_dir / 'training_curves.png'
        plt.savefig(training_curve_path)
        plt.close()
        self.logger.info(f"Training curves saved to: {training_curve_path.absolute()}")

    def plot_weight_distribution(self, model: torch.nn.Module, epoch: int):
        """Plot the distribution of model weights at a given epoch."""
        try:
            weights = []
            layer_names = []
            for name, param in model.named_parameters():
                if 'weight' in name:
                    weights.append(param.data.cpu().numpy().flatten())
                    layer_names.append(name)
            if not weights:
                self.logger.warning("No weights found for distribution plot.")
                return
            plt.figure(figsize=(12, 6))
            for w, name in zip(weights, layer_names):
                plt.hist(w, bins=50, alpha=0.5, label=name)
            plt.title(f'Weight Distribution at Epoch {epoch}')
            plt.xlabel('Weight Value')
            plt.ylabel('Count')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plot_path = self.save_dir / f'weight_dist_epoch_{epoch}.png'
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Weight distribution plot for epoch {epoch} saved to: {plot_path.absolute()}")
        except Exception as e:
            self.logger.error(f"Error plotting weight distribution: {str(e)}")

    def plot_confusion_matrix(self):
        """Plot a confusion matrix from stored predictions and true labels."""
        try:
            if not self.predictions or not self.true_labels:
                self.logger.warning("No predictions available to plot confusion matrix.")
                return
            cm = confusion_matrix(self.true_labels, self.predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plot_path = self.save_dir / 'confusion_matrix.png'
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Confusion matrix saved to: {plot_path.absolute()}")
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrix: {str(e)}")

    def plot_resource_usage(self):
        """Plot GPU memory usage and training time per epoch."""
        try:
            if not self.memory_usage:
                self.logger.warning("No resource usage data available to plot.")
                return
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            epochs = range(1, len(self.memory_usage) + 1)
            # Memory usage
            ax1.plot(epochs, self.memory_usage, 'b-', label='Current Memory (MB)')
            ax1.plot(epochs, self.peak_memory, 'r-', label='Peak Memory (MB)')
            ax1.set_title('GPU Memory Usage')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Memory (MB)')
            ax1.legend()
            ax1.grid(True)
            # Training time
            ax2.plot(epochs, self.epoch_times, 'g-')
            ax2.set_title('Training Time per Epoch')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Time (seconds)')
            ax2.grid(True)
            plt.tight_layout()
            plot_path = self.save_dir / 'resource_usage.png'
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Resource usage plot saved to: {plot_path.absolute()}")
        except Exception as e:
            self.logger.error(f"Error plotting resource usage: {str(e)}")

    def generate_performance_report(self, model: torch.nn.Module, epoch: int):
        """
        Generate a comprehensive performance report that re-plots key metrics and outputs summary statistics.
        """
        try:
            self.logger.info("Generating full performance report...")
            self.plot_training_curves()
            self.compute_and_plot_gradient_norms(model)
            self.plot_weight_distribution(model, epoch)
            self.plot_confusion_matrix()

            summary = {
                'Final Train Loss': self.train_losses[-1] if self.train_losses else None,
                'Final Val Loss': self.val_losses[-1] if self.val_losses else None,
                'Final Train Accuracy': self.train_accuracies[-1] if self.train_accuracies else None,
                'Final Val Accuracy': self.val_accuracies[-1] if self.val_accuracies else None,
                'Total Training Time (s)': sum(self.epoch_times),
                'Avg Epoch Time (s)': np.mean(self.epoch_times) if self.epoch_times else None,
                'Peak GPU Memory (MB)': max(self.peak_memory) if self.peak_memory else 0
            }

            for metric, value in summary.items():
                self.logger.info(f"{metric}: {value:.4f}" if value is not None else f"{metric}: N/A")

            self.logger.info("Report and plots saved successfully!")
            return summary
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {}
