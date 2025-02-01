import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional
import time
from pathlib import Path
import logging

class PerformanceAnalyzer:
    """Analyzes and visualizes model training performance"""
    
    def __init__(self, save_dir: str = "plots"):
        self.logger = logging.getLogger(__name__)
        self.save_dir = Path(save_dir).resolve()  # Get absolute path
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Performance Analyzer initialized. Plots will be saved to: {self.save_dir}")
        
        # Store metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.gradient_norms: Dict[str, List[float]] = {}
        self.epoch_times: List[float] = []
        self.memory_usage: List[float] = []
        self.peak_memory: List[float] = []
        
    def update_metrics(self, metrics: Dict[str, float], epoch: int):
        """Update metrics after each epoch"""
        self.train_losses.append(metrics.get('train_loss', 0))
        self.val_losses.append(metrics.get('val_loss', 0))
        self.train_accuracies.append(metrics.get('train_acc', 0))
        self.val_accuracies.append(metrics.get('val_acc', 0))
        self.epoch_times.append(metrics.get('epoch_time', 0))
        
        if torch.cuda.is_available():
            self.memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            self.peak_memory.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
    
    def compute_gradient_norms(self, model: torch.nn.Module):
        """Compute gradient norms for each layer"""
        try:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm().item()
                    if name not in self.gradient_norms:
                        self.gradient_norms[name] = []
                    self.gradient_norms[name].append(norm)
                    self.logger.debug(f"Gradient norm for {name}: {norm:.4f}")
            
            # Log total number of gradients collected
            total_grads = sum(len(norms) for norms in self.gradient_norms.values())
            self.logger.info(f"Collected gradient norms for {len(self.gradient_norms)} parameters, total records: {total_grads}")
            
        except Exception as e:
            self.logger.error(f"Error computing gradient norms: {str(e)}")
    
    def plot_training_curves(self):
        """Plot training and validation loss/accuracy curves"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Loss curves
            epochs = range(1, len(self.train_losses) + 1)
            ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
            ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy curves
            ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
            ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plot_path = self.save_dir / 'training_curves.png'
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Training curves saved to: {plot_path.absolute()}")
            
        except Exception as e:
            self.logger.error(f"Error plotting training curves: {str(e)}")
    
    def plot_gradient_norms(self):
        """Plot gradient norms across training"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Only create legend if we have gradient norms
            if self.gradient_norms:
                for name, norms in self.gradient_norms.items():
                    plt.plot(range(1, len(norms) + 1), norms, label=name)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                self.logger.warning("No gradient norms available to plot")
            
            plt.title('Gradient Norms During Training')
            plt.xlabel('Epochs')
            plt.ylabel('Gradient Norm')
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = self.save_dir / 'gradient_norms.png'
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Gradient norms plot saved to: {plot_path.absolute()}")
            
        except Exception as e:
            self.logger.error(f"Error plotting gradient norms: {str(e)}")
    
    def plot_weight_distribution(self, model: torch.nn.Module, epoch: int):
        """Plot weight distribution histograms"""
        try:
            weights = []
            layer_names = []
            
            for name, param in model.named_parameters():
                if 'weight' in name:
                    weights.append(param.data.cpu().numpy().flatten())
                    layer_names.append(name)
            
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
    
    def plot_resource_usage(self):
        """Plot computational resource usage"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Memory usage
            epochs = range(1, len(self.memory_usage) + 1)
            ax1.plot(epochs, self.memory_usage, 'b-', label='Current Memory')
            ax1.plot(epochs, self.peak_memory, 'r-', label='Peak Memory')
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
            
            self.logger.info(f"Resource usage plots saved to: {plot_path.absolute()}")
            
        except Exception as e:
            self.logger.error(f"Error plotting resource usage: {str(e)}")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            self.logger.info(f"\nGenerating plots in directory: {self.save_dir}")
            
            # Plot all metrics
            self.plot_training_curves()
            self.plot_gradient_norms()
            self.plot_resource_usage()
            
            # Calculate summary statistics
            summary = {
                'final_train_loss': self.train_losses[-1],
                'final_val_loss': self.val_losses[-1],
                'final_train_acc': self.train_accuracies[-1],
                'final_val_acc': self.val_accuracies[-1],
                'total_training_time': sum(self.epoch_times),
                'avg_epoch_time': np.mean(self.epoch_times),
                'peak_memory_usage': max(self.peak_memory) if self.peak_memory else 0
            }
            
            # Log summary and plot locations
            self.logger.info("\nTraining Performance Summary:")
            for metric, value in summary.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            self.logger.info("\nPlot files generated:")
            self.logger.info(f"Training curves: {self.save_dir/'training_curves.png'}")
            self.logger.info(f"Gradient norms: {self.save_dir/'gradient_norms.png'}")
            self.logger.info(f"Resource usage: {self.save_dir/'resource_usage.png'}")
            self.logger.info(f"Weight distributions: {self.save_dir/'weight_dist_epoch_*.png'}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {} 