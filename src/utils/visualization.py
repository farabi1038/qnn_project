import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Optional
import logging
import os
from pathlib import Path
import numpy as np

class TrainingVisualizer:
    """Handles visualization and metrics tracking during training"""
    
    def __init__(self, save_dir: str = "plots"):
        self.logger = logging.getLogger(__name__)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.batch_metrics: Dict[str, List[float]] = {}
        self.epoch_metrics: Dict[str, List[float]] = {}
        
        self.logger.info(f"Initialized TrainingVisualizer with save directory: {save_dir}")
        
    def update_batch_metrics(self, metrics: Dict[str, float]) -> None:
        """Update metrics for current batch"""
        try:
            for metric_name, value in metrics.items():
                if metric_name not in self.batch_metrics:
                    self.batch_metrics[metric_name] = []
                self.batch_metrics[metric_name].append(value)
        except Exception as e:
            self.logger.error(f"Error updating batch metrics: {str(e)}")
            
    def update_epoch_metrics(self, metrics: Dict[str, float]) -> None:
        """Update metrics for current epoch"""
        try:
            for metric_name, value in metrics.items():
                if metric_name not in self.epoch_metrics:
                    self.epoch_metrics[metric_name] = []
                self.epoch_metrics[metric_name].append(value)
        except Exception as e:
            self.logger.error(f"Error updating epoch metrics: {str(e)}")
            
    def plot_metrics(self, save: bool = True) -> None:
        """Plot all tracked metrics"""
        try:
            # Plot epoch metrics
            for metric_name, values in self.epoch_metrics.items():
                plt.figure(figsize=(10, 6))
                plt.plot(values, label=metric_name)
                plt.title(f'Training {metric_name} over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel(metric_name)
                plt.legend()
                plt.grid(True)
                
                if save:
                    plt.savefig(self.save_dir / f'{metric_name}_epochs.png')
                    plt.close()
                    
            # Plot batch metrics
            for metric_name, values in self.batch_metrics.items():
                plt.figure(figsize=(10, 6))
                plt.plot(values, label=metric_name)
                plt.title(f'Training {metric_name} over Batches')
                plt.xlabel('Batch')
                plt.ylabel(metric_name)
                plt.legend()
                plt.grid(True)
                
                if save:
                    plt.savefig(self.save_dir / f'{metric_name}_batches.png')
                    plt.close()
                    
            self.logger.info("Metrics plots saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error plotting metrics: {str(e)}")
            
    def save_metrics_csv(self) -> None:
        """Save all metrics to CSV files"""
        try:
            # Save epoch metrics
            if self.epoch_metrics:
                epoch_df = pd.DataFrame(self.epoch_metrics)
                epoch_df.to_csv(self.save_dir / 'epoch_metrics.csv', index=False)
                
            # Save batch metrics
            if self.batch_metrics:
                batch_df = pd.DataFrame(self.batch_metrics)
                batch_df.to_csv(self.save_dir / 'batch_metrics.csv', index=False)
                
            self.logger.info("Metrics saved to CSV successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to CSV: {str(e)}")
            
    def reset_batch_metrics(self) -> None:
        """Reset batch metrics for new epoch"""
        self.batch_metrics = {}

    def update_history(
        self,
        epoch: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        tpr: float = None,
        fpr: float = None
    ):
        """Update metrics history"""
        self.epoch_metrics['epoch'].append(epoch)
        self.epoch_metrics['loss'].append(loss)
        self.epoch_metrics['accuracy'].append(accuracy)
        self.epoch_metrics['learning_rate'].append(learning_rate)
        
        if tpr is not None:
            self.epoch_metrics['tpr'].append(tpr)
        if fpr is not None:
            self.epoch_metrics['fpr'].append(fpr)
            
    def plot_metrics_old(self, save: bool = True) -> None:
        """Plot training metrics"""
        try:
            plt.style.use('seaborn')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            df = pd.DataFrame(self.epoch_metrics)
            sns.lineplot(data=df, x='epoch', y='loss', ax=axes[0,0])
            axes[0,0].set_title('Training Loss')
            
            # Accuracy plot
            sns.lineplot(data=df, x='epoch', y='accuracy', ax=axes[0,1])
            axes[0,1].set_title('Validation Accuracy')
            
            # Learning rate plot
            sns.lineplot(data=df, x='epoch', y='learning_rate', ax=axes[1,0])
            axes[1,0].set_title('Learning Rate')
            
            # TPR/FPR plot
            if 'tpr' in df.columns and 'fpr' in df.columns:
                sns.lineplot(data=df, x='epoch', y='tpr', label='TPR', ax=axes[1,1])
                sns.lineplot(data=df, x='epoch', y='fpr', label='FPR', ax=axes[1,1])
                axes[1,1].set_title('TPR/FPR Metrics')
                axes[1,1].legend()
            
            plt.tight_layout()
            
            if save:
                filepath = os.path.join(self.save_dir, 'training_metrics.png')
                plt.savefig(filepath)
                self.logger.info(f"Saved training metrics plot to {filepath}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting metrics: {str(e)}")
            raise
            
    def save_metrics_csv_old(self) -> None:
        """Save metrics history to CSV"""
        try:
            df = pd.DataFrame(self.epoch_metrics)
            filepath = os.path.join(self.save_dir, 'metrics_history.csv')
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved metrics history to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics CSV: {str(e)}")
            raise 