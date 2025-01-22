import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from loguru import logger
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

class PlottingManager:
    def __init__(self):
        """Initialize plotting manager."""
        # Create necessary directories
        os.makedirs('plots', exist_ok=True)
        os.makedirs(os.path.join('plots', 'training_progress'), exist_ok=True)
        
    def plot_error_tracking(self, losses: List[float], current_epoch: int):
        """Plot error tracking visualization."""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(losses, 'b-', label='Training Loss')
            plt.axhline(y=min(losses), color='r', linestyle='--',
                       label=f'Best Loss: {min(losses):.4f}')
            
            if len(losses) > 2:
                try:
                    z = np.polyfit(range(len(losses)), losses, 1)
                    p = np.poly1d(z)
                    plt.plot(range(len(losses)), p(range(len(losses))), "r--", alpha=0.8,
                            label=f'Trend (slope: {z[0]:.2e})')
                except Exception as e:
                    logger.warning(f"Could not plot trend line: {str(e)}")
            
            plt.title('Training Loss Tracking')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            filepath = os.path.join('plots', f'error_tracking_epoch_{current_epoch}.png')
            plt.savefig(filepath)
            plt.close()
            logger.debug(f"Error tracking plot saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error in plot_error_tracking: {str(e)}")
            plt.close()

    def plot_training_metrics(self, metrics_dict: Dict, current_epoch: int):
        """Plot training metrics."""
        try:
            if not all(len(metrics_dict[metric]) > 0 for metric in ['accuracy', 'precision', 'recall', 'f1_score']):
                logger.warning("Not enough data to plot training metrics")
                return
            
            plt.figure(figsize=(12, 8))
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in metrics_to_plot:
                if metric in metrics_dict and len(metrics_dict[metric]) > 0:
                    plt.plot(metrics_dict[metric], label=metric.capitalize())
            
            plt.title('Training Metrics')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            filepath = os.path.join('plots', f'training_metrics_epoch_{current_epoch}.png')
            plt.savefig(filepath)
            plt.close()
            logger.debug(f"Training metrics plot saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error in plot_training_metrics: {str(e)}")
            plt.close()

    def plot_line_chart(self, x, y, xlabel, ylabel, title, filename):
        """Plot and save a line chart."""
        try:
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
            
        except Exception as e:
            logger.error(f"Error in plot_line_chart: {str(e)}")
            plt.close()

    def plot_histogram(self, data, xlabel, ylabel, title, filename, bins=30):
        """Plot and save a histogram."""
        try:
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
            
        except Exception as e:
            logger.error(f"Error in plot_histogram: {str(e)}")
            plt.close()

    def plot_precision_recall(self, precision, recall, filename):
        """Plot and save the Precision-Recall curve."""
        try:
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
            
        except Exception as e:
            logger.error(f"Error in plot_precision_recall: {str(e)}")
            plt.close()

    def plot_policy_distribution(self, policies: List[str]):
        """Plot the distribution of access policies."""
        try:
            policy_counts = {}
            for policy in policies:
                policy_counts[policy] = policy_counts.get(policy, 0) + 1
            
            plt.figure(figsize=(10, 6))
            plt.bar(policy_counts.keys(), policy_counts.values())
            plt.title('Distribution of Access Policies')
            plt.xlabel('Policy Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            filepath = os.path.join('plots', 'policy_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Policy distribution plot saved as {filepath}")
            
        except Exception as e:
            logger.error(f"Error in plot_policy_distribution: {str(e)}")
            plt.close()

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

    def create_all_plots(self, costs, anomaly_scores, metrics, risk_scores, 
                        segment_predictions, access_policies):
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

    @staticmethod
    def plot_training_history(history, save_dir):
        """Plot training metrics history."""
        plt.figure(figsize=(10, 6))
        for metric, values in history.items():
            plt.plot(values, label=metric)
        plt.title('Training History')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()

    @staticmethod
    def plot_learning_curves(costs, save_dir):
        """Plot learning curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(costs)
        plt.title('Learning Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
        plt.close()

    @staticmethod
    def plot_loss_distribution(costs, save_dir):
        """Plot distribution of loss values."""
        plt.figure(figsize=(10, 6))
        sns.histplot(costs, kde=True)
        plt.title('Loss Distribution')
        plt.xlabel('Loss Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'loss_distribution.png'))
        plt.close()

    @staticmethod
    def plot_model_performance(y_true, y_pred, save_dir):
        """Plot model performance metrics."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title('Model Performance')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.savefig(os.path.join(save_dir, 'model_performance.png'))
        plt.close()

    @staticmethod
    def plot_roc_curve(y_true, y_pred, save_dir):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()

    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred, save_dir):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_dir):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

    @staticmethod
    def plot_prediction_distribution(predictions, save_dir):
        """Plot distribution of predictions."""
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions, kde=True)
        plt.title('Prediction Distribution')
        plt.xlabel('Prediction Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, 'prediction_distribution.png'))
        plt.close()

    @staticmethod
    def plot_feature_importance(model, X, save_dir):
        """Plot feature importance if available."""
        try:
            importances = model.feature_importances_
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances)
            plt.title('Feature Importance')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
            plt.close()
        except:
            pass  # Skip if feature importance is not available

    @staticmethod
    def plot_error_analysis(X_errors, y_errors, save_dir):
        """Plot error analysis."""
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(X_errors)), X_errors[:, 0], c=y_errors)
        plt.title('Error Analysis')
        plt.xlabel('Sample Index')
        plt.ylabel('Feature Value')
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'))
        plt.close()

    @staticmethod
    def plot_threshold_impact(y_true, y_pred, save_dir):
        """Plot impact of different decision thresholds."""
        thresholds = np.linspace(0, 1, 100)
        accuracies = []
        for thresh in thresholds:
            accuracy = np.mean((y_pred > thresh) == y_true)
            accuracies.append(accuracy)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies)
        plt.title('Threshold Impact on Accuracy')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(save_dir, 'threshold_impact.png'))
        plt.close()

    @staticmethod
    def plot_model_architecture(model, save_dir):
        """Plot model architecture summary."""
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, str(model), ha='center', va='center')
        plt.axis('off')
        plt.title('Model Architecture')
        plt.savefig(os.path.join(save_dir, 'model_architecture.png'))
        plt.close()

    @staticmethod
    def plot_metrics_heatmap(metrics_dict, save_dir):
        """Plot heatmap of performance metrics."""
        metrics_array = np.array(list(metrics_dict.values())).reshape(1, -1)
        plt.figure(figsize=(12, 4))
        sns.heatmap(metrics_array, annot=True, fmt='.3f', xticklabels=list(metrics_dict.keys()))
        plt.title('Performance Metrics')
        plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'))
        plt.close() 