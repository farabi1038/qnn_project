import torch
import numpy as np
from typing import Tuple
import logging
from sklearn.metrics import accuracy_score
from .checkpoint import CheckpointManager
from utils.visualization import TrainingVisualizer
from tqdm import tqdm
import time

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        checkpoint_manager: CheckpointManager,
        visualizer: TrainingVisualizer,
        optimizer: torch.optim.Optimizer = None,
        criterion: torch.nn.Module = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # Setup device and GPU info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.logger.info(
                f"CUDA is available. Using GPU:\n"
                f"  Device: {torch.cuda.get_device_name(0)}\n"
                f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB\n"
                f"  Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB\n"
                f"  Device Count: {torch.cuda.device_count()}"
            )
        else:
            self.logger.warning("CUDA is not available. Using CPU.")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Log model parameters count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(
            f"Model Parameters:\n"
            f"  Total: {total_params:,}\n"
            f"  Trainable: {trainable_params:,}"
        )
        
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.visualizer = visualizer
        
        # Setup optimizer and criterion if not provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get('learning_rate', 0.001)
            )
        else:
            self.optimizer = optimizer
            
        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
            
        # Move criterion to device
        self.criterion = self.criterion.to(self.device)
        
        self.current_epoch = 0
        self.best_accuracy = 0.0
        
        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.logger.info(
            f"Performance Optimizations:\n"
            f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}\n"
            f"  TF32 Enabled: {torch.backends.cudnn.allow_tf32}"
        )
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        correct_predictions = 0
        total_samples = 0
        epoch_start_time = time.time()
        
        # Log GPU memory at start of epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.logger.info(
                f"GPU Memory at epoch start:\n"
                f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB\n"
                f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
            )
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}', 
                   leave=False, ncols=100)
        
        for batch_idx, (data, target) in enumerate(pbar):
            try:
                batch_start_time = time.time()
                
                # Move data to device
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass without mixed precision for quantum circuit
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass (without gradient scaling)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(target).sum().item()
                    correct_predictions += correct
                    total_samples += target.size(0)
                
                # Update metrics
                total_loss += loss.item()
                batch_acc = correct / target.size(0)
                
                # Update progress bar
                batch_time = time.time() - batch_start_time
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}',
                    'time': f'{batch_time:.2f}s'
                })
                
                # Log detailed batch information every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(
                        f"Epoch {self.current_epoch + 1} | "
                        f"Batch {batch_idx + 1}/{n_batches} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Accuracy: {batch_acc:.4f} | "
                        f"Time: {batch_time:.2f}s"
                    )
                
                # Log GPU memory usage periodically
                if (batch_idx + 1) % 50 == 0 and torch.cuda.is_available():
                    self.logger.info(
                        f"GPU Memory after batch {batch_idx + 1}:\n"
                        f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB\n"
                        f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n"
                        f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate epoch metrics
        epoch_loss = total_loss / n_batches if n_batches > 0 else float('inf')
        epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch summary
        self.logger.info(
            f"\nEpoch {self.current_epoch + 1} Summary:\n"
            f"  Average Loss: {epoch_loss:.4f}\n"
            f"  Average Accuracy: {epoch_acc:.4f}\n"
            f"  Total Time: {epoch_time:.2f}s\n"
            f"  Samples Processed: {total_samples}\n"
            f"  Learning Rate: {self.optimizer.param_groups[0]['lr']}"
        )
        
        return epoch_loss
        
    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float, np.ndarray]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = len(val_loader)
        eval_start_time = time.time()
        
        # Create progress bar for evaluation
        pbar = tqdm(val_loader, desc='Evaluating', leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                try:
                    # Move data to device
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    # Forward pass
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    # Get predictions
                    preds = output.argmax(dim=1).cpu().numpy()
                    
                    # Update metrics
                    total_loss += loss.item()
                    all_preds.extend(preds)
                    all_targets.extend(target.cpu().numpy())
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}'
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        # Calculate metrics
        val_loss = total_loss / n_batches if n_batches > 0 else float('inf')
        accuracy = accuracy_score(all_targets, all_preds)
        
        # Calculate TPR and FPR for each class
        class_metrics = {}
        for class_idx in range(self.model.out_classes):
            y_true = np.array(all_targets) == class_idx
            y_pred = np.array(all_preds) == class_idx
            
            tpr = np.sum(y_true & y_pred) / (np.sum(y_true) + 1e-10)
            fpr = np.sum(~y_true & y_pred) / (np.sum(~y_true) + 1e-10)
            
            class_metrics[f"Class {class_idx}"] = {
                "TPR": tpr,
                "FPR": fpr
            }
        
        eval_time = time.time() - eval_start_time
        
        # Log evaluation results
        self.logger.info(
            f"\nEvaluation Results:\n"
            f"  Loss: {val_loss:.4f}\n"
            f"  Accuracy: {accuracy:.4f}\n"
            f"  Time: {eval_time:.2f}s"
        )
        
        for class_name, metrics in class_metrics.items():
            self.logger.info(
                f"  {class_name}:\n"
                f"    TPR: {metrics['TPR']:.4f}\n"
                f"    FPR: {metrics['FPR']:.4f}"
            )
        
        return accuracy, class_metrics[f"Class {1}"]["TPR"], class_metrics[f"Class {1}"]["FPR"], np.array(all_preds)

    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop with checkpointing and visualization"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, num_epochs):
            try:
                # Training step
                avg_loss = self.train_epoch(train_loader)
                
                # Validation step
                val_acc, tpr, fpr, _ = self.evaluate(val_loader)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update visualization history
                self.visualizer.update_history(
                    epoch=epoch,
                    loss=avg_loss,
                    accuracy=val_acc,
                    learning_rate=current_lr,
                    tpr=tpr,
                    fpr=fpr
                )
                
                # Save checkpoint every N epochs and best model
                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    metrics = {
                        'loss': avg_loss,
                        'accuracy': val_acc,
                        'tpr': tpr,
                        'fpr': fpr,
                        'learning_rate': current_lr
                    }
                    
                    is_best = val_acc > self.best_accuracy
                    if is_best:
                        self.best_accuracy = val_acc
                        
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=metrics,
                        is_best=is_best
                    )
                    
                # Plot and save metrics
                self.visualizer.plot_metrics()
                self.visualizer.save_metrics_csv()
                
                self.current_epoch = epoch + 1
                
            except Exception as e:
                self.logger.error(f"Error in epoch {epoch}: {str(e)}")
                continue
                
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        try:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                filepath=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer
            )
            
            self.current_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Resumed training from epoch {self.current_epoch}")
            
        except Exception as e:
            self.logger.error(f"Error resuming from checkpoint: {str(e)}")
            raise 

    def evaluate_loss(self, val_loader):
        """Calculate validation loss"""
        self.model.eval()
        total_loss = 0.0
        n_batches = len(val_loader)
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / n_batches

    def get_last_train_accuracy(self):
        """Return the accuracy from the last training batch"""
        return self.last_train_accuracy if hasattr(self, 'last_train_accuracy') else 0.0 