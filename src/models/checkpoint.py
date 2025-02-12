import os
import torch
from datetime import datetime
from typing import Dict, Any, Optional
import json
import logging

class CheckpointManager:
    """Manages model checkpointing and training state"""
    
    def __init__(self, save_dir: str = "checkpoints"):
        self.logger = logging.getLogger(__name__)
        self.save_dir = save_dir
        self.best_metric = float('-inf')
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            self.logger.info(f"Created checkpoint directory: {save_dir}")
            
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> str:
        """Save model checkpoint with all training state"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'timestamp': timestamp
            }
            
            # Save regular checkpoint
            filepath = os.path.join(self.save_dir, f'checkpoint_{timestamp}.pth')
            torch.save(checkpoint, filepath)
            
            # Save as latest checkpoint
            latest_path = os.path.join(self.save_dir, 'latest.pth')
            torch.save(checkpoint, latest_path)
            
            # Save best checkpoint if applicable
            if is_best:
                best_path = os.path.join(self.save_dir, 'best.pth')
                torch.save(checkpoint, best_path)
                self.logger.info(f"Saved new best checkpoint with metrics: {metrics}")
            
            self.logger.info(f"Saved checkpoint at epoch {epoch} to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint and return training state"""
        try:
            self.logger.info(f"Loading checkpoint from {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
                
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            self.logger.info(
                f"Loaded checkpoint from epoch {checkpoint['epoch']}\n"
                f"Metrics: {checkpoint['metrics']}"
            )
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise
            
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint if it exists"""
        try:
            latest_path = os.path.join(self.save_dir, 'latest.pth')
            if os.path.exists(latest_path):
                self.logger.info(f"Found latest checkpoint: {latest_path}")
                return latest_path
            
            # If no latest.pth, try to find the most recent checkpoint
            checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith('checkpoint_')]
            if checkpoints:
                latest = max(checkpoints)
                latest_path = os.path.join(self.save_dir, latest)
                self.logger.info(f"Found most recent checkpoint: {latest_path}")
                return latest_path
                
            self.logger.info("No checkpoints found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting latest checkpoint: {str(e)}")
            raise
            
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the path to the best checkpoint if it exists"""
        try:
            best_path = os.path.join(self.save_dir, 'best.pth')
            if os.path.exists(best_path):
                self.logger.info(f"Found best checkpoint: {best_path}")
                return best_path
            
            self.logger.info("No best checkpoint found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting best checkpoint: {str(e)}")
            raise 
