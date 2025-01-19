import os
import torch
from typing import Dict
from src.utils import setup_logger

logger = setup_logger()

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