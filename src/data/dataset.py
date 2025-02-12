import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Optional, Tuple

class CasNet2024Dataset(Dataset):
    """Dataset class for CasNet 2024"""
    
    def __init__(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.logger = logging.getLogger(__name__)
        
        # Store data
        self.X = data
        self.y = labels
        self.num_samples = data.shape[0]
        self.in_features = data.shape[1]
        
        self.logger.info(
            f"Initialized CasNet2024Dataset with {self.num_samples} samples, "
            f"{self.in_features} features"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], torch.tensor(0)  # placeholder label when no labels exist
        
class DataManager:
    """Handles dataset creation and loading"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def setup_data_loaders(self):
        """Create and return train/val data loaders"""
        self.logger.info("Setting up data loaders...")
        
        try:
            # Initialize dataset
            dataset = CasNet2024Dataset(
                data=torch.randn(self.config.num_samples, self.config.in_features),
                labels=torch.randint(low=0, high=self.config.num_classes, size=(self.config.num_samples,))
            )
            
            # Split dataset
            train_size = self.config.train_size
            val_size = len(dataset) - train_size
            
            self.logger.info(f"Splitting dataset: train={train_size}, val={val_size}")
            
            train_data, val_data = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create loaders
            train_loader = DataLoader(
                train_data,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_data,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            self.logger.info("Data loaders created successfully")
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Error setting up data loaders: {str(e)}")
            raise 