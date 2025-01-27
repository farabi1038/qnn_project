import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Optional, Union
import yaml
import torch
import logging
from pathlib import Path
import glob

class DataPreprocessor:
    """Handles all data preprocessing operations for CESNET dataset"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.feature_columns = self.config['data']['feature_columns']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
            
    def get_data_files(self) -> List[Path]:
        """Get list of all data files"""
        try:
            base_path = Path(self.config['data']['base_path'])
            folder_path = base_path / self.config['data']['folder_path']
            pattern = self.config['data']['file_pattern']
            
            files = list(folder_path.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No files found matching pattern {pattern} in {folder_path}"
                )
                
            self.logger.info(f"Found {len(files)} data files in {folder_path}")
            return files
            
        except Exception as e:
            self.logger.error(f"Error getting data files: {str(e)}")
            raise
            
    def load_data(self) -> pd.DataFrame:
        """Load and combine all data files"""
        try:
            data_files = self.get_data_files()
            all_data = []
            
            for file_path in data_files:
                self.logger.info(f"Loading data from {file_path}")
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                all_data.append(df)
                
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(
                f"Combined data loaded successfully. Shape: {combined_df.shape}"
            )
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        try:
            self.logger.info("Starting data cleaning...")
            
            # Convert columns to numeric and handle missing values
            for col in self.feature_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Handle missing values
            missing_values = df[self.feature_columns].isnull().sum()
            if missing_values.any():
                self.logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
                
            if self.config['data'].get('handle_missing', True):
                df[self.feature_columns] = df[self.feature_columns].fillna(0)
                self.logger.info("Filled missing values with 0")
                
            # Remove outliers using IQR method if configured
            if self.config['data'].get('outlier_removal', True):
                before_shape = df.shape[0]
                for col in self.feature_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[
                        ~((df[col] < (Q1 - 1.5 * IQR)) | 
                          (df[col] > (Q3 + 1.5 * IQR)))
                    ]
                after_shape = df.shape[0]
                removed = before_shape - after_shape
                self.logger.info(f"Removed {removed} outlier rows")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise
            
    def normalize_features(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize features using StandardScaler"""
        try:
            self.logger.info("Normalizing features...")
            normalized_features = self.scaler.fit_transform(df[self.feature_columns])
            self.logger.info("Features normalized successfully")
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"Error in feature normalization: {str(e)}")
            raise
            
    def prepare_data(
        self,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Complete data preparation pipeline"""
        try:
            self.logger.info("Starting data preparation pipeline...")
            
            # Load and clean data
            df = self.load_data()
            df = self.clean_data(df)
            
            # Normalize features
            X = self.normalize_features(df)
            
            # Create labels (example: based on some threshold conditions)
            # You should modify these conditions based on your specific requirements
            y = self.create_labels(df)
            
            # Use config values if not provided
            if test_size is None:
                test_size = self.config['data'].get('test_size', 0.2)
            if random_state is None:
                random_state = self.config['data'].get('random_state', 42)
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y  # Ensure balanced split
            )
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            
            self.logger.info(
                f"Data preparation completed.\n"
                f"Training set shape: {X_train.shape}\n"
                f"Test set shape: {X_test.shape}\n"
                f"Training labels shape: {y_train.shape}\n"
                f"Test labels shape: {y_test.shape}"
            )
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
            
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create labels based on network behavior patterns.
        0: Normal traffic
        1: Suspicious traffic
        2: Malicious traffic
        """
        try:
            labels = np.zeros(len(df), dtype=int)
            
            # Example conditions - modify these based on your specific requirements
            # These are just example thresholds, you should adjust them based on your data
            
            # Suspicious traffic (label 1)
            suspicious_mask = (
                (df['n_flows'] > df['n_flows'].quantile(0.95)) |
                (df['n_packets'] > df['n_packets'].quantile(0.95)) |
                (df['n_bytes'] > df['n_bytes'].quantile(0.95))
            )
            labels[suspicious_mask] = 1
            
            # Malicious traffic (label 2)
            malicious_mask = (
                (df['n_flows'] > df['n_flows'].quantile(0.99)) |
                (df['n_packets'] > df['n_packets'].quantile(0.99)) |
                (df['n_bytes'] > df['n_bytes'].quantile(0.99)) |
                (df['n_dest_ip'] > df['n_dest_ip'].quantile(0.99))
            )
            labels[malicious_mask] = 2
            
            # Log label distribution
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(unique, counts))
            self.logger.info(f"Label distribution: {distribution}")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error creating labels: {str(e)}")
            raise
            
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return self.feature_columns
        
    def save_scaler(self, path: str = 'models/scaler.pkl'):
        """Save the fitted scaler"""
        import joblib
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, path)
            self.logger.info(f"Saved scaler to {path}")
        except Exception as e:
            self.logger.error(f"Error saving scaler: {str(e)}")
            raise
            
    def load_scaler(self, path: str = 'models/scaler.pkl'):
        """Load a previously fitted scaler"""
        import joblib
        try:
            self.scaler = joblib.load(path)
            self.logger.info(f"Loaded scaler from {path}")
        except Exception as e:
            self.logger.error(f"Error loading scaler: {str(e)}")
            raise 