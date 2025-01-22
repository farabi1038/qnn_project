import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import setup_logger
import torch
import yaml
from typing import Dict, Tuple

logger = setup_logger()

def load_cesnet_data(file_path=None):
    """Load and preprocess CESNET dataset."""
    try:
        logger.info("Starting data cleaning and preprocessing...")
        
        # Load config
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Use provided file_path or get from config
        if file_path is None:
            file_path = config.get('data', {}).get('cesnet_path', 'data/cesnet/cesnet_data.csv')
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Get all columns except id_time
        feature_columns = [col for col in df.columns if col != 'id_time']
        logger.info(f"Found columns: {df.columns.tolist()}")
        
        # Basic preprocessing
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
        
        # Normalize numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_columns])
        y = None
            
        # logger.info(f"Data preprocessing completed. X shape: {X.shape}")
        logger.info(f"Data preprocessing completed. X shape: {X.shape}, y shape:")
        return [X, y]
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def normalize_features(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize numerical features to [0,1] range."""
    feature_columns = [
        'n_flows', 'n_packets', 'n_bytes', 'n_dest_asr', 'n_dest_port', 
        'n_dest_ip', 'tcp_udp_ratio', 'tcp_udp_ratio_dest', 'dir_ratio_p', 
        'dir_ratio_b', 'avg_duration', 'avg_ttl'
    ]
    
    result = data.copy()
    for col in feature_columns:
        min_val = data[col].min()
        max_val = data[col].max()
        result[col] = (data[col] - min_val) / (max_val - min_val + 1e-8)
    return result

# ... any other data preprocessing functions ...
