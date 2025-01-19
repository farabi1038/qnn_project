import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.utils import setup_logger

logger = setup_logger()

def load_cesnet_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads and preprocesses CESNET data for anomaly detection.
    
    Args:
        csv_path: Path to the CSV file containing the dataset.
        test_size: Fraction of the data to use for testing.
        random_state: Seed for reproducibility in train-test split.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    try:
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading file {csv_path}: {e}")
        raise

    # Example preprocessing: Adjust to match your dataset schema
    logger.info("Starting data cleaning and preprocessing...")

    # Define features and target column
    features = [
        "id_time", "n_flows", "n_packets", "n_bytes", "n_dest_asn",
        "n_dest_ports", "n_dest_ip", "tcp_udp_ratio_packets",
        "tcp_udp_ratio_bytes", "dir_ratio_packets", "dir_ratio_bytes",
        "avg_duration", "avg_ttl"
    ]

    # Define a synthetic target column if needed (binary classification for anomalies)
    # Here we create an example "label" column based on a threshold on `n_packets`
    threshold = 75000  # Example threshold for anomaly classification
    df["label"] = (df["n_packets"] > threshold).astype(int)

    if "label" not in df.columns:
        logger.error("No 'label' column found in the dataset.")
        raise ValueError("The dataset must include a 'label' column.")

    # Separate features and labels
    X = df[features].values
    y = df["label"].values

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Data normalization complete. Shape: {X_scaled.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Data split complete: Train size = {len(X_train)}, Test size = {len(X_test)}")

    return X_train, X_test, y_train, y_test

def normalize_features(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize numerical features to [0,1] range."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    result = data.copy()
    for col in numeric_cols:
        min_val = data[col].min()
        max_val = data[col].max()
        result[col] = (data[col] - min_val) / (max_val - min_val + 1e-8)
    return result

# ... any other data preprocessing functions ...
