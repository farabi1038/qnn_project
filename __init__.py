# qnn_project/__init__.py

__version__ = "0.1.0"

# Import central logger so it's accessible across the package
from .logger import logger

# Import primary classes and functions for cleaner imports
from .quantum_utils import QuantumUtils, qubit0, qubit1, qubit0mat, qubit1mat
from .qnn_architecture import QNNArchitecture
from .anomaly_detection import AnomalyDetection
from .data_preprocessing import DataPreprocessing

# Optionally, define a list of public objects for wildcard imports
__all__ = [
    "logger",
    "QuantumUtils",
    "QNNArchitecture",
    "AnomalyDetection",
    "DataPreprocessing",
    "qubit0",
    "qubit1",
    "qubit0mat",
    "qubit1mat",
]
