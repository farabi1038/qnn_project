import torch
from typing import Dict, List

from src.utils import setup_logger
from src.models import ContinuousVariableQNN, DiscreteVariableQNN, QNNArchitecture
from src.utils.quantum_utils import QuantumUtils

logger = setup_logger()

import numpy as np
import qutip as qt

class AnomalyDetector:
    """Quantum-based anomaly detection system."""
    
    def __init__(self, model_type: str = 'continuous'):
        self.quantum_utils = QuantumUtils()
        if model_type == 'continuous':
            self.model = ContinuousVariableQNN()
        else:
            self.model = DiscreteVariableQNN()
        logger.info(f"Initialized AnomalyDetector with {model_type} model")

    @staticmethod
    def detect_anomaly(qnn_model, input_state: np.ndarray, threshold: float) -> float:
        """
        Pass input_state through the trained QNN and compute an anomaly score.

        :param qnn_model: Trained QNN model (discrete or continuous).
        :param input_state: Classical input state as a NumPy array.
        :param threshold: Current anomaly threshold for classification.
        :return: Anomaly score (float).
        """
        logger.debug("detect_anomaly => Starting anomaly detection with threshold=%.4f", threshold)

        try:
            # Forward pass through the QNN
            anomaly_score = qnn_model.forward_pass(input_state)

            # Ensure score is bounded
            anomaly_score = max(0.0, min(1.0, anomaly_score))
            logger.debug("detect_anomaly => Computed anomaly score: %.4f", anomaly_score)
        except Exception as e:
            logger.error("Error during anomaly detection: %s", e, exc_info=True)
            anomaly_score = 1.0  # Default to highly anomalous if error occurs

        logger.info("detect_anomaly => Final anomaly score: %.6f (threshold=%.4f)", anomaly_score, threshold)
        return anomaly_score

    @staticmethod
    def adjust_threshold(anomaly_scores: list, percentile: float = 80) -> float:
        """
        Adjust threshold based on a chosen percentile.

        :param anomaly_scores: List of computed anomaly scores.
        :param percentile: Percentile to calculate the new threshold.
        :return: Adjusted threshold value.
        """
        if not anomaly_scores:
            logger.warning("adjust_threshold => No anomaly scores provided. Returning default threshold of 0.5.")
            return 0.5
        new_thresh = np.percentile(anomaly_scores, percentile)
        logger.info("adjust_threshold => New threshold set to %.4f based on the %.2f percentile.", new_thresh, percentile)
        return new_thresh

    @staticmethod
    def adjust_policy(current_policy: str, anomaly_score: float, threshold: float) -> str:
        """
        Adjust access policy based on the anomaly score.

        :param current_policy: Current policy state, e.g., "Default".
        :param anomaly_score: Computed anomaly score.
        :param threshold: Threshold above which the policy tightens.
        :return: Updated policy state.
        """
        logger.debug("adjust_policy => Current policy: %s, Anomaly Score: %.4f, Threshold: %.4f",
                     current_policy, anomaly_score, threshold)

        if anomaly_score > threshold:
            new_policy = "Restricted"
        elif anomaly_score > threshold / 2:
            new_policy = "Monitored"
        elif anomaly_score > threshold / 4:
            new_policy = "Inspected"
        else:
            new_policy = "Open"

        logger.info("adjust_policy => Updated policy: %s", new_policy)
        return new_policy

    @staticmethod
    def simulate_traffic(qnn_model, num_samples: int, anomaly_threshold: float,
                         policy_threshold: float, noise_level: float = 0.2) -> None:
        """
        Simulate traffic using a QNN model and dynamically classify/analyze states.

        :param qnn_model: Trained QNN model (discrete or continuous).
        :param num_samples: Number of samples to simulate.
        :param anomaly_threshold: Threshold for anomaly classification.
        :param policy_threshold: Threshold for policy updates.
        :param noise_level: Gaussian noise level added to inputs.
        """
        logger.info("simulate_traffic => Simulating %d samples with thresholds: anomaly=%.4f, policy=%.4f",
                    num_samples, anomaly_threshold, policy_threshold)

        results = []
        for i in range(num_samples):
            # Generate a random input state
            input_state = QuantumUtils.random_qubit_state(qnn_model.n_qubits) if hasattr(qnn_model, 'n_qubits') \
                else np.random.rand(qnn_model.n_qumodes)

            if noise_level > 0.0:
                input_state += np.random.normal(scale=noise_level, size=input_state.shape)

            # Detect anomaly
            anomaly_score = AnomalyDetector.detect_anomaly(qnn_model, input_state, anomaly_threshold)

            # Adjust policy based on anomaly score
            policy = AnomalyDetector.adjust_policy("Default", anomaly_score, policy_threshold)
            results.append((i + 1, anomaly_score, policy))

            logger.info("simulate_traffic => Sample %d: Anomaly Score=%.4f, Policy=%s", i + 1, anomaly_score, policy)

        # Print summary
        for result in results:
            print(f"Sample {result[0]}: Anomaly Score={result[1]:.4f}, Policy={result[2]}")
