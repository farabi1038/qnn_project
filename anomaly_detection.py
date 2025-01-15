# qnn_project/anomaly_detection.py

import numpy as np
import qutip as qt
from quantum_utils import QuantumUtils
from qnn_architecture import QNNArchitecture


class AnomalyDetection:
    """
    A class for anomaly detection functionalities, 
    including anomaly scoring, threshold adjustments, and policy decisions.
    """

    @staticmethod
    def detect_anomaly(qnn_arch, trained_unitaries, input_state: qt.Qobj, threshold: float) -> float:
        """
        Pass input_state through the trained QNN and compute an anomaly score.
        """
        current_state = input_state
        for l in range(1, len(qnn_arch)):
            current_state = QNNArchitecture.make_layer_channel(qnn_arch, trained_unitaries, l, current_state)

        # Make sure we have density matrices
        if current_state.isket:
            current_state = current_state.proj()
        if input_state.isket:
            input_state = input_state.proj()

        try:
            fidelity = abs((current_state * input_state).tr().real)
            trace_distance = 1 - abs(current_state.tr().real)
            entropy = -abs(current_state.tr().real) * np.log(abs(current_state.tr().real) + 1e-10)
            anomaly_score = trace_distance + (1 - fidelity) + entropy
        except Exception as e:
            print(f"Error calculating anomaly score: {e}")
            anomaly_score = 0.0  # Default to 0 if error occurs
        return max(anomaly_score, 1e-6)

    @staticmethod
    def adjust_threshold(anomaly_scores: list, percentile: float = 80) -> float:
        """
        Adjust threshold based on percentile of anomaly scores.
        """
        return np.percentile(anomaly_scores, percentile)

    @staticmethod
    def adjust_policy(current_policy: str, anomaly_score: float, threshold: float) -> str:
        """
        Simple example of policy adjustment based on anomaly_score vs threshold.
        """
        if anomaly_score > threshold:
            return "Restricted"
        elif anomaly_score > threshold / 2:
            return "Monitored"
        elif anomaly_score > threshold / 4:
            return "Inspected"
        return "Open"

    @staticmethod
    def simulate_traffic(qnn_arch, trained_unitaries, num_samples: int,
                         anomaly_threshold: float, policy_threshold: float,
                         noise_level: float = 0.2) -> None:
        """
        Example simulation over random input states with optional noise.
        """
        for _ in range(num_samples):
            input_state = QuantumUtils.random_qubit_state(qnn_arch[0])
            if noise_level > 0.0:
                input_state = QuantumUtils.noisy_state(input_state, noise_level)
            anomaly_score = AnomalyDetection.detect_anomaly(
                qnn_arch, trained_unitaries, input_state, anomaly_threshold
            )
            policy = AnomalyDetection.adjust_policy("Default", anomaly_score, policy_threshold)
            print(f"Anomaly Score: {anomaly_score:.4f}, Policy: {policy}")
