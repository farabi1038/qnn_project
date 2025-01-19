from typing import List, Dict
import numpy as np

from ..utils import setup_logger
from .anomaly_detection import AnomalyDetector

logger = setup_logger()

class MicroSegmentation:
    """
    Dynamically isolates segments (flows) deemed high-risk based on anomaly scores.
    """

    def __init__(self, segment_threshold: float):
        """
        Initialize the MicroSegmentation class with a segment-level threshold.

        :param segment_threshold: Threshold for anomaly ratio to flag a segment as high-risk.
        """
        self.segment_threshold = segment_threshold

    def isolate_segments(self, X_seg: list, predictions: list) -> list:
        """
        Identify which segments to isolate based on anomaly predictions.

        :param X_seg: List of feature sets for each segment.
        :param predictions: List of anomaly predictions (1=anomaly, 0=normal) for flows.
        :return: List of indices of flows to isolate.
        """
        # Calculate the ratio of anomalies in the segment
        anomaly_ratio = sum(predictions) / len(predictions) if predictions else 0

        # Flag the entire segment if the anomaly ratio exceeds the threshold
        if anomaly_ratio > self.segment_threshold:
            return list(range(len(X_seg)))

        return []

    def segment_summary(self, X_seg: list, predictions: list) -> dict:
        """
        Generate a summary report for a segment.

        :param X_seg: List of feature sets for the segment.
        :param predictions: List of anomaly predictions for the flows in the segment.
        :return: Dictionary summarizing segment information.
        """
        anomaly_count = sum(predictions)
        total_flows = len(predictions)
        return {
            "total_flows": total_flows,
            "anomalies_detected": anomaly_count,
            "anomaly_ratio": anomaly_count / total_flows if total_flows else 0
        }