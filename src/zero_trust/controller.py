import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ZeroTrustMetrics:
    """Container for zero-trust related metrics"""
    tpr: float
    fpr: float
    risk_score: float
    anomaly_score: float

class ZeroTrustController:
    """
    Dynamic Zero-Trust Controller that adapts thresholds based on model performance
    """
    def __init__(
        self,
        gamma_init: float = 0.5,
        tau_init: float = 0.5,
        alpha: float = 0.1,  # Learning rate for threshold updates
        min_gamma: float = 0.3,
        max_gamma: float = 0.8,
        min_tau: float = 0.2,
        max_tau: float = 0.9
    ):
        self.logger = logging.getLogger(__name__)
        self.gamma_q = gamma_init
        self.tau = tau_init
        self.alpha = alpha
        
        # Bounds for thresholds
        self.gamma_bounds = (min_gamma, max_gamma)
        self.tau_bounds = (min_tau, max_tau)
        
        # History tracking
        self.gamma_history = [gamma_init]
        self.tau_history = [tau_init]
        self.tpr_history = []
        self.fpr_history = []
        
        self.logger.info(
            f"Initialized ZeroTrustController:\n"
            f"  Initial γ_q: {gamma_init}\n"
            f"  Initial τ: {tau_init}\n"
            f"  Update rate α: {alpha}\n"
            f"  Gamma bounds: {self.gamma_bounds}\n"
            f"  Tau bounds: {self.tau_bounds}"
        )

    def compute_risk_score(
        self,
        user_context: float,
        device_context: float,
        malicious_prob: float
    ) -> float:
        """
        Compute dynamic risk score based on context and malicious probability
        """
        # Weight factors that adapt based on current thresholds
        w_user = 1.0 - self.gamma_q  # Lower gamma -> higher weight on user context
        w_device = 1.0 - self.gamma_q  # Lower gamma -> higher weight on device context
        w_mal = self.gamma_q  # Higher gamma -> higher weight on malicious probability
        
        # Normalize weights
        total_weight = w_user + w_device + w_mal
        w_user /= total_weight
        w_device /= total_weight
        w_mal /= total_weight
        
        risk_score = (
            w_user * user_context +
            w_device * device_context +
            w_mal * malicious_prob
        )
        
        return risk_score

    def dynamic_update_thresholds(self, stats: Dict[str, float]) -> None:
        """
        Update thresholds based on model performance metrics
        """
        try:
            tpr = stats['TPR']
            fpr = stats['FPR']
            
            # Store metrics history
            self.tpr_history.append(tpr)
            self.fpr_history.append(fpr)
            
            # Calculate metrics for threshold adjustment
            f1_score = 2 * (tpr * (1-fpr)) / (tpr + (1-fpr)) if tpr + (1-fpr) > 0 else 0
            detection_rate = tpr
            false_alarm_rate = fpr
            
            # Dynamic adjustment factors
            gamma_adjustment = self.alpha * (f1_score - 0.5)  # Adjust based on F1 score
            tau_adjustment = self.alpha * (detection_rate - false_alarm_rate)
            
            # Update gamma_q with bounds
            new_gamma = self.gamma_q + gamma_adjustment
            self.gamma_q = np.clip(new_gamma, *self.gamma_bounds)
            
            # Update tau with bounds
            new_tau = self.tau + tau_adjustment
            self.tau = np.clip(new_tau, *self.tau_bounds)
            
            # Store updated thresholds
            self.gamma_history.append(self.gamma_q)
            self.tau_history.append(self.tau)
            
            # Log adjustments
            self.logger.info(
                f"Dynamic threshold update:\n"
                f"  TPR: {tpr:.4f}, FPR: {fpr:.4f}, F1: {f1_score:.4f}\n"
                f"  γ_q adjustment: {gamma_adjustment:.4f}\n"
                f"  τ adjustment: {tau_adjustment:.4f}\n"
                f"  New γ_q: {self.gamma_q:.4f}\n"
                f"  New τ: {self.tau:.4f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {str(e)}")
            raise

    def micro_segmentation_policy(self, segment_id: str, anomaly_score: float) -> str:
        """
        Determine micro-segmentation policy based on anomaly score and current gamma
        """
        if anomaly_score > self.gamma_q:
            decision = f"ISOLATE segment {segment_id} (score: {anomaly_score:.3f} > γ_q: {self.gamma_q:.3f})"
        else:
            decision = f"NORMAL segment {segment_id} (score: {anomaly_score:.3f} <= γ_q: {self.gamma_q:.3f})"
        return decision

    def get_threshold_history(self) -> Dict[str, list]:
        """Return the history of threshold updates and performance metrics"""
        return {
            'gamma_history': self.gamma_history,
            'tau_history': self.tau_history,
            'tpr_history': self.tpr_history,
            'fpr_history': self.fpr_history
        } 