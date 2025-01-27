import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class ZeroTrustMetrics:
    """Container for zero-trust related metrics"""
    tpr: float
    fpr: float
    risk_score: float
    anomaly_score: float

class ZeroTrustController:
    """Manages zero-trust logic and decisions"""
    
    def __init__(self, gamma_init=0.5, tau_init=0.5):
        self.logger = logging.getLogger(__name__)
        self.gamma_q = gamma_init
        self.tau = tau_init
        
        self.logger.info(
            f"ZeroTrustController initialized with:\n"
            f"  gamma_q: {gamma_init}\n"
            f"  tau: {tau_init}"
        )
        
    def compute_risk_score(self, user_context, device_context, malicious_prob) -> float:
        """Compute risk score based on context and malicious probability"""
        try:
            base_risk = 0.5 * (user_context + device_context)
            final_risk = base_risk + 0.5 * malicious_prob
            
            self.logger.debug(
                f"Risk score computed:\n"
                f"  User context: {user_context:.3f}\n"
                f"  Device context: {device_context:.3f}\n"
                f"  Malicious prob: {malicious_prob:.3f}\n"
                f"  Final risk: {final_risk:.3f}"
            )
            
            return final_risk
            
        except Exception as e:
            self.logger.error(f"Error computing risk score: {str(e)}")
            raise
            
    def dynamic_update_thresholds(self, stats_dict: Dict[str, float]):
        """Update thresholds based on TPR/FPR statistics"""
        self.logger.info("Updating thresholds...")
        
        try:
            tpr = stats_dict.get('TPR', 0.0)
            fpr = stats_dict.get('FPR', 0.0)
            
            old_gamma = self.gamma_q
            old_tau = self.tau

            # Update gamma_q based on FPR
            if fpr > 0.2:
                self.gamma_q += 0.02
            else:
                self.gamma_q -= 0.01

            # Update tau based on TPR
            if tpr < 0.8:
                self.tau -= 0.02
            else:
                self.tau += 0.01

            # Clamp values
            self.gamma_q = np.clip(self.gamma_q, 0.0, 1.0)
            self.tau = np.clip(self.tau, 0.0, 1.0)
            
            self.logger.info(
                f"Thresholds updated:\n"
                f"  gamma_q: {old_gamma:.3f} -> {self.gamma_q:.3f}\n"
                f"  tau: {old_tau:.3f} -> {self.tau:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {str(e)}")
            raise 