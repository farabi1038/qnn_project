# qnn_project/main.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_preprocessing import DataPreprocessing
from quantum_utils import QuantumUtils
from qnn_architecture import QNNArchitecture
from anomaly_detection import AnomalyDetection

from logger import logger


def main():
    logger.info("=== Starting QNN Project Main Driver ===")

    try:
        # Example usage: Loading CESNET data
        data_path = "data/ip_addresses_sample/agg_10_minutes/11.csv"
        logger.info("Loading CESNET dataset from %s...", data_path)
        normalized_data, raw_df = DataPreprocessing.load_cesnet_data(data_path, num_samples=5)
        logger.info("CESNET dataset loaded successfully with %d samples.", len(normalized_data))

        # Convert data to quantum states
        logger.info("Encoding traffic data into quantum states...")
        quantum_states = DataPreprocessing.encode_traffic_to_quantum(normalized_data)
        logger.info("Encoded traffic data into %d quantum states.", len(quantum_states))

        # Build training data with noisy targets
        logger.info("Preparing training data with noise level 0.1...")
        training_data = DataPreprocessing.prepare_training_data(quantum_states, noise_level=0.1)
        logger.info("Prepared training data with %d samples.", len(training_data))

        # Number of qubits needed (based on number of features)
        num_features = normalized_data.shape[1]
        num_qubits = int(np.ceil(np.log2(num_features)))
        logger.info("Calculated number of qubits needed: %d.", num_qubits)

        # Example network architectures
        networks = [
            ([num_qubits, num_qubits * 2, num_qubits], len(training_data) // 2),
            ([num_qubits, num_qubits + 1, num_qubits], len(training_data) // 2),
            ([num_qubits, num_qubits], len(training_data) // 2),
        ]
        logger.info("Defined network architectures: %s", networks)

        simulation_results = []

        # Train & Test each network
        for arch, num_pairs in networks:
            logger.info("Training network %s with %d training pairs...", arch, num_pairs)
            qnn_tuple = QNNArchitecture.random_network(arch, num_pairs)
            trained_unitaries = QNNArchitecture.qnn_training(
                qnn_tuple[0],  # qnn_arch
                qnn_tuple[1],  # unitaries
                training_data[:num_pairs],
                lda=1, ep=0.1, training_rounds=100
            )[1]
            logger.info("Training completed for network %s.", arch)

            logger.info("Testing network %s with remaining data...", arch)
            test_data = training_data[num_pairs:]
            anomaly_scores = []
            for test_state, _ in test_data:
                current_state = test_state
                for l in range(1, len(arch)):
                    current_state = QNNArchitecture.make_layer_channel(arch, trained_unitaries, l, current_state)
                try:
                    anomaly_score = 1 - abs(current_state.tr().real)
                    if np.isnan(anomaly_score):
                        anomaly_score = 0.0
                except Exception as e:
                    logger.error("Error calculating anomaly score: %s", e)
                    anomaly_score = 0.0
                anomaly_scores.append(anomaly_score)

            # Threshold: 95th percentile
            threshold = np.percentile(anomaly_scores, 95)
            logger.info("Threshold for anomaly detection set at 95th percentile: %.4f.", threshold)

            logger.info("Analyzing test results for network %s...", arch)
            for i, score in enumerate(anomaly_scores[:20]):
                policy = "Restricted" if score > threshold else "Open"
                logger.info("Sample %d - Anomaly Score: %.4f, Policy: %s", i + 1, score, policy)

            # Plot distribution of anomaly scores if valid
            if any(not np.isnan(s) for s in anomaly_scores):
                plt.figure(figsize=(10, 6))
                plt.hist(anomaly_scores, bins=20, alpha=0.7)
                plt.title(f"Anomaly Score Distribution for Network {arch}")
                plt.xlabel("Anomaly Score")
                plt.ylabel("Frequency")
                plt.show()

        # Additional network and traffic simulation example
        logger.info("Simulating traffic for additional networks...")
        network121 = QNNArchitecture.random_network([1, 2, 1], 10)
        trained_unitaries = QNNArchitecture.qnn_training(
            network121[0], network121[1], network121[2],
            lda=1, ep=0.1, training_rounds=500
        )[1]

        AnomalyDetection.simulate_traffic(
            network121[0], trained_unitaries,
            num_samples=20, anomaly_threshold=0.5, policy_threshold=0.7, noise_level=0.2
        )

        logger.info("Simulating extended network architectures...")
        networks_extended = [
            ([3, 6, 3], 10),
            ([3, 4, 3], 10),
            ([3, 3], 10),
        ]

        for arch, num_pairs in networks_extended:
            logger.info("Simulating for network %s with %d training pairs...", arch, num_pairs)
            net = QNNArchitecture.random_network(arch, num_pairs)
            trained_uni = QNNArchitecture.qnn_training(
                net[0], net[1], net[2], 1, 0.1, 200
            )[1]

            for sample_id in range(1, 21):
                input_state = QuantumUtils.random_qubit_state(arch[0])
                input_state = QuantumUtils.noisy_state(input_state, noise_level=0.2)
                anomaly_score = AnomalyDetection.detect_anomaly(arch, trained_uni, input_state, threshold=0.5)
                policy = AnomalyDetection.adjust_policy("Default", anomaly_score, 0.7)

                logger.info("Sample %d - Anomaly Score: %.4f, Policy: %s", sample_id, anomaly_score, policy)
                simulation_results.append([sample_id, str(arch), anomaly_score, policy])

        # Save simulation results
        simulation_df = pd.DataFrame(simulation_results,
                                     columns=["Sample", "Network Architecture", "Anomaly Score", "Policy"])
        simulation_df.to_csv("data/output/simulation_results.csv", index=False)
        logger.info("Simulation results saved to simulation_results.csv")

    except Exception as e:
        logger.critical("A critical error occurred: %s", e, exc_info=True)


# If you want to run this script directly:
if __name__ == "__main__":
    main()
