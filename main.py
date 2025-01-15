# qnn_project/main_driver.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .data_preprocessing import DataPreprocessing
from .quantum_utils import QuantumUtils
from .qnn_architecture import QNNArchitecture
from .anomaly_detection import AnomalyDetection


def main():
    print("=== QNN Project: Main Driver ===")

    # Example usage: Loading CESNET data
    data_path = "ip_addresses_sample/agg_10_minutes/11.csv"
    print("Loading CESNET dataset...")
    normalized_data, raw_df = DataPreprocessing.load_cesnet_data(data_path, num_samples=5)

    # Convert data to quantum states
    quantum_states = DataPreprocessing.encode_traffic_to_quantum(normalized_data)

    # Build training data with noisy targets
    training_data = DataPreprocessing.prepare_training_data(quantum_states, noise_level=0.1)

    # Number of qubits needed (based on number of features)
    num_features = normalized_data.shape[1]
    num_qubits = int(np.ceil(np.log2(num_features)))

    # Example network architectures
    networks = [
        ([num_qubits, num_qubits * 2, num_qubits], len(training_data) // 2),
        ([num_qubits, num_qubits + 1, num_qubits], len(training_data) // 2),
        ([num_qubits, num_qubits], len(training_data) // 2)
    ]

    simulation_results = []

    # Train & Test each network
    for arch, num_pairs in networks:
        print(f"\nTraining network {arch} with {num_pairs} training pairs...")
        qnn_tuple = QNNArchitecture.random_network(arch, num_pairs)
        trained_unitaries = QNNArchitecture.qnn_training(
            qnn_tuple[0],  # qnn_arch
            qnn_tuple[1],  # unitaries
            training_data[:num_pairs],
            lda=1, ep=0.1, training_rounds=100
        )[1]

        print(f"\nTesting network {arch} with real traffic data...")

        test_data = training_data[num_pairs:]
        anomaly_scores = []
        for test_state, _ in test_data:
            current_state = test_state
            for l in range(1, len(arch)):
                current_state = QNNArchitecture.make_layer_channel(arch, trained_unitaries, l, current_state)
            # Example anomaly score = 1 - (trace of final state's density matrix)
            try:
                anomaly_score = 1 - abs(current_state.tr().real)
                if np.isnan(anomaly_score):
                    anomaly_score = 0.0
            except:
                anomaly_score = 0.0
            anomaly_scores.append(anomaly_score)

        # Threshold: 95th percentile
        threshold = np.percentile(anomaly_scores, 95)
        print(f"Testing traffic patterns...")
        for i, score in enumerate(anomaly_scores[:20]):
            policy = "Restricted" if score > threshold else "Open"
            print(f"Sample {i+1} - Anomaly Score: {score:.4f}, Policy: {policy}")

        # Plot distribution of anomaly scores if valid
        if any(not np.isnan(s) for s in anomaly_scores):
            plt.figure(figsize=(10, 6))
            plt.hist(anomaly_scores, bins=20, alpha=0.7)
            plt.title(f"Anomaly Score Distribution for Network {arch}")
            plt.xlabel("Anomaly Score")
            plt.ylabel("Frequency")
            plt.show()

    # Another example of building a small network and simulating traffic
    print("\n=== Additional Example: [1, 2, 1] Network ===")
    network121 = QNNArchitecture.random_network([1, 2, 1], 10)
    trained_unitaries = QNNArchitecture.qnn_training(network121[0], network121[1], network121[2],
                                                     lda=1, ep=0.1, training_rounds=500)[1]

    # Simulate traffic
    AnomalyDetection.simulate_traffic(
        network121[0], trained_unitaries,
        num_samples=20, anomaly_threshold=0.5, policy_threshold=0.7, noise_level=0.2
    )

    # Example of further simulations with different networks
    networks_extended = [
        ([3, 6, 3], 10),
        ([3, 4, 3], 10),
        ([3, 3], 10)
    ]

    for arch, num_pairs in networks_extended:
        print(f"\nSimulation for network {arch} with {num_pairs} training pairs:")
        net = QNNArchitecture.random_network(arch, num_pairs)
        trained_uni = QNNArchitecture.qnn_training(net[0], net[1], net[2], 1, 0.1, 200)[1]

        # Simulate traffic
        for sample_id in range(1, 21):
            input_state = QuantumUtils.random_qubit_state(arch[0])
            input_state = QuantumUtils.noisy_state(input_state, noise_level=0.2)
            anomaly_score = AnomalyDetection.detect_anomaly(arch, trained_uni, input_state, threshold=0.5)
            policy = AnomalyDetection.adjust_policy("Default", anomaly_score, 0.7)

            print(f"Sample {sample_id} - Anomaly Score: {anomaly_score:.4f}, Policy: {policy}")
            simulation_results.append([sample_id, str(arch), anomaly_score, policy])

    # Convert simulation results to DataFrame
    simulation_df = pd.DataFrame(simulation_results,
                                 columns=["Sample", "Network Architecture", "Anomaly Score", "Policy"])
    simulation_df.to_csv("simulation_results.csv", index=False)
    print("Simulation results saved to simulation_results.csv")

    # Some example plotting of the final results
    unique_networks = simulation_df["Network Architecture"].unique()
    for network in unique_networks:
        network_data = simulation_df[simulation_df["Network Architecture"] == network]

        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.hist(network_data["Anomaly Score"], bins=10, alpha=0.7)
        plt.title(f"Anomaly Score Distribution for Network {network}")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        # Plot policy distribution
        plt.figure(figsize=(8, 6))
        policy_counts = network_data["Policy"].value_counts()
        plt.bar(policy_counts.index, policy_counts.values, alpha=0.7)
        plt.title(f"Policy Distribution for Network {network}")
        plt.xlabel("Policy")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

        # Heatmap
        pivot_table = network_data.pivot(index="Sample", columns="Policy", values="Anomaly Score")
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm",
                    cbar_kws={'label': 'Anomaly Score'})
        plt.title(f"Heatmap of Anomaly Scores for Network {network}")
        plt.xlabel("Policy")
        plt.ylabel("Sample")
        plt.show()

        # CDF plot
        plt.figure(figsize=(10, 6))
        sorted_scores = network_data["Anomaly Score"].sort_values()
        plt.plot(sorted_scores, 
                 np.arange(1, len(sorted_scores)+1) / len(sorted_scores),
                 marker='o')
        plt.title(f"Cumulative Distribution of Anomaly Scores for Network {network}")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.show()


# If you want to run this script directly:
if __name__ == "__main__":
    main()
