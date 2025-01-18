import os
import yaml
from logger import logger
from data_preprocessing import load_cesnet_data
from anomaly_detection import AnomalyDetection
from qnn_architecture import QNNArchitecture
from continuous_qnn import ContinuousVariableQNN
from discrete_qnn import DiscreteVariableQNN
from micro_segmentation import MicroSegmentation
from zero_trust_framework import ZeroTrustFramework

def load_config(config_path="config.yml"):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    logger.info("Starting Quantum Anomaly Detection Pipeline")

    # Load configuration
    config = load_config()

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_cesnet_data(
        config["data"]["csv_path"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    logger.info("Data successfully loaded and preprocessed.")

    # Initialize the appropriate QNN model
    if config["quantum"]["type"] == "discrete":
        qnn_model = DiscreteVariableQNN(
            n_qubits=config["quantum"]["n_qubits"],
            n_layers=config["quantum"]["n_layers"]
        )
    else:
        qnn_model = ContinuousVariableQNN(
            n_qumodes=config["quantum"]["n_qumodes"],
            n_layers=config["quantum"]["n_layers"],
            cutoff_dim=config["quantum"]["cutoff_dim"]
        )
    logger.info(f"Initialized {config['quantum']['type']} QNN model.")

    # Create random QNN architecture and training data
    qnn_arch = config["qnn_architecture"]["architecture"]
    num_training_pairs = config["qnn_architecture"]["num_training_pairs"]

    logger.info("Creating random QNN network and training data...")
    _, unitaries, training_data, _ = QNNArchitecture.random_network(qnn_arch, num_training_pairs)

    # Train QNN model
    logger.info("Starting QNN training...")
    trained_unitaries = QNNArchitecture.qnn_training(
        qnn_arch=qnn_arch,
        unitaries=unitaries,
        training_data=training_data,
        learning_rate=config["training"]["learning_rate"],
        epochs=config["training"]["epochs"]
    )
    logger.info("QNN training completed.")

    # Detect anomalies using the trained QNN
    logger.info("Detecting anomalies on test data...")
    anomaly_scores = [
        AnomalyDetection.detect_anomaly(qnn_model, x, config["anomaly"]["threshold"])
        for x in X_test
    ]

    # Optimize threshold
    best_threshold = AnomalyDetection.adjust_threshold(anomaly_scores, config["anomaly"]["percentile"])
    logger.info(f"Optimized anomaly detection threshold: {best_threshold:.4f}")

    # Micro-segmentation
    micro_segmentation = MicroSegmentation(segment_threshold=config["zero_trust"]["segment_threshold"])
    segment_predictions = [1 if score > best_threshold else 0 for score in anomaly_scores]
    isolated_segments = micro_segmentation.isolate_segments(X_test, segment_predictions)
    logger.info(f"Segments flagged for isolation: {isolated_segments}")

    # Zero Trust Framework
    zt_framework = ZeroTrustFramework(risk_threshold=config["zero_trust"]["risk_threshold"])

    for i, (x_sample, score) in enumerate(zip(X_test, anomaly_scores)):
        user_ctx = {"role": "user"}
        device_ctx = {"location": "local"}
        risk_score = zt_framework.compute_risk_score(user_ctx, device_ctx, score)
        access_decision = zt_framework.decide_access(risk_score)
        logger.info(f"Sample {i + 1}: Risk Score={risk_score:.4f}, Access={'Granted' if access_decision else 'Denied'}")

    logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()
