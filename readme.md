# Quantum-driven ZTA with Dynamic Anomaly Detection in 7G Technology: A Neural Network Approach

A scalable quantum-classical hybrid platform that integrates Quantum Neural Networks (QNN) for anomaly detection with Zero Trust Architecture (ZTA) for enhanced security in next-generation networks.

![Quantum ZTA paper ](https://arxiv.org/abs/2502.07779)

## Table of Contents
- [Overview](#overview)
- [Research Paper](#research-paper)
- [Features](#features)
- [System Architecture](#system-architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Security Framework](#security-framework)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Overview

This repository contains the implementation of our quantum-classical hybrid system that integrates:
- Quantum Neural Networks (QNN) for anomaly detection and decision making
- Zero Trust Architecture (ZTA) with dynamic micro-segmentation
- Three-class classification for network traffic analysis
- Real-time risk scoring and adaptive policy enforcement
- Quantum feature encoding for enhanced detection capabilities

Our system provides a novel approach to network security by combining quantum computing principles with traditional security frameworks. The integration of quantum neural networks significantly improves detection accuracy, reduces false positives, and enhances security posture compared to classical approaches.

## Research Paper

This implementation is based on our research paper "*Quantum-driven ZTA with Dynamic Anomaly Detection in 7G Technology: A Neural Network Approach*" by S. Ahmed et al. The paper details our approach to integrating quantum computing techniques with zero trust security principles for next-generation networks. Key aspects covered include:

1. **Quantum Neural Network Framework**: Theoretical foundation and implementation of QNNs for security applications
2. **Embedding Classical NN into Quantum**: Methodology for integrating classical architectures within quantum formalism
3. **Linear Interferometers**: Mathematical foundation for transformations in quantum systems
4. **Specialized Quantum Architectures**: Extensions beyond fully connected architectures
5. **Advantages of QNNs**: Key benefits of quantum approaches over classical methods
6. **Experimental Results**: Comprehensive evaluation using the CESNET dataset

The paper provides detailed mathematical formulations, theoretical underpinnings, and experimental validation of our approach. For a comprehensive understanding of the system, please refer to the full paper.

## Features

- **QNN-based Anomaly Detection**: Advanced anomaly detection using quantum-inspired computational techniques
- **Three-Class Classification**: Precise categorization of network traffic as normative, suspicious, or malicious
- **Zero-Trust Security**: Comprehensive security framework with micro-segmentation for system isolation
- **Dynamic Risk Scoring**: Real-time risk evaluation using the quantum-enhanced function `Rq(u,d) = Fq(cu, cd, xi, ŷqi)`
- **Adaptive Micro-Segmentation**: Dynamic adjustment of network segments based on detected anomalies
- **GPU Acceleration**: CUDA-optimized algorithms for quantum circuit simulation and training
- **Quantum Feature Encoding**: Transformation of classical data into quantum states using amplitude and angle encoding

## System Architecture

Our system implements a hybrid quantum-classical architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                Data Collection & Preprocessing               │
│   ┌───────────────────────────────────────────────────────┐ │
│   │     Network Traffic Data & Contextual Information      │ │
│   └─────────────────────────────┬─────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────┐
│                  Quantum Feature Encoding                      │
│   ┌───────────────────────────────────────────────────────┐   │
│   │              E(xi) → |ψi⟩ Transformation              │   │
│   └─────────────────────────────┬─────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────┐
│                  Quantum Neural Network                        │
│   ┌───────────────────┐    ┌───────────────────────────────┐  │
│   │ Variational Circuit│    │     Quantum Measurement       │  │
│   │     Processing     │    │      & Classification         │  │
│   └─────────┬─────────┘    └───────────────┬───────────────┘  │
└─────────────┼───────────────────────────────────────────────────┘
              │                             │
┌─────────────▼─────────────────────────────▼───────────────────┐
│                     Dynamic Risk Assessment                    │
│   ┌───────────────────────────────────────────────────────┐   │
│   │         Rq(u,d) = Fq(cu, cd, xi, ŷqi) Scoring         │   │
│   └───────────────────────────┬───────────────────────────┘   │
└───────────────────────────────┼───────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────┐
│                   Adaptive Micro-Segmentation                  │
│   ┌───────────────────────────────────────────────────────┐   │
│   │       P'(Sj) ← Gq(P(Sj), ŷqi) Policy Adjustment       │   │
│   └───────────────────────────┬───────────────────────────┘   │
└───────────────────────────────┼───────────────────────────────┘
                                │
                                ▼
                      Access Control Decisions
```

The architecture implements a novel approach to network security through:

1. **Data Collection & Preprocessing**: Processing network traffic data and contextual information
2. **Quantum Feature Encoding**: Transforming classical features into quantum states using E(xi)
3. **Quantum Neural Network**: Processing quantum states through variational circuits to generate anomaly scores
4. **Dynamic Risk Assessment**: Computing risk scores using the Fq function
5. **Adaptive Micro-Segmentation**: Adjusting security policies using the Gq function
6. **Access Control**: Making final decisions based on risk thresholds

## System Requirements

### Hardware
- NVIDIA GPU with CUDA support (minimum 8GB VRAM)
- Minimum 16GB RAM
- Minimum 256GB SSD storage

### Software
- Ubuntu 20.04 LTS / Windows 10/11
- CUDA 11.4+
- Python 3.8+
- PyTorch 1.10+
- PennyLane 0.21+ (for quantum simulation)
- TensorFlow 2.6+ (optional)

## Installation

### Prerequisites
```bash
# Install system dependencies (Ubuntu)
sudo apt update
sudo apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools python3-venv

# For Windows users
# - Install Python from https://www.python.org/downloads/
# - Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
```

### Setup
```bash
# Clone the repository
git clone https://github.com/farabi1038/qnn_project.git
cd quatum_anomaly

# Create and activate virtual environment (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Usage

### Configuration
Edit the configuration parameters in `src/config.yaml` to customize:
- Model hyperparameters
- QNN architecture
- Security settings
- Feature encoding methods
- Logging preferences

Example configuration:
```yaml
# QNN Configuration
model:
  qnn:
    n_qubits: 4
    n_layers: 2
    optimizer: adam
    learning_rate: 0.001
    encoding: "amplitude"  # Options: "amplitude", "angle", "hybrid"
  
# ZTA Configuration
security:
  zero_trust:
    enabled: true
    micro_segmentation: true
    validation_interval_ms: 100
    gamma_q1: 0.65  # Threshold for class 1-2 separation
    gamma_q2: 0.8   # Threshold for class 2-3 separation
    
# Data Configuration
data:
  preprocessing:
    outlier_removal: true
    iqr_factor: 1.5
    zero_imputation: true
  training_split: 0.8
```

### Running the System
```bash
# Run the main application
python src/main.py

# Run with custom config file
python src/main.py --config path/to/custom_config.yaml

# Run in specific mode
python src/main.py --mode [train|evaluate|infer]
```

### Visualization
The system generates visualizations in the `src/plots/` directory for:
- ROC curves with AUC metrics
- Training and validation performance curves
- Risk score evolution over time
- Adaptive micro-segmentation maps

## Project Structure

```
quatum_anomaly/
├── src/                     # Source code (main directory)
│   ├── main.py              # Main entry point
│   ├── config.yaml          # Configuration file
│   ├── logger_config.py     # Logging configuration
│   ├── __init__.py          # Package initialization
│   ├── models/              # Neural network model implementations
│   │   ├── qnn.py           # Quantum neural network implementation
│   │   └── classical.py     # Classical neural network for comparison
│   ├── zero_trust/          # Zero-trust security framework
│   │   ├── micro_seg.py     # Micro-segmentation implementation
│   │   └── validator.py     # Continuous validation and verification
│   ├── data/                # Data processing and storage
│   │   ├── preprocessor.py  # Data cleaning and preparation
│   │   ├── encoding.py      # Quantum feature encoding methods
│   │   └── cesnet.py        # CESNET dataset handling
│   ├── utils/               # Utility functions and helpers
│   │   ├── visualization.py # Visualization utilities
│   │   └── metrics.py       # Performance metrics calculation
│   ├── plots/               # Visualization outputs
│   ├── checkpoints/         # Model checkpoints
│   └── logs/                # Application logs
├── papers/                  # Research papers and documentation
│   └── quantum_zta.pdf      # Main research paper
├── LICENSE
├── README.md
└── requirements.txt
```

## Model Training

To train the quantum neural network for anomaly detection:

```bash
# Run training with default parameters
python src/main.py --mode train

# Run training with custom parameters
python src/main.py --mode train --epochs 12 --batch_size 128 --learning_rate 0.001

# Resume training from checkpoint
python src/main.py --mode train --resume --checkpoint src/checkpoints/last_checkpoint.pt
```

Training checkpoints will be saved to `src/checkpoints/` for later use or evaluation.

### Training Process

The training process follows Algorithm 1 from the paper:

1. **Data Loading**: Load network traffic data from the CESNET dataset
2. **Preprocessing**: Clean data, remove outliers using IQR method, and apply feature normalization
3. **Quantum Encoding**: Transform classical features into quantum states using amplitude or angle encoding
4. **Three-Class Classification**:
   - Class 1: Normative traffic (ŷqi ≤ γq1)
   - Class 2: Potentially suspicious traffic (γq1 < ŷqi ≤ γq2)
   - Class 3: Potentially malicious traffic (ŷqi > γq2)
5. **QNN Training**: Train the variational quantum circuit using gradient descent
6. **Dynamic Threshold Optimization**: Adjust γq1 and γq2 using feedback functions
7. **Risk Score Calculation**: Compute Rq(u,d) for each user-device pair
8. **Policy Adjustment**: Update segment policies P'(Sj) based on anomaly scores

## Evaluation

To evaluate the model on test data:

```bash
# Run evaluation
python src/main.py --mode evaluate --checkpoint src/checkpoints/best_model.pt

# Run evaluation with custom test data
python src/main.py --mode evaluate --checkpoint src/checkpoints/best_model.pt --test_data path/to/test_data

# Export evaluation results
python src/main.py --mode evaluate --checkpoint src/checkpoints/best_model.pt --export_results
```

Evaluation results will be generated in the `src/plots/` directory, including ROC curves, confusion matrices, and performance metrics.

## Security Framework

The zero-trust security framework is implemented in the `src/zero_trust/` directory. This comprehensive security model includes:

- **Three-Class Classification**: Precise categorization of network traffic
- **Dynamic Risk Scoring**: Real-time computation of Rq(u,d) scores
- **Adaptive Micro-Segmentation**: Dynamic adjustment of segment policies P'(Sj)
- **Continuous Validation**: Real-time verification of system components
- **Least Privilege Access**: Components only have access to necessary resources

The security framework integrates with the QNN-based anomaly detection to provide a comprehensive security solution for next-generation networks.

## Results

Our QNN-enhanced ZTA achieves significant improvements over classical approaches:

| Metric | QNN-ZTA | Classical Baseline | Improvement |
|--------|----------------------|----------------------|-------------|
| Detection Accuracy | 87.4% | 82.7% | +4.7% |
| AUC Score | 0.985 | 0.937 | +0.048 |
| False Positive Rate | 2.3% | 5.8% | -3.5% |
| Processing Latency | 8.2ms | 12.7ms | -4.5ms |
| Attack Surface Reduction | 78.3% | 45.6% | +32.7% |

Key achievements include:
- Successful implementation of three-class classification with high accuracy
- Dynamic threshold optimization with effective TPR/FPR balance
- Adaptive micro-segmentation with efficient threat containment
- Significant reduction in training time with GPU acceleration (87.4% improvement)
- Scalable implementation suitable for large-scale network environments

## Contributing

We welcome contributions to improve the quantum-driven ZTA system:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite our paper:

```
@article{ahmed2025quantum,
  title={Quantum-driven ZTA with Dynamic Anomaly Detection in 7G Technology: A Neural Network Approach},
  author={Ahmed, S. and others},
  journal={SoftwareX},
  volume={4},
  pages={1--21},
  year={2025}
}
```

## Contact

Project Maintainer - [ibnfarabishihab@gmail.com](mailto:ibnfarabishihab@gmail.com)

Repository: [https://github.com/farabi1038/qnn_project.git](https://github.com/farabi1038/qnn_project.git)
