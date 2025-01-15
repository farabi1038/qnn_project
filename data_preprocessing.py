# qnn_project/data_preprocessing.py

import numpy as np
import pandas as pd
import qutip as qt

from .quantum_utils import QuantumUtils


class DataPreprocessing:
    """
    Data loading, preprocessing, and quantum encoding utilities.
    """

    @staticmethod
    def load_cesnet_data(file_path, num_samples=5):
        """
        Load CESNET data from CSV, return normalized data and original dataframe.
        """
        df = pd.read_csv(file_path)
        df = df.head(num_samples)
        features = [
            'n_flows', 'n_packets', 'n_bytes',
            'n_dest_ip', 'n_dest_ports',
            'tcp_udp_ratio_packets',
            'dir_ratio_packets',
            'avg_duration'
        ]
        data = df[features].values

        # Avoid division by zero for near-constant features
        std = np.std(data, axis=0)
        std = np.where(std < 1e-10, 1e-10, std)

        normalized_data = (data - np.mean(data, axis=0)) / std
        return normalized_data, df

    @staticmethod
    def preprocess_features(df):
        """
        Basic feature engineering example.
        """
        df['bytes_per_packet'] = df['n_bytes'] / df['n_packets']
        df['packets_per_flow'] = df['n_packets'] / df['n_flows']
        df['ports_per_ip'] = df['n_dest_ports'] / df['n_dest_ip']
        return df

    @staticmethod
    def encode_traffic_to_quantum(normalized_data):
        """
        Convert normalized traffic data to an array of quantum states.
        Each row is normalized; we map it to a quantum state vector.
        """
        quantum_states = []
        for sample in normalized_data:
            norm = np.linalg.norm(sample)
            if norm != 0:
                quantum_state = sample / norm
            else:
                quantum_state = sample

            # Number of qubits needed to embed 'len(sample)' features
            num_qubits = int(np.ceil(np.log2(len(sample))))
            padded_vec = np.zeros(2**num_qubits, dtype=np.complex128)

            # Place the normalized features in the first len(sample) slots
            padded_vec[:len(sample)] = quantum_state
            padded_vec = padded_vec / np.linalg.norm(padded_vec)

            # Create ket
            qobj_state = qt.Qobj(padded_vec.reshape(-1, 1),
                                 dims=[[2]*num_qubits, [1]*num_qubits])
            quantum_states.append(qobj_state)
        return quantum_states

    @staticmethod
    def prepare_training_data(quantum_states, noise_level=0.1):
        """
        Prepare training data pairs of (clean_state, noisy_target_state).
        """
        training_data = []
        for state in quantum_states:
            noisy_target = QuantumUtils.noisy_state(state, noise_level)
            training_data.append([state, noisy_target])
        return training_data
