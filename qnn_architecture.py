# qnn_project/qnn_architecture.py

import numpy as np
import qutip as qt
from copy import deepcopy
from quantum_utils import QuantumUtils
from logger import logger


class QNNArchitecture:
    """
    QNN Architecture functions: building networks, feedforward, cost function, 
    training logic, etc.
    """

    @staticmethod
    def random_network(qnn_arch: list, num_training_pairs: int):
        """
        Create a random QNN network based on the architecture and training pairs.
        Returns:
          (qnn_arch, unitaries, training_data, exact_unitary)
        """
        # Architecture example: [num_qubits_in, ..., num_qubits_out]
        # Must start and end with the same dimension for this example.
        assert qnn_arch[0] == qnn_arch[-1], "Not a valid QNN-Architecture."

        # The 'exact_unitary' is used to generate training data
        network_unitary = QuantumUtils.random_qubit_unitary(qnn_arch[-1])
        network_training_data = QNNArchitecture.random_training_data(
            network_unitary, num_training_pairs
        )

        # Build unitaries layer by layer
        network_unitaries = [[]]
        for l in range(1, len(qnn_arch)):
            num_input_qubits = qnn_arch[l - 1]
            num_output_qubits = qnn_arch[l]
            network_unitaries.append([])
            for j in range(num_output_qubits):
                unitary = QuantumUtils.random_qubit_unitary(num_input_qubits + 1)
                # If multiple output qubits, then apply identity to the rest
                if num_output_qubits - 1 != 0:
                    unitary = qt.tensor(
                        QuantumUtils.random_qubit_unitary(num_input_qubits + 1),
                        QuantumUtils.tensored_id(num_output_qubits - 1)
                    )
                # Swap the new qubit into the correct position
                unitary = QuantumUtils.swapped_op(unitary, num_input_qubits, num_input_qubits + j)
                network_unitaries[l].append(unitary)

        return (qnn_arch, network_unitaries, network_training_data, network_unitary)

    @staticmethod
    def random_training_data(unitary: qt.Qobj, N: int):
        """
        Generate training data based on a given unitary.
        Each data pair is (random_state, unitary_applied_state).
        """
        num_qubits = len(unitary.dims[0])
        training_data = []
        for _ in range(N):
            t = QuantumUtils.random_qubit_state(num_qubits)
            ut = unitary * t
            if not isinstance(t, qt.Qobj) or not isinstance(ut, qt.Qobj):
                raise ValueError("Training data contains non-Qobj elements.")
            training_data.append([t, ut])
        return training_data

    @staticmethod
    def feedforward(qnn_arch: list, unitaries: list, training_data: list):
        """
        Feed forward the input states through all layers.
        Returns a list of layerwise states for each training sample.
        """
        stored_states = []
        for x in range(len(training_data)):
            current_state = training_data[x][0] * training_data[x][0].dag()  # density matrix
            layerwise_list = [current_state]
            for l in range(1, len(qnn_arch)):
                current_state = QNNArchitecture.make_layer_channel(
                    qnn_arch, unitaries, l, current_state
                )
                layerwise_list.append(current_state)
            stored_states.append(layerwise_list)
        return stored_states

    @staticmethod
    def make_layer_channel(qnn_arch: list, unitaries: list, l: int, input_state: qt.Qobj):
        """
        Single layer channel: 
        tensor input with |0> states for new qubits, apply unitaries, 
        then partial trace out the input qubits to get the output qubits only.
        """
        num_input_qubits = qnn_arch[l - 1]
        num_output_qubits = qnn_arch[l]
        if input_state.type == 'ket':
            input_state = input_state * input_state.dag()  # convert ket to density matrix

        state = qt.tensor(input_state, QuantumUtils.tensored_qubit0(num_output_qubits))
        layer_uni = unitaries[l][0].copy()
        for i in range(1, num_output_qubits):
            layer_uni = unitaries[l][i] * layer_uni
        result = layer_uni * state * layer_uni.dag()
        # keep only output qubits
        return result.ptrace(list(range(num_input_qubits, num_input_qubits + num_output_qubits)))

    @staticmethod
    def make_adjoint_layer_channel(qnn_arch: list, unitaries: list, l: int, output_state: qt.Qobj):
        """
        Adjoint channel (backward pass).
        """
        num_input_qubits = qnn_arch[l - 1]
        num_output_qubits = qnn_arch[l]
        input_id = QuantumUtils.tensored_id(num_input_qubits)
        state1 = qt.tensor(input_id, QuantumUtils.tensored_qubit0(num_output_qubits))
        state2 = qt.tensor(input_id, output_state)

        layer_uni = unitaries[l][0].copy()
        for i in range(1, num_output_qubits):
            layer_uni = unitaries[l][i] * layer_uni

        return QuantumUtils.partial_trace_keep(
            state1 * layer_uni.dag() * state2 * layer_uni,
            list(range(num_input_qubits))
        )

    @staticmethod
    def cost_function(training_data: list, output_states: list) -> float:
        """
        Compute average cost = average fidelity (or a related measure).
        """
        cost_sum = 0
        for i in range(len(training_data)):
            t = training_data[i][1]   # target state
            o = output_states[i]      # output state
            if not isinstance(t, qt.Qobj) or not isinstance(o, qt.Qobj):
                raise ValueError(f"TrainingData[{i}] or OutputState[{i}] is not a Qobj.")
            term = (t.dag() * o * t)
            # If term is Qobj, get trace; else it might be a scalar
            if isinstance(term, qt.Qobj):
                cost_sum += term.tr().real
            else:
                cost_sum += term.real
        return cost_sum / len(training_data)

    @staticmethod
    def qnn_training(qnn_arch: list,
                     initial_unitaries: list,
                     training_data: list,
                     lda: float,
                     ep: float,
                     training_rounds: int,
                     alert: int = 0):
        """
        Main QNN training loop. 
        Returns [plotlist, trained_unitaries].
        """
        current_unitaries = initial_unitaries
        # feed forward once at start
        stored_states = QNNArchitecture.feedforward(qnn_arch, current_unitaries, training_data)
        output_states = [states[-1] for states in stored_states]

        s = 0  # for plotting x-axis
        plotlist = [[s], [QNNArchitecture.cost_function(training_data, output_states)]]

        for k in range(training_rounds):
            if alert > 0 and k % alert == 0:
                print(f"In training round {k}")

            new_unitaries = [deepcopy(layer) for layer in current_unitaries]

            # For each layer
            for l in range(1, len(qnn_arch)):
                num_output_qubits = qnn_arch[l]
                for j in range(num_output_qubits):
                    update_mat = QNNArchitecture.make_update_matrix_tensored(
                        qnn_arch, current_unitaries, training_data, stored_states,
                        lda, ep, l, j
                    )
                    new_unitaries[l][j] = update_mat * current_unitaries[l][j]

            s += ep
            current_unitaries = new_unitaries
            # re-feedforward
            stored_states = QNNArchitecture.feedforward(qnn_arch, current_unitaries, training_data)
            output_states = [layer_list[-1] for layer_list in stored_states]

            plotlist[0].append(s)
            plotlist[1].append(QNNArchitecture.cost_function(training_data, output_states))

        print(f"Trained {training_rounds} rounds for a {qnn_arch} network and {len(training_data)} training pairs")
        return [plotlist, current_unitaries]

    # ---------------------------------------------------------------------
    # The methods below break out the update matrix logic
    # ---------------------------------------------------------------------
    @staticmethod
    def make_update_matrix_tensored(qnn_arch, unitaries, training_data, stored_states,
                                    lda, ep, l, j):
        num_input_qubits = qnn_arch[l - 1]
        num_output_qubits = qnn_arch[l]
        res = QNNArchitecture.make_update_matrix(
            qnn_arch, unitaries, training_data, stored_states, lda, ep, l, j
        )
        if num_output_qubits - 1 != 0:
            res = qt.tensor(res, QuantumUtils.tensored_id(num_output_qubits - 1))
        return QuantumUtils.swapped_op(res, num_input_qubits, num_input_qubits + j)

    @staticmethod
    def make_update_matrix(qnn_arch, unitaries, training_data, stored_states,
                           lda, ep, l, j):
        """
        Summation of commutators over the training samples, then exponentiated.
        """
        num_input_qubits = qnn_arch[l - 1]
        summ = 0
        for x in range(len(training_data)):
            first_part = QNNArchitecture.update_matrix_first_part(
                qnn_arch, unitaries, stored_states, l, j, x
            )
            second_part = QNNArchitecture.update_matrix_second_part(
                qnn_arch, unitaries, training_data, l, j, x
            )
            mat = qt.commutator(first_part, second_part)
            # keep the relevant qubits
            keep = list(range(num_input_qubits))
            keep.append(num_input_qubits + j)
            mat = QuantumUtils.partial_trace_keep(mat, keep)
            summ += mat
        # final exponent
        summ = (-ep * (2**num_input_qubits) / (lda * len(training_data))) * summ
        return summ.expm()

    @staticmethod
    def update_matrix_first_part(qnn_arch, unitaries, stored_states, l, j, x):
        num_input_qubits = qnn_arch[l - 1]
        num_output_qubits = qnn_arch[l]
        state = qt.tensor(
            stored_states[x][l - 1],
            QuantumUtils.tensored_qubit0(num_output_qubits)
        )
        product_uni = unitaries[l][0]
        for i in range(1, j + 1):
            product_uni = unitaries[l][i] * product_uni
        return product_uni * state * product_uni.dag()

    @staticmethod
    def update_matrix_second_part(qnn_arch, unitaries, training_data, l, j, x):
        num_input_qubits = qnn_arch[l - 1]
        num_output_qubits = qnn_arch[l]

        # target state
        state = training_data[x][1] * training_data[x][1].dag()
        # propagate backward from top layer down to l+1
        for i in range(len(qnn_arch) - 1, l, -1):
            state = QNNArchitecture.make_adjoint_layer_channel(qnn_arch, unitaries, i, state)

        state = qt.tensor(QuantumUtils.tensored_id(num_input_qubits), state)
        product_uni = QuantumUtils.tensored_id(num_input_qubits + num_output_qubits)
        for i in range(j + 1, num_output_qubits):
            product_uni = unitaries[l][i] * product_uni
        return product_uni.dag() * state * product_uni
