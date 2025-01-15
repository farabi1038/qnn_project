# qnn_project/quantum_utils.py

import numpy as np
import scipy as sc
import qutip as qt


# Predefined basis states (common in many quantum scripts)
qubit0 = qt.basis(2, 0)
qubit1 = qt.basis(2, 1)
qubit0mat = qubit0 * qubit0.dag()
qubit1mat = qubit1 * qubit1.dag()


class QuantumUtils:
    """
    A collection of utility functions for quantum operations, using QuTiP objects.
    """

    @staticmethod
    def partial_trace_keep(obj: qt.Qobj, keep: list) -> qt.Qobj:
        """
        Keep partial trace over specified subsystem indices.
        """
        if len(keep) == len(obj.dims[0]):
            return obj
        return obj.ptrace(keep)

    @staticmethod
    def partial_trace_remove(obj: qt.Qobj, remove: list) -> qt.Qobj:
        """
        Remove partial trace over specified subsystem indices.
        """
        keep = list(range(len(obj.dims[0])))
        for x in sorted(remove, reverse=True):
            keep.pop(x)
        if len(keep) == len(obj.dims[0]):
            return obj
        return obj.ptrace(keep)

    @staticmethod
    def swapped_op(obj: qt.Qobj, i: int, j: int) -> qt.Qobj:
        """
        Permute the subsystem ordering by swapping subsystem i and j.
        """
        if i == j:
            return obj
        permute = list(range(len(obj.dims[0])))
        permute[i], permute[j] = permute[j], permute[i]
        return obj.permute(permute)

    @staticmethod
    def tensored_id(num_qubits: int) -> qt.Qobj:
        """
        Return the identity operator acting on a tensor product space of dimension 2^num_qubits.
        """
        res = qt.qeye(2**num_qubits)
        dims = [[2] * num_qubits, [2] * num_qubits]
        res.dims = dims
        return res

    @staticmethod
    def tensored_qubit0(num_qubits: int) -> qt.Qobj:
        """
        Return the density operator |0...0><0...0| for num_qubits qubits.
        """
        # The state |0...0> is a single basis vector in a 2^N dimensional space.
        state = qt.basis(2**num_qubits, 0)
        proj = state * state.dag()
        dims = [[2] * num_qubits, [2] * num_qubits]
        proj.dims = dims
        return proj

    @staticmethod
    def random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        """
        Generate a random unitary for a system of 'num_qubits' qubits.
        """
        dim = 2**num_qubits
        mat = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        mat = sc.linalg.orth(mat)
        res = qt.Qobj(mat)
        dims = [[2] * num_qubits, [2] * num_qubits]
        res.dims = dims
        return res

    @staticmethod
    def random_qubit_state(num_qubits: int) -> qt.Qobj:
        """
        Generate a random pure state for 'num_qubits' qubits.
        """
        dim = 2**num_qubits
        vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        vec = vec / np.linalg.norm(vec)
        res = qt.Qobj(vec)
        dims = [[2] * num_qubits, [1] * num_qubits]
        res.dims = dims
        return res

    @staticmethod
    def noisy_state(input_state: qt.Qobj, noise_level: float = 0.1) -> qt.Qobj:
        """
        Add random noise to a state vector or density operator, then re-normalize.
        """
        noise = (np.random.normal(size=input_state.shape) 
                 + 1j * np.random.normal(size=input_state.shape)) * noise_level
        noisy_state = input_state.full() + noise
        noisy_state = noisy_state / np.linalg.norm(noisy_state)
        return qt.Qobj(noisy_state, dims=input_state.dims)
