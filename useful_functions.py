import numpy as np
from qiskit.quantum_info import Operator

import numpy as np

def UJ_fidelity(state_a, state_b):
    """Computes the Uhlmann-Jozsa fidelity between two quantum states.

    The fidelity quantifies how close two density matrices are. It is defined as:

        F(A, B) = (Tr(sqrt(sqrt(A) * B * sqrt(A))))^2

    This measure ranges from 0 (orthogonal states) to 1 (identical states).

    Args:
        state_a: A NumPy array representing the first density matrix.
        state_b: A NumPy array representing the second density matrix.

    Returns:
        A float representing the fidelity between `state_a` and `state_b`.
    """
    sqrt_a = np.sqrt(state_a)
    product = sqrt_a @ state_b @ sqrt_a
    fidelity = np.trace(np.sqrt(product)) ** 2
    return fidelity.real


def create_density_matrix(num_qubits, result, shot_count=1024):
    """Constructs a diagonal density matrix from measurement results.

    The function assumes that the state is classical (i.e., already measured) and builds 
    a diagonal density matrix based on the observed counts, normalized by `shot_count`.

    Args:
        num_qubits: Number of qubits in the system.
        result: Qiskit Result object from circuit execution.
        shot_count: Total number of measurement shots used in the simulation.

    Returns:
        A `qiskit.quantum_info.Operator` instance representing the diagonal density matrix.
    """
    dim = 2 ** num_qubits
    density_matrix = np.zeros((dim, dim), dtype=complex)

    counts = result.get_counts(0)
    for i in range(dim):
        bitstring = format(i, f"0{num_qubits}b")
        if bitstring in counts:
            probability = counts[bitstring] / shot_count
            density_matrix[i, i] = probability

    return Operator(density_matrix)

