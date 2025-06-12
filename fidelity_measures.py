
import numpy as np
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile


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


def circuit_fidelity(n_qb, qc_exact, qc_noisy, noise_model=None):
    """
    Compare the fidelity between an ideal quantum circuit and its noisy/decomposed counterpart.

    This function simulates both the ideal quantum circuit and a noisy or decomposed version of it, 
    measures all qubits in both circuits, and computes the fidelity between the resulting output 
    states (in terms of density matrices). It returns the fidelity along with the measurement 
    counts from both circuits, which can be used for further analysis or visualization.

    Args:
        n_qb (int): Number of qubits in the circuits.
        qc_exact (QuantumCircuit): The ideal quantum circuit to compare against.
        qc_noisy (QuantumCircuit): The noisy or decomposed version of the circuit.
        noise_model (NoiseModel, optional): A Qiskit noise model to simulate physical imperfections. 
            If None, the noisy circuit is simulated ideally but using the native gate set.

    Returns:
        tuple:
            - float: The fidelity between the two circuits' output states.
            - dict: Measurement counts from the ideal circuit.
            - dict: Measurement counts from the noisy/decomposed circuit.
    """
    
    # Add measurement operations to all qubits in both circuits
    for n in range(n_qb):
        qc_exact.measure(n, n)
        qc_noisy.measure(n, n)

    # Simulate the ideal circuit without noise
    simulator = AerSimulator()
    compiled_circuit_exact = transpile(qc_exact, simulator)
    result_exact = simulator.run(compiled_circuit_exact).result()
    counts_exact = result_exact.get_counts()

    # Simulate the noisy or decomposed circuit
    simulator_noisy = AerSimulator(noise_model=noise_model)
    compiled_circuit_decomp = transpile(qc_noisy, simulator_noisy)
    result_decomp = simulator_noisy.run(compiled_circuit_decomp).result()
    counts_decomp = result_decomp.get_counts()

    # Convert results into density matrices for fidelity computation
    rho = create_density_matrix(n_qb, result_exact)   # Ideal circuit state
    sigma = create_density_matrix(n_qb, result_decomp)  # Noisy circuit state

    # Compute fidelity between ideal and noisy density matrices
    fid = UJ_fidelity(rho.data, sigma.data).real

    return fid, counts_exact, counts_decomp