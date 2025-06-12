import numpy as np
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from collections import defaultdict
from fidelity_measures import *


def make_circuit(angles, basis_count, circuit, m, n):
    """
    Builds a quantum circuit from Euler angles and a specified number of two-qubit basis gates.

    The function applies a sequence of single-qubit Euler rotations and interleaves them with 
    RXX(π/2) gates according to the specified `basis_count`. Each level of decomposition includes 
    an additional RXX gate and associated X and Y rotations of the qubits. 

    Args:
        angles (dict): A nested dictionary of the form:
            {
                'q0': {'theta': [...], 'phi': [...], 'xi': [...]},
                'q1': {'theta': [...], 'phi': [...], 'xi': [...]}
            }
            Each list contains the angles for the respective gates at different levels of decomposition.
        basis_count (int): The number of RXX(π/2) basis gates to include in the circuit. Must be ≥ 0.
        circuit (QuantumCircuit): The circuit object to which the gates will be appended.
        m (int): Index of the first qubit.
        n (int): Index of the second qubit.

    Returns:
        QuantumCircuit: The modified circuit with the applied gates.
    """

    # Apply gates in reverse order: from highest basis count down to 0
    # At each stage, apply Euler angles (if non-zero) followed by RXX gate if applicable

    if basis_count >= 3:
        # Single-qubit Euler rotations for both qubits
        if angles['q0']['theta'][3] != 0.0:
            circuit.rx(angles['q0']['theta'][3], m)
        if angles['q0']['phi'][3] != 0.0:
            circuit.ry(angles['q0']['phi'][3], m)
        if angles['q0']['xi'][3] != 0.0:
            circuit.rx(angles['q0']['xi'][3], m)

        if angles['q1']['theta'][3] != 0.0:
            circuit.rx(angles['q1']['theta'][3], n)
        if angles['q1']['phi'][3] != 0.0:
            circuit.ry(angles['q1']['phi'][3], n)
        if angles['q1']['xi'][3] != 0.0:
            circuit.rx(angles['q1']['xi'][3], n)

        # Two-qubit entangling gate
        circuit.rxx(np.pi / 2, m, n)

    if basis_count >= 2:
        if angles['q0']['theta'][2] != 0.0:
            circuit.rx(angles['q0']['theta'][2], m)
        if angles['q0']['phi'][2] != 0.0:
            circuit.ry(angles['q0']['phi'][2], m)
        if angles['q0']['xi'][2] != 0.0:
            circuit.rx(angles['q0']['xi'][2], m)

        if angles['q1']['theta'][2] != 0.0:
            circuit.rx(angles['q1']['theta'][2], n)
        if angles['q1']['phi'][2] != 0.0:
            circuit.ry(angles['q1']['phi'][2], n)
        if angles['q1']['xi'][2] != 0.0:
            circuit.rx(angles['q1']['xi'][2], n)

        circuit.rxx(np.pi / 2, m, n)

    if basis_count >= 1:
        if angles['q0']['theta'][1] != 0.0:
            circuit.rx(angles['q0']['theta'][1], m)
        if angles['q0']['phi'][1] != 0.0:
            circuit.ry(angles['q0']['phi'][1], m)
        if angles['q0']['xi'][1] != 0.0:
            circuit.rx(angles['q0']['xi'][1], m)

        if angles['q1']['theta'][1] != 0.0:
            circuit.rx(angles['q1']['theta'][1], n)
        if angles['q1']['phi'][1] != 0.0:
            circuit.ry(angles['q1']['phi'][1], n)
        if angles['q1']['xi'][1] != 0.0:
            circuit.rx(angles['q1']['xi'][1], n)

        circuit.rxx(np.pi / 2, m, n)

    if basis_count >= 0:
        if angles['q0']['theta'][0] != 0.0:
            circuit.rx(angles['q0']['theta'][0], m)
        if angles['q0']['phi'][0] != 0.0:
            circuit.ry(angles['q0']['phi'][0], m)
        if angles['q0']['xi'][0] != 0.0:
            circuit.rx(angles['q0']['xi'][0], m)

        if angles['q1']['theta'][0] != 0.0:
            circuit.rx(angles['q1']['theta'][0], n)
        if angles['q1']['phi'][0] != 0.0:
            circuit.ry(angles['q1']['phi'][0], n)
        if angles['q1']['xi'][0] != 0.0:
            circuit.rx(angles['q1']['xi'][0], n)

    # return the modified circuit
    return circuit

def process_circuit_angles(target, basis_gate, euler_basis, num_basis_gates):
    """
    Decompose a two-qubit unitary into a sequence of basis gates and then extract Euler angles.

    This function uses a specified two-qubit basis gate (e.g., RXX, CNOT) and an Euler angle
    basis (e.g. 'XYX') to decompose a target unitary operation. It then parses the 
    resulting quantum circuit to extract (in this case) RX and RY rotation angles for each qubit in 
    the decomposition.

    Args:
        target (Operator or np.ndarray): The 4x4 unitary matrix to be decomposed.
        basis_gate (Gate): A two-qubit basis gate to use in the decomposition (e.g., RXXGate()).
        euler_basis (str): The basis for decomposing single-qubit gates (e.g. 'XYX').
        num_basis_gates (int): Number of basis gate uses to allow in the decomposition. Between 0 and 3.

    Returns:
        dict: A dictionary of the form:
            {
                'q0': {'theta': [...], 'phi': [...], 'xi': [...]},
                'q1': {'theta': [...], 'phi': [...], 'xi': [...]}
            }
            where each list contains the Euler angles (in reversed order) for that qubit.
    """
    
    # Decompose the target unitary into a quantum circuit
    decomp = TwoQubitBasisDecomposer(basis_gate, euler_basis=euler_basis)
    qc = decomp(target, _num_basis_uses=num_basis_gates) # use the specified number of basis gates

    # Dictionary to store Euler angles for each qubit
    angles = defaultdict(lambda: {'theta': [], 'phi': [], 'xi': []}) # theta, phi, xi angles for each qubit for RX, RY, and RX gates respectively
    
    # Map Qiskit internal qubit identifiers to labels like 'q0', 'q1'
    qubit_label_map = {}
    qubit_counter = 0

    # Expected sequence of single-qubit gates used in Euler decomposition (modify if using a different basis)
    rotation_sequence = ['rx', 'ry', 'rx']

    # Qiskit will omit a single qubit rotation if it is not needed, so we need to handle this
    # by checking the sequence of operations and then if an expected rotation is missing, we replace it with a rotation of angle 0.
    # This keeps out lists the same length and allows us to process the circuit correctly.


    # Track which qubit is currently being processed
    current_qubit = None

    # Temporary storage for a single Euler rotation sequence (up to 3 angles)
    temp_angles = []

    # Index to track which rotation (rx, ry, rx) we expect next
    rotation_index = 0

    # Iterate over all instructions in the quantum circuit
    for instruction in qc.data:
        # Check if the instruction is a rotation (rx or ry)
        if instruction.operation.name in ['rx', 'ry']:
            # Get a unique identifier for the qubit (used for labeling)
            qubit_uid = str(instruction.qubits[0])

            # Map each qubit UID to a label like 'q0', 'q1' only once
            if qubit_uid not in qubit_label_map:
                qubit_label_map[qubit_uid] = f"q{qubit_counter}"
                qubit_counter += 1
            qubit_label = qubit_label_map[qubit_uid]

            # Extract the rotation angle from the instruction
            angle = instruction.operation.params[0]

            # If we've switched to a new qubit or finished a 3-angle rotation
            if current_qubit != qubit_label or rotation_index == 3:
                if temp_angles and current_qubit is not None:
                    # Pad with zeros if we don't have all 3 expected angles
                    while rotation_index < 3:
                        temp_angles.append(0.0)
                        rotation_index += 1
                    # Store the completed set of angles into the angles dictionary
                    angles[current_qubit]['theta'].append(temp_angles[0])
                    angles[current_qubit]['phi'].append(temp_angles[1])
                    angles[current_qubit]['xi'].append(temp_angles[2])
                # Reset for the new qubit or next sequence
                current_qubit = qubit_label
                temp_angles = []
                rotation_index = 0

            # Check which rotation is expected next based on the rotation sequence
            expected_rotation = rotation_sequence[rotation_index]

            if instruction.operation.name == expected_rotation:
                # If the current gate matches the expected one, store the angle
                temp_angles.append(angle)
                rotation_index += 1
            else:
                # If it's not the expected rotation, assume a missing gate and pad with 0.0
                temp_angles.append(0.0)
                rotation_index += 1
                # Check again if the current gate matches the *next* expected rotation
                if rotation_index < 3 and instruction.operation.name == rotation_sequence[rotation_index]:
                    temp_angles.append(angle)
                    rotation_index += 1
                else:
                    continue  # Skip this instruction if the sequence is not valid

            # If we’ve collected three angles (rx, ry, rx), store them
            if rotation_index == 3:
                angles[current_qubit]['theta'].append(temp_angles[0])
                angles[current_qubit]['phi'].append(temp_angles[1])
                angles[current_qubit]['xi'].append(temp_angles[2])
                temp_angles = []
                rotation_index = 0

    # After finishing the loop, check if there are any leftover angles to store
    if temp_angles and current_qubit is not None:
        while rotation_index < 3:
            temp_angles.append(0.0)
            rotation_index += 1
        angles[current_qubit]['theta'].append(temp_angles[0])
        angles[current_qubit]['phi'].append(temp_angles[1])
        angles[current_qubit]['xi'].append(temp_angles[2])

    # Qiskit stores operations in reverse order (last instruction first),
    # so we reverse the angle lists to match the actual application order
    for qubit in angles:
        angles[qubit]['theta'].reverse()
        angles[qubit]['phi'].reverse()
        angles[qubit]['xi'].reverse()

    return dict(angles)


def optimal_basis_gate_number(target, basis_gate, euler_basis, noise_model=None):
    """
    Determines the optimal number of applications of a given basis gate to decompose a target 2-qubit unitary,
    with and without precomposing it with a SWAP gate. Returns the configuration that yields the highest fidelity.
    
    Args:
        target: Operator representing a 2-qubit unitary to be decomposed.
        basis_gate: Basis gate used for decomposition.
        euler_basis: Basis used for Euler angle decomposition.
        noise_model: Optional Qiskit noise model.
    
    Returns:
        best_n: Number of basis gate applications for the best fidelity.
        best_angles: Corresponding Euler angles for the best decomposition.
        used_swap: Boolean indicating whether a SWAP was composed with the target.
    """
    # Define the number of qubits for the unitary operations.
    n_qb = 2  # Fixed to 2-qubit unitaries.


    # Initialize variables to track the best decomposition found.
    best_fidelity = -1.0  # Stores the highest fidelity achieved.
    best_n = None  # Stores the number of basis gates for the best decomposition.
    best_angles = None  # Stores the angles for the basis gates in the best decomposition.

    unitary = target 

    # Create an 'exact' quantum circuit representing the ideal target unitary
    # (potentially after a SWAP). This serves as the reference for fidelity calculation.
    qc_exact = QuantumCircuit(n_qb, n_qb)
    qc_exact.unitary(unitary, [0, 1], label='Target Unitary')

    # Attempt decompositions using 0 to 3 applications of the basis gate.
    for num_basis_gates in range(4):
        # Create a quantum circuit for the decomposed unitary.
        qc_decomp = QuantumCircuit(n_qb, n_qb)

        # Calculate the specific angles for the basis gates to approximate the target unitary.
        angles = process_circuit_angles(unitary, basis_gate, euler_basis, num_basis_gates)

        # Construct the decomposed circuit using the calculated angles and basis gates.
        make_circuit(angles, num_basis_gates, qc_decomp, 0, 1)

        # Calculate the fidelity between the exact target circuit and the decomposed circuit including noise
        fidelity, _, _ = circuit_fidelity(n_qb, qc_exact.copy(), qc_decomp, noise_model=noise_model)

        # Check if the current decomposition yields a higher fidelity than the best found so far.
        if fidelity > best_fidelity:
            # Update the best results if a higher fidelity is achieved.
            best_fidelity = fidelity
            best_n = num_basis_gates
            best_angles = angles

    # Return the parameters of the best decomposition found across all trials.
    return best_n, best_angles
