# Standard library imports
import random
from collections import defaultdict

# Third-party library imports
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer

# Qiskit Aer imports
from qiskit_aer import AerSimulator

# Custom module imports
from fidelity_measures import *


def make_circuit(angles, basis_count, circuit, m, n):
    """
    Applies gates within a quantum circuit to specific qubits 
    from Euler angles and a specified number of two-qubit basis gates.

    The function applies a sequence of single-qubit Euler rotations and interleaves them with 
    RXX(π/2) two-qubit gates according to the specified `basis_count`.
    The angles for the single-qubit rotations are provided in a nested dictionary format, 
    where each qubit has its own set of angles for each level of decomposition.
    The function assumes that the angles are provided in a specific order for each qubit,
    corresponding to the RX, RY, and RX gates in the Euler decomposition.
    The function applies the gates in reverse order of decomposition, starting from the highest level down to 0.
    This is to ensure that the circuit is built correctly according to the decomposition levels.

    Args:
        angles (dict): A nested dictionary of the form:
            {
                'q0': {'theta': [...], 'phi': [...], 'xi': [...]},
                'q1': {'theta': [...], 'phi': [...], 'xi': [...]}
            }
            Each list contains the angles for the respective gates at different levels of decomposition.
            theta: angle for RX gate,
            phi: angle for RY gate,
            xi: angle for RX gate.
        basis_count (int): The number of RXX(π/2) basis gates to include in the circuit. Must be ≥ 0.
        circuit (QuantumCircuit): The circuit object to which the gates will be appended.
        m (int): Index of the first qubit.
        n (int): Index of the second qubit.

    Returns:
        QuantumCircuit: The modified circuit with the applied gates.
    """

    def apply_single_qubit_rotations(level):
        """
        Apply single-qubit rotations to the quantum circuit based on the angles at the specified level of decomposition.

        This function iterates over two qubits ('q0' and 'q1') and applies rotation gates (Rx and Ry) to each qubit 
        using the angles provided in the `angles` dictionary. The rotations are applied only if the angle is non-zero 
        to avoid adding unnecessary gates, which can introduce noise into the quantum circuit.

        Args:
            level (int): The current level of decomposition, used to extract the corresponding angles 
                         (theta, phi, xi) for each qubit from the `angles` dictionary.
        """

        # loop through the two qubits and apply rotations
        for q, qubit in zip(['q0', 'q1'], [m, n]):

            # extract angles from dictionary 
            theta = angles[q]['theta'][level]
            phi = angles[q]['phi'][level]
            xi = angles[q]['xi'][level]

            # apply rotations (providing the angle is non-zero, which avoids adding unnecessary gates and therefore nosie)
            if theta > 0 or theta < 0:
                circuit.rx(theta, qubit)
            if phi > 0 or phi < 0:
                circuit.ry(phi, qubit)
            if xi > 0 or xi < 0:
                circuit.rx(xi, qubit)

    # find the number of two-qubit basis gates to apply
    max_level = min(basis_count, 3)

    # Apply single-qubit rotations and two-qubit RXX gates in reverse order
    for level in reversed(range(max_level + 1)):
        # apply single-qubit rotations for the current level
        apply_single_qubit_rotations(level)
        
        # after applying single-qubit rotations, apply the two-qubit RXX gate (if applicable)
        if level >= 1:
            circuit.rxx(np.pi / 2, m, n)

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
    Determines the optimal number of applications of a given basis gate to decompose a target 2-qubit unitary. 
    Returns the configuration that yields the highest fidelity.
    If noise_model is provided, it will be used to calculate the fidelity of the decomposed circuit and choose the optimal decomposition.
    
    Args:
        target: Operator representing a 2-qubit unitary to be decomposed.
        basis_gate: Basis gate used for decomposition.
        euler_basis: Basis used for Euler angle decomposition.
        noise_model: Optional Qiskit noise model.
    
    Returns:
        best_n: Number of basis gate applications for the best fidelity.
        best_angles: Corresponding Euler angles for the best decomposition.
    """

    # Define the number of qubits for the unitary operations.
    n_qb = 2  # Fixed to 2-qubit unitaries.

    # Initialize variables to track the best decomposition found.
    best_fidelity = -1.0  # Stores the highest fidelity achieved.
    best_n = None  # Stores the number of basis gates for the best decomposition.
    best_angles = None  # Stores the angles for the basis gates in the best decomposition.

    # Create an 'exact' quantum circuit representing the ideal target unitary
    # This serves as the reference for fidelity calculation.
    qc_exact = QuantumCircuit(n_qb, n_qb)

    # apply the target unitary operation to the exact circuit
    qc_exact.unitary(target, [0, 1], label='Target Unitary')

    # Attempt decompositions using 0 to 3 applications of the basis gate.
    for num_basis_gates in range(4):
        
        # Create a quantum circuit for the decomposed/noisy unitary.
        qc_decomp = QuantumCircuit(n_qb, n_qb)

        # Calculate the specific angles for the basis gates to approximate the target unitary.
        angles = process_circuit_angles(target, basis_gate, euler_basis, num_basis_gates)

        # Construct the decomposed circuit using the calculated angles and basis gates.
        qc_decomp = make_circuit(angles, num_basis_gates, qc_decomp, 0, 1)

        # Calculate the fidelity between the exact circuit and the decomposed/noisy circuit
        fidelity, _, _ = circuit_fidelity(n_qb, qc_exact.copy(), qc_decomp, noise_model=noise_model)

        # Check if the current decomposition yields a higher fidelity than the best found so far.
        if fidelity > best_fidelity:
            # Update the best results if a higher fidelity is achieved.
            best_fidelity = fidelity
            best_n = num_basis_gates
            best_angles = angles
    
    # Optional print statement to show the best fidelity and number of basis gates
    # print(f'Best fidelity: {best_fidelity} with {best_n} basis gates')
    
    # Return the parameters of the best decomposition found across all trials.
    return best_n, best_angles


def compile(target, basis_gate, euler_basis, circuit, n, m, noise_model=None):
    """
    Compile a target unitary operation into a quantum circuit using a specified basis gate and Euler angles.
    This function decomposes the target unitary into a sequence of single-qubit rotations and two-qubit gates,
    and then applies these gates to the specified qubits in the provided quantum circuit.

    Args:
        target (Operator or np.ndarray): The target unitary operation to be decomposed.
        basis_gate (Gate): The two-qubit basis gate to use for decomposition (e.g., RXXGate()).
        euler_basis (str): The Euler angle basis to use for decomposition (e.g. 'XYX').
        circuit (QuantumCircuit): The quantum circuit to which the gates will be added.
        n (int): Index of the first qubit.
        m (int): Index of the second qubit.
        noise_model (NoiseModel, optional): A noise model to apply to the gates in the circuit. Defaults to None.
    Returns:
        QuantumCircuit: The modified quantum circuit with the decomposed gates applied.
    """

    # find the optimal decomposition for the target unitary, including noise if provided     
    optimal_basis_count, optimal_angles = optimal_basis_gate_number(target, basis_gate, euler_basis, noise_model)
    
    # add the compiled gates to the circuit using the optimal angles and basis count
    circuit = make_circuit(optimal_angles, optimal_basis_count, circuit, n, m)

    # return the circuits with the compiled gates added
    return circuit

def run_compiler(number_of_qubits, circuit_depth, basis_gate, euler_basis, total_noise_model=None, shuttle=True):
    """
    Create three quantum circuits: an ideal circuit, a decomposed circuit, and a noisy circuit.
    The ideal circuit contains no decomposition or noise, the decomposed circuit applies a decomposition
    of two-qubit gates, and the noisy circuit applies the same decomposition with noise to each gate.
    
    Args:
        number_of_qubits (int): The number of qubits in the quantum circuit.
        circuit_depth (int): The depth of the quantum circuit, i.e., the number of layers of gates (pre-compilation).
        basis_gate (Gate): The two-qubit basis gate to use for decomposition (e.g., RXXGate()).
        euler_basis (str): The Euler angle basis to use for decomposition (e.g. 'XYX').
        total_noise_model (NoiseModel, optional): A noise model to apply to the noisy circuit. Defaults to None.
        If None then the decomposed and noisy circuits will be the same.
        shuttle (bool): If True, adds an identity gate to each qubit between layers of depth to simulate shuttling.

        Including noise can lead to a different optimal decomposition than without noise, since, for example
        you may get a better decomposition with more basis gates, but if each basis gate has a lot of noise, then
        the overall fidelity may be lower than a decomposition with fewer basis gates but less noise.

    Returns:
        tuple: A tuple containing three QuantumCircuit objects:
            - qc_exact: The ideal quantum circuit with no decomposition or noise.
            - qc_decomposed: The decomposed quantum circuit.
            - qc_noisy: The noisy quantum circuit: same decomposition as qc_decomposed but with noise added to the gates.
    """

    #create ideal quantum circuit (no decomposition, no noise)
    qc_exact = QuantumCircuit(number_of_qubits,number_of_qubits)

    #create decomposed quantum circuit (decomposition and no noise)
    qc_decomposed = QuantumCircuit(number_of_qubits,number_of_qubits)

    #create noisy quantum circuit (decomposition and noise)
    qc_noisy = QuantumCircuit(number_of_qubits,number_of_qubits)

    # list of qubit labels, used for random pairing of qubits
    qb_label = np.arange(0, number_of_qubits).tolist() 

    #loop through depth of circuit
    for d in range(circuit_depth):

        print(f'--- Run {d+1} of {circuit_depth} ---')
        
        # shuffle the qubit labels to pair them randomly
        print(f'Shuffling qubits...')
        random.shuffle(qb_label)

        if shuttle: 
            print(f'Shuttling qubits...')
            for m in np.arange(0,number_of_qubits): 
                # loop through all qubits in exact (noisy) circuit and add an identity (rz(0)) for shuttling
                # this assumes that there is shuttling inbetween each layer of depth to get qubits next 
                # to each other in order to perform two qubit gates (e.g. in a trapped ion quantum computer)

                qc_exact.id(m) # shuttle each qubit in the ideal circuit

                # for the decomposed circuits, we use rz(0) to shuttle the qubits so that noise can be added to this gate to simulate decoherence
                qc_decomposed.rz(0, m) # shuttle each qubit in the decomposed circuit
                qc_noisy.rz(0, m) # shuttle each qubit in the noisy circuit
        
        # loop through the qubits in pairs
        print(f'Pairing qubits and applying two-qubit gates...')
        for n in np.arange(0,number_of_qubits,2):

            if len(qb_label) % 2 == 1 and n == number_of_qubits-1:
                # if there are an odd number of qubits, add an identity to the one that hasn't been paired
                # this will add some decoherence to this qubit (in the noisy circuit) while it is not being used
                print(f'Identity applied to Qubit {qb_label[n]}\n')
                qc_exact.id(qb_label[n])
                qc_decomposed.rz(0,(qb_label[n]))
                qc_noisy.rz(0, qb_label[n])

            # pair qubits together and create a random two-qubit gates between them
            else:
                print(f'Two qubit gate between qubits {qb_label[n]} and {qb_label[n+1]} ')

                #for each pair of qubits create a random two qubit gate
                target_unitary = Operator(random_unitary(4))

                # add the target unitary to the ideal circuit
                qc_exact.unitary(target_unitary, [qb_label[n], qb_label[n+1]], label=f'U{qb_label[n], qb_label[n+1]}')

                # add the decomposed target unitary to the decomposed circuit
                qc_decomposed = compile(target_unitary, basis_gate, euler_basis, qc_decomposed, qb_label[n], qb_label[n+1], noise_model=None)
                
                # add the decomposed target unitary to the noisy circuit, with noise model if provided
                qc_noisy = compile(target_unitary, basis_gate, euler_basis, qc_noisy, qb_label[n], qb_label[n+1], noise_model=total_noise_model)

    # return the three circuits: ideal, decomposed, and noisy
    return qc_exact, qc_decomposed, qc_noisy


