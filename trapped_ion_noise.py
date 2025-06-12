# Here we provide code to simulate the noise in a trapped ion system.

# imports from standard libraries
import numpy as np
from numpy.polynomial import Polynomial as P

# imports from Qiskit
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error, reset_error, thermal_relaxation_error

def trapped_ion_noise_model(number_of_qubits, circuit_depth): 
    """
    Creates a noise model for a trapped ion quantum computer.
    
    Args:
        number_of_qubits (int): Number of qubits in the quantum computer.
        circuit_depth (int): Depth of the quantum circuit.
        
    Returns:
        NoiseModel: A noise model for the trapped ion quantum computer.
    """
    # The data below came from a trapped ion quantum computer, and is used to model the noise in the quantum computer.

    #single qubit gate data
    x = [0.00000927357382,0.00001260986452,0.00002928106781,0.00003274528310,0.00004924902626,0.00006055222635,0.00011460445423,0.00016071899407,0.00022938562612,0.00027801912678,0.00030185445510,0.00043266624068,0.00044177729723,0.00057279002193,0.00057682838342,0.00070890425941,0.00077947268436,0.00079515532825,0.00080682762917,0.00085089608318,0.00100219273868,0.00104089382283,0.00109095947198,0.00104372058209,0.00116174830392,0.00123813079998,0.00129937860462,0.00128349409828,0.00130493212361,0.00139111136819,0.00145414403424,0.00154043057681,0.00153916316729,0.00176066840966,0.00213276093073,0.00245376808135,0.00265083092714,0.00265166760575,0.00324197197114,0.00326583722022,0.00320380389140,0.00328885321692,0.00348062433166,0.00347240388496,0.00323130829847,0.00344292847993,0.00356279994437,0.00383968605530,0.00396580149891,0.00433704004305,0.00426199843599,0.00398851122681,0.00420891018456,0.00397290379810,0.00430589372375,0.00449442198642,0.00528310345302,0.00592808713104,0.00606595182665,0.00596611750736,0.00626052406424,0.00596401086915,0.00559063279007,0.00570511992288,0.00575084539024,0.00610864910472,0.00612002018331,0.00711666299345,0.00719744952783,0.00676697535723,0.00683955664349,0.00737435778289,0.00770993945309,0.00738052359704,0.00789639090326,0.00879793044710,0.00920989799069,0.00917868385183,0.00939672082347,0.00923738673725,0.00861016938581,0.00834737700795,0.00786299725598,0.00813393990978,0.00851132102235,0.00961866303807,0.01016420541433,0.01134483981584,0.01215935006530,0.01228410946338,0.01239877667888,0.01247468051176,0.01314578489714,0.01271362217630,0.01259739901016,0.01293545345938,0.01223001241374]
    y = [0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.029,0.03,0.031,0.032,0.033,0.034,0.035,0.036,0.037,0.038,0.039,0.04,0.041,0.042,0.043,0.044,0.045,0.046,0.047,0.048,0.049,0.05,0.051,0.052,0.053,0.054,0.055,0.056,0.057,0.058,0.059,0.06,0.061,0.062,0.063,0.064,0.065,0.066,0.067,0.068,0.069,0.07,0.071,0.072,0.073,0.074,0.075,0.076,0.077,0.078,0.079,0.08,0.081,0.082,0.083,0.084,0.085,0.086,0.087,0.088,0.089,0.09,0.091,0.092,0.093,0.094,0.095,0.096]

    polynomial_single = P.fit(x, y, 10) # 10th degree polynomial

    #two qubit gate data
    x = [2.74212E-05,6.74248E-05,0.000152449,0.000257546,0.000388857,0.000545713,0.000941022,0.000879523,0.001199727,0.001450736,0.00191575,0.001881703,0.002287163,0.002890946,0.003120413,0.003645147,0.003690388,0.004128885,0.004342591,0.004629127,0.005132646,0.006086767,0.00615817,0.006646223,0.007228772,0.006925123,0.008122698,0.008420623,0.008606475,0.008097651,0.009887001,0.010068802,0.010989772,0.011232219,0.010628959,0.012174436,0.011912745,0.012518089,0.014012194,0.013461999,0.014363502,0.015807069,0.014238445,0.016843228,0.016473268,0.016262249,0.017594919,0.018297332,0.018997607,0.018163527,0.019728193,0.021631942,0.020939053]
    y = [0,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.0011,0.0012,0.0013,0.0014,0.0015,0.0016,0.0017,0.0018,0.0019,0.002,0.0021,0.0022,0.0023,0.0024,0.0025,0.0026,0.0027,0.0028,0.0029,0.003,0.0031,0.0032,0.0033,0.0034,0.0035,0.0036,0.0037,0.0038,0.0039,0.004,0.0041,0.0042,0.0043,0.0044,0.0045,0.0046,0.0047,0.0048,0.0049,0.005,0.0051,0.0052]

    polynomial_double = P.fit(x, y, 10) # 10th degree polynomial

    #---------------------------SET ERROR PARAMETERS---------------------------#
    # two qubit gate fidelity
    two_qubit_gate_fid = 99.99 #Valid range: 98-99.99%

    # single qubit gate fidelity
    single_qubit_gate_fid = 99.999 #Valid range: 99-99.999

    #Coherence times [ns]
    t1 = 2e12
    t2 = 2e9

    # shuttle time [ns]
    time_shuttle = 5e6

    #---------------------------CREATE NOISE MODEL---------------------------#
    total_noise_model = NoiseModel(['rz', 'rx', 'ry', 'rxx'])

    # #---------------------------THERMAL NOISE---------------------------#

    # randomly sample T1 and T2 coherence times
    T1s = np.random.normal(t1, t1*0.2, number_of_qubits)
    T2s = np.random.normal(t2, t2*0.2, number_of_qubits)

    # Truncate to ensure T2s <= 2 * T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(number_of_qubits)])

    #create decoherence errors
    errors_shuttle  = [thermal_relaxation_error(t1, t2, time_shuttle) for t1, t2 in zip(T1s, T2s)]

    # add decoherence during shuttling  (rz(0) gate in between each layer of gates)
    for j in range(number_of_qubits): # for each qubit
        total_noise_model.add_quantum_error(errors_shuttle[j], "rz", [j]) # add the error

    #---------------------------OPERATION NOISE---------------------------#

    #operation based errors: probability of implementing the wrong gate on individual or two qubits.
    p_error_single = polynomial_single(1-single_qubit_gate_fid/100)
    p_error_double = polynomial_double(1-two_qubit_gate_fid/100)

    op_error_single = pauli_error([('X', p_error_single), ('Y', p_error_single), ('Z', p_error_single), ('I', 1 - 3*p_error_single)]) #equal chance of each pauli gate
    op_error_double = pauli_error([('X', p_error_double), ('Y', p_error_double), ('Z', p_error_double), ('I', 1 - 3*p_error_double)])
    op_error2 = op_error_double.tensor(op_error_double) #tensor the error operation with itself to be used in 2qb gate

    for j in np.arange(number_of_qubits):
        j = int(j) #convert to int for noise model
        #single qubit operation errors
        total_noise_model.add_quantum_error(op_error_single, "rx", [j])
        total_noise_model.add_quantum_error(op_error_single, "ry", [j])
        
        for k in np.arange(number_of_qubits):
            k=int(k) # convert to int for noise model
            #two qubit errors
            if j != k: 
                #only choose two qubit errors for different qubits
                total_noise_model.add_quantum_error(op_error2, "rxx", [j, k])

    # TODO : add reset errors, add measurement errors

    return total_noise_model
