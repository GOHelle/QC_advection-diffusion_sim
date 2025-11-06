import numpy as np
import sys
from Adv_Diff import Simulation_QC
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from Adv_Diff.Angles_QSVT import JA_exp_Angles, JA_2exp_Angles, comb_exp_Angles
from Adv_Diff.Adv_Diff_QC import Block_enc
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.extensions.quantinuum import QuantinuumBackend
from qiskit.circuit.library import QFT, MCXGate

# this file contains the same QSVT function as Adv_Diff_QC.py (but with a decomposition of U_inv.control(1)) and a simplified Sim function that only returns the quantum circuit
# convert_sim function added at the end to convert the circuit to tket and print gate counts

def QSVT(n:int, Phi1:np.array, Phi2:np.array, method:str, order:int) -> QuantumCircuit:
    U, qr0_anc = Block_enc(n, order)

    qr_anc = QuantumRegister(2, name = 'anc')
    qr0 = QuantumRegister(qr0_anc)
    qr1 = QuantumRegister(n)
    circ = QuantumCircuit(qr_anc,qr0, qr1)

    # Preparing the appropriate linear combinations 
    circ.h(qr_anc[0])   
    if method != "pure_diff":               
        circ.s(qr_anc[0])
    circ.h(qr_anc[1])
    
    circ.append(QFT(n),qr1[:])

    CX_gate = MCXGate(num_ctrl_qubits = qr0_anc, ctrl_state = qr0_anc*'0')

    if len(Phi1) > len(Phi2):
        Phi3 = Phi2
        Phi2 = Phi1
        Phi1 = Phi3
    l = len(Phi1)
    
    # Setting up the controlled circuit first 
    U_inv = U.inverse()
    cU_inv = U_inv.control(1).decompose()   # decompose to avoid issues with tket compilation
    circ.append(cU_inv,[qr_anc[0]]+qr0[:]+qr1[:])
    circ.append(CX_gate,qr0[:]+qr_anc[1:])
    circ.crz(2*Phi2[l],qr_anc[0],qr_anc[1],ctrl_state = '1')
    circ.append(CX_gate,qr0[:]+qr_anc[1:])
    
    # The QSVT loop 
    s = 1
    for k in range(l-1,-1,-1):
        if s == 1:
            circ.append(U, qr0[:] + qr1[:])
            s = 0
        else:
            circ.append(U.inverse(),qr0[:] + qr1[:])
            s = 1

        circ.append(CX_gate,qr0[:] + qr_anc[1:])
        circ.crz(2*Phi1[k], qr_anc[0], qr_anc[1], ctrl_state = '0')
        circ.crz(2*Phi2[k], qr_anc[0], qr_anc[1], ctrl_state = '1')
        circ.append(CX_gate,qr0[:] + qr_anc[1:]) 

    circ.h(qr_anc[0])
    circ.h(qr_anc[1]) 
    
    circ.append(QFT(n,inverse = True),qr1[:])

    return circ

def QSVT_single(n:int, Phi:np.array, order:int) -> QuantumCircuit:

    U, qr0_anc = Block_enc(n, order)

    qr_anc = QuantumRegister(1, name = 'anc')
    qr0 = QuantumRegister(qr0_anc)
    qr1 = QuantumRegister(n)
    circ = QuantumCircuit(qr_anc,qr0, qr1)

    circ.h(qr_anc[0])   
    circ.append(QFT(n),qr1[:])

    CX_gate = MCXGate(num_ctrl_qubits = qr0_anc, ctrl_state = qr0_anc*'0')
    
    # The QSVT loop 
    l = len(Phi)
    s = 1
    for k in range(l-1,-1,-1):
        if s == 1:
            circ.append(U, qr0[:] + qr1[:])
            s = 0
        else:
            circ.append(U.inverse(),qr0[:] + qr1[:])
            s = 1

        circ.append(CX_gate,qr0[:] + qr_anc[:])
        circ.rz(2*Phi[k], qr_anc[0])
        circ.append(CX_gate,qr0[:] + qr_anc[:]) 

    circ.h(qr_anc[0])
    circ.append(QFT(n,inverse = True),qr1[:])

    return circ

def Sim(n: int, T: float, c: float, nu: float, d:float=4, init_f = lambda x: np.exp(-10*(x-4/3)**2), order:int = 2, eps=10**(-8)):
    
    # exceptions and input formatting
    if c == 0 and nu == 0: sys.exit("Error: c and nu cannot both be 0")
    if order not in [2,4,6,14]: sys.exit("Error: The order should be either 2, 4, 6 or 14")

    # Identify which PDE evolution type applies
    method = "pure_diff" if c==0 else "pure_adv" if nu==0 else "adv_diff"

    # Choose number of ancilla qubits based on method and order
    anc = 4 if order ==2 else 6 if order == 14 else 5 
    if method =="pure_diff": anc = anc-1

    # Computing time-evolution parameter M
    dx = d/(2**n)
    dt_factors = {2: 1, 4: 3/2, 6: 11/6, 14: 363/140}
    M_adv = c * T * dt_factors[order] / dx
    M_diff = nu * T * dt_factors[order]**2 / (dx**2)

    # scaling
    adv_scale = 0.95
    diff_scale = 0.95

    # Spatial grid
    x = np.linspace(0,d,2**n,endpoint=False)
    y = init_f(x)
    if not np.all(y >= 0): sys.exit("Error: initial function not positive")
    norm_y = np.linalg.norm(y)
    y /= norm_y

    qr_anc, cr_anc = QuantumRegister(anc), ClassicalRegister(anc)
    qr, cr = QuantumRegister(n), ClassicalRegister(n)
    qc = QuantumCircuit(qr_anc, qr, cr_anc, cr)

    # State preparation 
    qc.prepare_state(Statevector(y),qr)

    # Generate phase angles for QSVT evolution
    if method == "pure_adv": # We only have to apply the advection QSVT
        Phi_cos, Phi_sin = JA_exp_Angles(M_adv, adv_scale, eps)
        qc.append(QSVT(n, Phi_cos, Phi_sin, method, order), qr_anc[:] + qr[:])
    elif method == "pure_diff":  # We only have to apply the diffusion QSVT
        Phi_even = JA_2exp_Angles(M_diff, diff_scale, eps)
        qc.append(QSVT_single(n, Phi_even, order), qr_anc[:] + qr[:])
    else:
        Phi_even, Phi_odd = comb_exp_Angles(eps, M_diff, M_adv)
        qc.append(QSVT(n, Phi_odd, Phi_even, method, order), qr_anc[:] + qr[:])

    # Measurement simulation
    qc.measure(qr_anc,cr_anc)
    qc.measure(qr,cr)
    
    return qc

def convert_sim(n=4, T=0.5, c=1, nu=0.01, d=4, init_f = lambda x: np.exp(-10*(x-4/3)**2), order = 6, eps=10**(-8)):
    
    circ = Sim(n, T, c, nu, d, init_f, order, eps)
    # print gate counts for compiled circuit for quantum backend
    tk_circ = qiskit_to_tk(circ)
    backend = QuantinuumBackend("H1-1", machine_debug=True) # machine_debug=True uses local simulator instead of accessing real backend
    compiled_circ = backend.get_compiled_circuit(tk_circ) # compiles and optimizes to the backend
    num_1q = compiled_circ.n_1qb_gates()
    num_2q = compiled_circ.n_2qb_gates()
    cost = backend.cost(compiled_circ, n_shots=100) # cost in HQC's

    print(f"Single-qubit gates: {num_1q}")
    print(f"Two-qubit gates: {num_2q}")
    print(f"Cost: {cost}") # this is 0 when we use the simulator

# Advection example
Simulation_QC.Sim(5, T = 0.5, c = 1, nu = 0, init_f = lambda x: np.exp(-10*(x-4/3)**2), order = 4, sim_type="meas")   # for visualizing the results with these parameters
convert_sim(5, T = 0.5, c = 1, nu = 0, init_f = lambda x: np.exp(-10*(x-4/3)**2), order = 4)

# Diffusion example
Simulation_QC.Sim(6, T = 0.5, c = 0, nu = 0.01, init_f = lambda x: np.exp(-10*(x-4/3)**2), order = 6)   # for visualizing the results with these parameters
convert_sim(6, T = 0.5, c = 0, nu = 0.01, init_f = lambda x: np.exp(-10*(x-4/3)**2), order = 6)