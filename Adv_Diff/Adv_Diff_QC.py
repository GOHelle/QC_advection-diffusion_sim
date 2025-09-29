import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, MCXGate
from qiskit.circuit.library.standard_gates import GlobalPhaseGate

""" 
This module defines functions to construct Quantum Singular Value Transformation (QSVT) circuits 
for simulating advection and diffusion of order 2, 4, and 6.

It includes methods Phase_adder, Prep, Block_enc, QSVT, and QSVT_single.

Phase_adder construct a phase adder circuit used in the block encodings. 

Prep prepares the appropriate state preparation gates used in the block encodings.

Block_enc returns the appropriate block encoding, given a choice of method order, and the number of ancillary qubits 
          needed. The block encoding is used for constructing the QSVT circuit

QSVT and QSVT_single returns the appropriate QSVT circuit depending on whether pure diffusion, pure advection or 
                     advection-diffusion is simulated. QSVT constructs a full two-angle QSVT circuit, while QSVT_single 
                     constructs a simplified single-angle QSVT circuit
"""  

def Phase_adder(m:int, n:int, k:float=None) -> QuantumCircuit:
    """Constructs a phase adder circuit implementing: |a>|b> --> xi^{ab}|a>|b> for xi = exp(2/pi i /2^n).

    Args:
        m: Number of qubits in register |a>.
        n: Number of qubits in register |b>.
        k: Optional phase shift parameter.

    Returns:
        circ: Quantum circuit implementing the phase addition.
    """

    if m>n:
        m = m%n 

    circ = QuantumCircuit(m+n,name = 'Phase_adder')
    for i in range(n):
        for j in range(min(m,n-i)):
            circ.cp(2*np.pi/2**(n-i-j),j,m+i)
    
    # Optional phase rotation applied if k is given
    if k:
        a = 2*np.pi*k
        for i in range(n):
            circ.p(a/2**(n-i),i+m)

    return circ

def Prep(order:int) -> tuple[QuantumCircuit, QuantumCircuit]:
    """
    Returns a pair of state preparation gates used in block encodings for a given order.

    Args:
        order: Order of the method. Supported values are 2, 4, or 6.

    Returns:
        (g1, g2): A tuple of state preparation gates.
    """

    if order == 2:
        qc = QuantumCircuit(2)
        qc.append(GlobalPhaseGate(np.pi/2),0)
        qc.h(1)
        g1 = qc.to_gate()
        qc = QuantumCircuit(2)
        qc.x(1)
        qc.h(1)
        g2 = qc.to_gate()
        return g1, g2 
    
    elif order == 4:
        theta = 2*np.arcsin(np.sqrt(2)*2/3)
        
        qc = QuantumCircuit(3)
        qc.append(GlobalPhaseGate(np.pi/2),0)
        qc.ry(-theta,0)
        qc.x(2)
        qc.ch(0,2,ctrl_state = '0')
        qc.x(2)
        qc.x(1)
        qc.ch(0,1,ctrl_state = '1')
        qc.x(1)
        g1 = qc.to_gate()

        qc = QuantumCircuit(3)
        qc.ry(theta,0)
        qc.ch(0,2,ctrl_state = '0')
        qc.ch(0,1,ctrl_state = '1')
        g2 = qc.to_gate()

        return g1, g2
    
    else:
        phi_1 = np.arcsin(np.sqrt(9/11)) 
        phi_2 = np.arcsin(-3 / np.sqrt(10)) 
        
        qc = QuantumCircuit(3)
        qc.x(2)
        qc.s(2)
        qc.h(2)
        qc.ry(2*phi_1,1)
        qc.cry(-2*phi_2,1,0,ctrl_state = '0')
        qc.ccx(0,2,1,ctrl_state = '10')
        g1 = qc.to_gate()
        
        qc = QuantumCircuit(3)
        qc.h(2)
        qc.ry(2*phi_1,1)
        qc.cry(2*phi_2,1,0,ctrl_state = '0')
        qc.ccx(0,2,1,ctrl_state = '10')
        g2 = qc.to_gate()
        
        return g1, g2

def Block_enc(n:int, order:int) -> tuple[QuantumCircuit, int]:
    """
    Constructs the block encoding of a discretized differential operator for a given method order.

    Parameters:
        n: Number of qubits.
        order: Order of the method. Supported values are 2, 4, or 6.

    Returns:
        qc: The block-encoding circuit
        anc: the number of ancillary qubits used.
    """

    if order == 2:
        anc = 2  # number of ancillary qubits
        k = -1  # argument for Phase_adder
        g1,g2 = Prep(2)
    else: 
        anc = 3
        if order == 4:
            k = -2
            g1,g2 = Prep(4)
        else:
            k = -3
            g1,g2 = Prep(6)

    qc = QuantumCircuit(n+anc)
    qc.append(g1,range(anc))
    qc.append(Phase_adder(anc,n,k), range(n+anc))
    qc.append(g2.inverse(), range(anc))

    return qc, anc

def QSVT(n:int, Phi1:np.array, Phi2:np.array, method:str, order:int) -> QuantumCircuit:
    """ Constructs a two-angle QSVT circuit using a chosen method and order.

    Parameters:
        n: Number of qubits.
        Phi1: Sequence of phase angles for QSVT.
        Phi2: Sequence of phase angles for QSVT. Must be one element longer than Phi1.
        method: Supported values:
                  - "pure_diff": pure diffusion
                  - "pure_adv": pure advection
                  - "adv_diff": combined advection-duffusion
        order: Order of the method. Supported values are 2, 4, or 6.

    Returns:
        circ: QSVT circuit implementing the transformation.
    """

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
    circ.append(U.inverse().control(1),[qr_anc[0]]+qr0[:]+qr1[:])
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
    """
    Constructs a simplified single-angle QSVT circuit.

    Args:
        n: Number of qubits.
        Phi1: Sequence of phase angles for QSVT.
        order: Order of the method. Supported values are 2, 4, or 6.

    Returns:
        circ: QSVT circuit implementing the transformation.

    Note:
        Single-angle QSVT is only used for pure diffusion, so a method input is not required. 
    """

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
