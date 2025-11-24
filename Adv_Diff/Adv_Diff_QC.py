import numpy as np
from math import prod
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, MCXGate, RYGate

"""
This module defines functions for constructing Quantum Singular Value Transformation (QSVT)
circuits used to simulate advection and diffusion operators of order 2, 4, 6, and 14.

Functions:
- phase_adder: Build a controlled phase-addition circuit used in block encodings.
- prep: Construct state-preparation gates for a given finite-difference order.
- block_encoding: Build block encodings for the discretized differential operators.
- qsvt: Construct two-angle QSVT circuits for diffusion, advection, or combined cases.
- qsvt_single: Construct a simplified single-angle QSVT circuit (for pure diffusion).
"""


def phase_adder(num_qubits_a: int, num_qubits_b: int, k: float | None = None) -> QuantumCircuit:
    """Construct a phase-adder circuit implementing
    |a>|b> --> xi^{ab}|a>|b> where xi = exp(2/pi i /2^n).

    Args:
        num_qubits_a: Number of qubits in register |a⟩.
        n: Number of qubits in register |b⟩.
        k: Optional additional phase-shift parameter.

    Returns:
        QuantumCircuit implementing the phase addition.
    """

    if num_qubits_a > num_qubits_b:
        num_qubits_a = num_qubits_a % num_qubits_b

    circuit = QuantumCircuit(num_qubits_a + num_qubits_b, name="phase_adder")

    for i in range(num_qubits_b):
        for j in range(min(num_qubits_a, num_qubits_b - i)):
            circuit.cp(2 * np.pi / 2 ** (num_qubits_b - i - j), j, num_qubits_a + i)

    # Optional phase rotation applied if k is given
    if k is not None:
        phase_factor = 2 * np.pi * k
    for i in range(num_qubits_b):
        circuit.p(phase_factor / 2 ** (num_qubits_b - i), i + num_qubits_a)

    return circuit


def prep(order: int) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Return state-preparation gates used in the block encodings.

    Args:
        order: Finite-difference order (supported: 2, 4, 6, 14).

    Returns:
        A tuple of state preparation gates.
    """

    if order == 2:
        qc1 = QuantumCircuit(2)
        qc1.h(1)
        prep_gate_1 = qc1.to_gate()

        qc2 = QuantumCircuit(2)
        qc2.x(1)
        qc2.sdg(1)
        qc2.h(1)
        prep_gate_2 = qc2.to_gate()

        return prep_gate_1, prep_gate_2
          
    elif order == 4:
        theta = 2 * np.arcsin(np.sqrt(2) * 2 / 3)

        qc1 = QuantumCircuit(3)
        qc1.ry(-theta, 0)
        qc1.x(2)
        qc1.s(2)
        qc1.ch(0, 2, ctrl_state="0")
        qc1.x(2)
        qc1.x(1)
        qc1.ch(0, 1, ctrl_state="1")
        qc1.x(1)
        prep_gate_1 = qc1.to_gate()

        qc2 = QuantumCircuit(3)
        qc2.ry(theta, 0)
        qc2.ch(0, 2, ctrl_state="0")
        qc2.ch(0, 1, ctrl_state="1")
        prep_gate_2 = qc2.to_gate()

        return prep_gate_1, prep_gate_2
    
    elif order == 6:
        phi1 = np.arcsin(np.sqrt(9 / 11))
        phi2 = np.arcsin(-3 / np.sqrt(10))

        qc1 = QuantumCircuit(3)
        qc1.x(2)
        qc1.s(2)
        qc1.h(2)
        qc1.ry(2 * phi1, 1)
        qc1.cry(-2 * phi2, 1, 0, ctrl_state="0")
        qc1.ccx(0, 2, 1, ctrl_state="10")
        prep_gate_1 = qc1.to_gate()

        qc2 = QuantumCircuit(3)
        qc2.h(2)
        qc2.ry(2 * phi1, 1)
        qc2.cry(2 * phi2, 1, 0, ctrl_state="0")
        qc2.ccx(0, 2, 1, ctrl_state="10")
        prep_gate_2 = qc2.to_gate()

        return prep_gate_1, prep_gate_2

    else: # order = 14
        p = order // 2
        coeffs = np.zeros(p)

        # Preparing coefficients, ignoring the alternating sign 
        for j in range(1, p + 1):
            coeffs[j - 1] = prod([(p - j + s) / (p + s) for s in range(1, j + 1)]) / j
        coeffs = coeffs / np.sum(coeffs)   # normalizing. Exact value: sum(c) = 363/280
        coeffs = np.sqrt(coeffs)

        # Computing rotation angles 
        phi_vals = np.zeros(p - 1)
        a = 1
        for j in range(p - 1):
            phi_vals[j] = np.arccos(coeffs[j] / a)
            a *= np.sin(phi_vals[j])
            
        for idx in range(2):
            qc = QuantumCircuit(4)

        # Initial phase adjustment
            if idx == 0:
                qc.x(0)
                qc.s(0)
                qc.x(0)
            else:
                phi_vals = -phi_vals
            
            qc.ry(2 * phi_vals[0], 0)
            qc.cry(2 * phi_vals[1], 0, 1)
            qc.cry(-2 * phi_vals[2], 1, 0)
            qc.append(RYGate(2 * phi_vals[3]).control(2, ctrl_state="10"), [0, 1, 2])
            qc.cry(2 * phi_vals[4], 2, 0)
            qc.append(RYGate(-2 * phi_vals[5]).control(2), [0, 2, 1])
            
            # Convert Gray code to regular binary
            qc.cx(1, 0)
            qc.cx(2, 0)
            qc.cx(2, 1)

            # Final ancilla preparation 
            if idx == 0:
                qc.h(3)
            else:
                qc.x(3)
                qc.h(3)
                
            qc.mcx([0, 3], 1, ctrl_state="00")
            qc.mcx([0, 3], 2, ctrl_state="00")
            qc.mcx([0, 1, 3], 2, ctrl_state="001")
            
            if idx == 0:
                prep_gate_1 = qc.to_gate()
            else:
                prep_gate_2 = qc.to_gate()

        return prep_gate_1, prep_gate_2


def block_encoding(num_qubits: int, order: int) -> tuple[QuantumCircuit, int]:
    """Construct the block encoding for a discretized differential operator.

    Args:
        num_qubits: Number of qubits.
        order: Finite-difference order (2, 4, 6, or 14).

    Returns:
        A tuple containing the block-encoding circuit and the number of ancillary qubits used.
    """

    # Determine ancilla count and phase-shift constant
    if order == 2:
        num_ancillas = 2
        phase_shift = -1
        prep_gate_1, prep_gate_2 = prep(2)
    elif order == 14:
        num_ancillas = 4
        phase_shift = -7
        prep_gate_1, prep_gate_2 = prep(14)
    else:
        num_ancillas = 3
        if order == 4:
            phase_shift = -2
            prep_gate_1, prep_gate_2 = prep(4)
        else:
            phase_shift = -3
            prep_gate_1, prep_gate_2 = prep(6)

    block_circuit = QuantumCircuit(num_qubits + num_ancillas)

    block_circuit.append(prep_gate_1, range(num_ancillas))
    block_circuit.append(phase_adder(num_ancillas, num_qubits, phase_shift), range(num_qubits + num_ancillas))
    block_circuit.append(prep_gate_2.inverse(), range(num_ancillas))

    return block_circuit, num_ancillas


def qsvt(num_qubits: int, angle_seq_1: np.ndarray, angle_seq_2: np.ndarray, method: str, order: int) -> QuantumCircuit:
    """Construct a two-angle QSVT circuit.

    Args:
        num_qubits: Number of qubits.
        angle_seq_1: First list of phase angles.
        angle_seq_2: Second list of phase angles (must be one longer).
        method: One of "pure_diff", "pure_adv", "adv_diff".
        order: Finite-difference order. (2, 4, 6, or 14)

    Returns:
        QSVT circuit implementing the two-angle QSVT protocol.
    """

    block_unitary, num_anc0 = block_encoding(num_qubits, order)

    qr_anc = QuantumRegister(2, name = "anc")
    qr0 = QuantumRegister(num_anc0)
    qr1 = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qr_anc,qr0, qr1)

    # Prepare the appropriate linear combinations 
    circuit.h(qr_anc[0])   
    if method != "pure_diff":               
        circuit.s(qr_anc[0])
    circuit.h(qr_anc[1])
    
    circuit.append(QFT(num_qubits), qr1[:])

    cx_gate = MCXGate(num_ctrl_qubits = num_anc0, ctrl_state = num_anc0 * "0")

    if len(angle_seq_1) > len(angle_seq_2):
        angle_seq_1, angle_seq_2 = angle_seq_2, angle_seq_1
    num_phases = len(angle_seq_1)

    # Initial controlled-step
    circuit.append(block_unitary.inverse().control(1), [qr_anc[0]] + qr0[:] + qr1[:])
    circuit.append(cx_gate, qr0[:] + qr_anc[1:])
    circuit.crz(2 * angle_seq_2[num_phases], qr_anc[0], qr_anc[1], ctrl_state = "1")
    circuit.append(cx_gate, qr0[:] + qr_anc[1:])
    
    # Main QSVT loop 
    use_u = True
    for k in range(num_phases - 1, -1, -1):
        if use_u:
            circuit.append(block_unitary, qr0[:] + qr1[:])
        else:
            circuit.append(block_unitary.inverse(), qr0[:] + qr1[:])
        use_u = not use_u

        circuit.append(cx_gate, qr0[:] + qr_anc[1:])
        circuit.crz(2 * angle_seq_1[k], qr_anc[0], qr_anc[1], ctrl_state = "0")
        circuit.crz(2 * angle_seq_2[k], qr_anc[0], qr_anc[1], ctrl_state = "1")
        circuit.append(cx_gate, qr0[:] + qr_anc[1:]) 

    circuit.h(qr_anc[0])
    circuit.h(qr_anc[1]) 
    circuit.append(QFT(num_qubits, inverse = True),qr1[:])

    return circuit


def qsvt_single(num_qubits: int, angle_seq: np.array, order: int) -> QuantumCircuit:
    """
    Construct a simplified single-angle QSVT circuit.

    Args:
        num_qubits: Number of qubits.
        angle_seq: Sequence of phase angles.
        order: Finite-difference order. (2, 4, 6 or 14).

    Returns:
        circuit: QSVT circuit implementing the single-angle QSVT protocol.

    Note:
        Single-angle QSVT is only used for pure diffusion, so a method input is not required. 
    """

    block_unitary, num_anc0 = block_encoding(num_qubits, order)

    qr_anc = QuantumRegister(1, name="anc")
    qr0 = QuantumRegister(num_anc0)
    qr1 = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qr_anc, qr0, qr1)

    circuit.h(qr_anc[0])   
    circuit.append(QFT(num_qubits), qr1[:])

    cx_gate = MCXGate(num_ctrl_qubits=num_anc0, ctrl_state=num_anc0 * "0")
    
    # The QSVT loop 
    num_phases = len(angle_seq)
    use_u = True
    for k in range(num_phases - 1, -1, -1):
        if use_u:
            circuit.append(block_unitary, qr0[:] + qr1[:])
        else:
            circuit.append(block_unitary.inverse(), qr0[:] + qr1[:])
        use_u = not use_u

        circuit.append(cx_gate, qr0[:] + qr_anc[:])
        circuit.rz(2*angle_seq[k], qr_anc[0])
        circuit.append(cx_gate, qr0[:] + qr_anc[:]) 

    circuit.h(qr_anc[0])
    circuit.append(QFT(num_qubits, inverse=True), qr1[:])

    return circuit
