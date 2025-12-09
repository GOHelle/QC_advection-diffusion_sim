import numpy as np
import sys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from typing import Callable
from Adv_Diff import Adv_Diff_QC
from Adv_Diff.Angles_QSVT import jacobi_anger_exp_angles, jacobi_anger_squared_exp_angles, combined_exp_angles
from Adv_Diff.Fourier import fourier_coefficients_2d, fourier_approximation_2d


"""
This module provides a quantum simulation method to approximate the solution of the 2D advection-diffusion PDE.

Method 'simulate_adv_diff_2d' utilizes QSVT methods defined in 'Adv_Diff_QC.py' to simulate the evolution of an initial function over a 2D 
spatial domain under the advection-diffusion PDE. It compares the quantum results (statevector and/or measurement) with a classical Fourier approximation, 
optionally prints circuit complexity, and plots results.
"""

def simulate_adv_diff_2d(
    num_qubits: int, 
    time: float, 
    adv_speed_x: float, 
    adv_speed_y: float, 
    diff_coeff: float, 
    domain_length: float = 4.0, 
    init_f: Callable = lambda X, Y: np.sin(np.pi * (0.5 * X + Y)) ** 2, 
    shots: int = 10 ** 7, 
    report_complexity: bool = True, 
    order: int = 2, 
    tolerance: float = 1e-6, 
    sim_type: str = "both", 
    compute_exact: bool = True, 
    plot: bool = True
    ):

    """ Quantum simulation of the 2D advection-diffusion equation via QSVT with classical comparison..

    This function constructs and runs quantum circuits to simulate the evolution of a 2D initial function 
    under the advection-diffusion PDE by applying 1D advection-diffusion in both x and y directions.
    The simulation therefore selects the required number of ancilla qubits based on the simulation type (in each direction) and QSVT order. 
    Postselection on ancillary measurements is performed to extract the quantum output.
    The output is compared to a classical Fourier-based approximation, and results are plotted for visual inspection. 
    relevant results, such as success rate, max error and circuit complexity, are printed and summarized in a table.

    Args:
        num_qubits: Number of qubits per spatial dimension (total grid size is 2^num_qubits x 2^num_qubits).
        time: Final time for the evolution.
        adv_speed_x: Advection velocity in the x direction.
        adv_speed_y: Advection velocity in the y direction.
        diff_coeff: Diffusion coefficient.
        domain_length: Length of each spatial dimension (domain is [0, domain_length] x [0, domain_length]).
        init_f: Initial condition function f(x,y).
        shots: Number of measurement shots for the quantum simulation (if measurement simulation requested).
        report_complexity: If True, prints gate counts and depth after transpiling.
        order: Method order (2, 4, 6 or 14).
        tolerance: Tolerance parameter used for angle calculations.
        sim_type: One of "sv" (statevector simulation only), "meas" (measurement only), or "both".
        compute_exact: If True, compute the classical Fourier solution for comparison.
        plot: If True, plots the initial condition and the solutions

    Outputs:
        - 3D surface plots of the initial condition, fourier solution, and quantum solution at final time if plot = True. 
        - Gate counts and circuit depth if report_complexity=True.
        - Success rate of postselection
        - Max errors between quantum and fourier solutions if compute_exact = True.

    Returns:
    A tuple containing:
        - Spatial grid points in x direction
        - Spatial grid points in y direction
        - Inital condition over space discretization
        - Quantum measurement-based solution. None if sim_type = "sv".
        - Fourier solution. None if compute_exact = False
        - Quantum statevector solution. None if sim_type = "meas"
        - List of maximum errors from measurement and/or statevector solutions. Only non-empty if compute_exact = True
        - Success rate of postselection.
        - List of complexity data containing number of 1-qubit gates, number of cnot gates, total number of gates, 
          and circuit depth. None if report_complexity = False
    """

    # exceptions
    if (diff_coeff == 0 and adv_speed_x == 0) or (diff_coeff == 0 and adv_speed_y == 0):
        sys.exit("Error: if diff_coeff=0, adv_speed_x or adv_speed_y should be nonzero, to avoid 1d evolution")
    if order not in [2, 4, 6, 14]: sys.exit("Error: The order should be either 2, 4, 6 or 14")
    if sim_type not in ["sv", "meas", "both"]: sys.exit("Error: sim_type should be either sv', 'meas' or 'both'")

    # Determine PDE evolution type
    method_x = "pure_adv" if diff_coeff == 0 else "pure_diff" if adv_speed_x == 0 else "adv_diff"
    method_y = "pure_adv" if diff_coeff == 0 else "pure_diff" if adv_speed_y == 0 else "adv_diff"

    # Choose number of ancilla qubits based on method and order
    num_anc1 = 4 if order == 2 else 6 if order == 14 else 5 
    num_anc2 = 4 if order == 2 else 6 if order == 14 else 5
    if method_x =="pure_diff": num_anc1 -= 1
    if method_y =="pure_diff": num_anc2 -= 1
    num_anc = max(num_anc1, num_anc2) + 1
    print(f"\n-- QUBITS --")
    num_spatial = 2 * num_qubits
    num_qubits_total = num_anc + num_spatial
    print("Ancillas = ", num_anc, "Spatial = ", num_spatial, "Total = ", num_qubits_total)

    # Spatial discretization and computing time-evolution parameter M
    num_points = 2**num_qubits  
    dx = domain_length / num_points
    dt_factors = {2: 1, 4: 3/2, 6: 11/6, 14: 363/140}
    M_adv_x = adv_speed_x * time * dt_factors[order] / dx
    M_adv_y = adv_speed_y * time * dt_factors[order] / dx
    M_diff = diff_coeff * time * dt_factors[order] ** 2 / (dx ** 2)

    # scaling factors
    adv_scale = 0.95
    diff_scale = 0.95

    # Construct registers and circuit
    qr_anc, cr_anc = QuantumRegister(num_anc), ClassicalRegister(num_anc)
    qr, cr = QuantumRegister(2 * num_qubits), ClassicalRegister(2 * num_qubits)
    qc = QuantumCircuit(qr_anc, qr, cr_anc, cr)

    # Spatial grid and normalized initial state preparation
    x = np.linspace(0, domain_length, num_points, endpoint=False)
    y = np.linspace(0, domain_length, num_points, endpoint=False)
    X, Y = np.meshgrid(x, y)
    init_values = init_f(X, Y) 
    if not np.all(init_values >= 0):
        sys.exit("Error: initial function must be non-negative on the domain")   
    norm_init = np.linalg.norm(init_values)       
    init_normalized = init_values / norm_init
    qc.prepare_state(Statevector(init_normalized.flatten()), qr) 

    # Generate and append QSVT angle sequences for QSVT evolution
    print(f"-- ANGLE SEQUENCES --")
    def build_qsvt(method, M_adv):
        """ Build QSVT circuit to apply separately to each direction."""
        if method == "pure_adv":
            angle_seq_cos, angle_seq_sin = jacobi_anger_exp_angles(M_adv, adv_scale, tolerance)
            return Adv_Diff_QC.qsvt(num_qubits, angle_seq_cos, angle_seq_sin, method, order)
        elif method == "pure_diff":
            angle_seq_even = jacobi_anger_squared_exp_angles(M_diff, diff_scale, tolerance)
            return Adv_Diff_QC.qsvt_single(num_qubits, angle_seq_even, order)
        else:
            angle_seq_even, angle_seq_odd = combined_exp_angles(tolerance, M_diff, M_adv)
            return Adv_Diff_QC.qsvt(num_qubits, angle_seq_odd, angle_seq_even, method, order)

    qsvt_x = build_qsvt(method_x, M_adv_x)  # QSVT evolution in x direction
    qc.append(qsvt_x, qr_anc[:num_anc - 1] + qr[:num_qubits])

    qc.mcx(qr_anc[:num_anc - 1], qr_anc[num_anc - 1], ctrl_state = (num_anc - 1) * '0')   # Composition trick 
    qc.x(qr_anc[num_anc - 1])

    qsvt_y = build_qsvt(method_y, M_adv_y)  # QSVT evolution in y direction
    qc.append(qsvt_y, qr_anc[:num_anc - 1] + qr[num_qubits:])

    # Fourier approximation
    fourier_result = None
    if compute_exact:
        if diff_coeff != 0:
            fourier_func = fourier_approximation_2d(*fourier_coefficients_2d(init_f, 1e-8, domain_length), domain_length)
            fourier_result = fourier_func(X, Y, time, diff_coeff, adv_speed_x, adv_speed_y)
        else:  # pure advection
            fourier_result = init_f((X - adv_speed_x * time) % domain_length, (Y - adv_speed_y * time) % domain_length)

    # Statevector simulation
    statevec_result = None
    if sim_type != "meas":
        # using Aer statevector simulator
        simulator = AerSimulator(method='statevector')
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        qc_sv = transpile(qc_sv, simulator)
        result = simulator.run(qc_sv).result()
        sv_data = result.get_statevector(qc_sv)
        statevec_result = np.asarray(sv_data).reshape(num_points, num_points, 2 ** num_anc)[:, :, 0]

        if sim_type != "both":
            success_rate = np.linalg.norm(statevec_result) ** 2
        # Rescale according to method
        statevec_result *= 2**2 * norm_init
        statevec_result /= adv_scale if method_x == "pure_adv" else (2 * diff_scale if method_x == "pure_diff" else 1) 
        statevec_result /= adv_scale if method_y == "pure_adv" else (2 * diff_scale if method_y == "pure_diff" else 1)

    # Measurement simulation
    meas_result  = None
    complexity = None
    if sim_type != "sv":
        qc.measure(qr_anc,cr_anc)
        qc.measure(qr,cr)
        sim = AerSimulator()
        qc_comp = transpile(qc,sim)
        res = sim.run(qc_comp,shots = shots).result()
        counts = res.get_counts(0)

        # Postselection
        total_selected = 0
        meas_result = np.zeros([num_points, num_points])
        select_str = "0" * num_anc

        for key in counts:
            parts = key.split()
            if parts[1] == select_str:
                i = int(parts[0][:num_qubits], 2)
                j = int(parts[0][num_qubits:], 2)
                meas_result[i, j] = np.sqrt(counts[key] / shots) * norm_init * 2 ** 2
                total_selected += counts[key]
        # Reverse scaling factors
        meas_result /= adv_scale if method_x == "pure_adv" else (2 * diff_scale if method_x == "pure_diff" else 1)
        meas_result /= adv_scale if method_y == "pure_adv" else (2 * diff_scale if method_y == "pure_diff" else 1)

        success_rate = total_selected / shots
        print(f"\n-- SUCCESS RATE -- \n succes rate of postselection: {success_rate:.4f} ")

    # Printing gate counts and circuit depth
    if report_complexity:
        tqc = transpile(qc, basis_gates=["u", "cx"])    # Generic 1Q gate and CNOT
        gts = tqc.count_ops()
        gate_1q = gts['u']
        gate_2q = gts['cx']                  
        print(f"\n-- COMPLEXITY-- \nTotal: {gate_1q + gate_2q}\nCircuit depth after transpiling:{tqc.depth()}\n")
        complexity = [gate_1q, gate_2q, gate_1q + gate_2q, tqc.depth()]

    # Max error
    max_err = []
    if compute_exact:
        print(f"-- MAX ERROR --")
        if sim_type != "sv": 
            max_err_meas = np.max(np.abs(meas_result - fourier_result))
            max_err.append(max_err_meas)
            print(f"Max error from measurement: {max_err_meas:.3e}")
        if sim_type != "meas": 
            max_err_sv = np.max(np.abs(statevec_result - fourier_result))
            max_err.append(max_err_sv)
            print(f"Max error from statevector: {max_err_sv:.3e}")

    # Plotting
    if plot:
        fig = plt.figure(figsize=(21, 6))
        z_min, z_max = np.min(init_values), np.max(init_values)
        data = [init_values, fourier_result] if compute_exact else [init_values]
        titles = ["Initial Condition", rf"Exact Solution at Time $T = {time}$"] if compute_exact else ["Initial Condition"]

        if sim_type != "meas": 
            data.append(statevec_result.real)
            titles.append(rf"Quantum Statevector at Time $T = {time}$")
        if sim_type != "sv":
            data.append(meas_result)
            titles.append(rf"Quantum Measurements at Time $T = {time}$")

        for i, (title, Z) in enumerate(zip(titles, data), start=1):
            ax = fig.add_subplot(1, len(data), i, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlim(z_min - 0.05, z_max + 0.05)

        if diff_coeff == 0: 
            title_str = f"2D Advection Simulation of Order {order} with {num_qubits} Spatial Qubits per Dimension\n" \
                        rf"Advection Speeds are $c_x = {adv_speed_x:.3g}, \; c_y = {adv_speed_y:.3g}$"
        elif adv_speed_x == 0 and adv_speed_y == 0: 
            title_str = f"2D Diffusion Simulation of Order {order} with {num_qubits} Spatial Qubits per Dimension\n" \
                        rf"Diffusion Coefficient is $\nu = {diff_coeff:.3g}$"
        else: 
            title_str = f"2D Advection-Diffusion Simulation of Order {order} with {num_qubits} Spatial Qubits per Dimension\n" \
                        rf"Diffusion Coefficient is $\nu = {diff_coeff:.3g}$, and Advection Speeds are $c_x = {adv_speed_x:.3g}, \; c_y = {adv_speed_y:.3g}$"
        fig.suptitle(title_str, fontsize=16)
        plt.tight_layout()
        plt.show()

    return x, y, init_values, meas_result, fourier_result, statevec_result, num_qubits_total, max_err, success_rate, complexity
