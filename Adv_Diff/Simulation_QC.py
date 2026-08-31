import numpy as np
import sys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from tabulate import tabulate
from typing import Callable
from Adv_Diff import Adv_Diff_QC
from Adv_Diff.Angles_QSVT import jacobi_anger_exp_angles, jacobi_anger_squared_exp_angles, combined_exp_angles
from Adv_Diff.Fourier import fourier_coefficients, fourier_approximation

"""
This module provides a quantum simulation method to approximate the solution of the advection-diffusion PDE.

Method 'simulate_adv_diff' utilizes QSVT methods defined in 'Adv_Diff_QC.py' to simulate the evolution of an initial condition 
under the advection-diffusion PDE. It compares the quantum results (statevector and/or measurement)  with a classical Fourier
approximation, optionally prints circuit complexity, and plots results.
"""

def simulate_adv_diff(
    num_qubits: int, 
    times: np.array, 
    adv_speed: float, 
    diff_coeff: float, 
    domain_length: float = 4.0, 
    init_f: Callable = lambda x: np.exp(-10 * (x - 4 / 3) ** 2), 
    shots: int = 10 ** 6, 
    report_complexity: bool = True, 
    order: int = 2, 
    tolerance: float = 1e-8, 
    sim_type: str = "both", 
    compute_exact: bool = True, 
    plot: bool = True
    ):

    """ Quantum simulation of the advection-diffusion equation via QSVT, with classical comparison.

    This function supports pure diffusion, pure advection, and combined advection-diffusion evolution, and method orders 2, 4, 6 and 14.
    The simulation therefore selects the required number of ancilla qubits based on the simulation type and QSVT order. 
    Postselection on ancillary measurements is performed to extract the quantum output.
    The output is compared to a classical Fourier-based approximation, and results are plotted for visual inspection. 
    relevant results, such as success rate, max error and circuit complexity, are printed and summarized in a table.

    Args:
        num_qubits: Number of qubits to discretize the spatial domain (2^num_qubits grid points).
        times: Array (or scalar) of final times at which the solution is evaluated. Each entry in times triggers a separate simulation run. 
        adv_speed: Advection speed parameter.
        diff_coeff: Diffusion coefficient.
        domain_length: Length of the spatial domain.
        init_f: Initial condition function f(x). Default is a Gaussian centered at 4/3.
        shots: Number of measurement shots (if measurement simulation requested).
        report_complexity: If True, prints gate counts and depth after transpiling.
        order: Method order (2, 4, 6 or 14).
        tolerance: Tolerance parameter used for angle calculations.
        sim_type: One of "sv" (statevector simulation only), "meas" (measurement only), or "both".
        compute_exact: If True, compute the classical Fourier solution for comparison.
        plot: If True, plots the initial condition and the solutions for each value in times.

    Returns:
        A tuple containing: 
        - The space discretization
        - The initial function values on the discretized space
        - A list of quantum measurement-based solutions for each value in 'times'. Only non-empty if sim_type != "sv".
        - A list of Fourier solutions for each value in 'times'. Only non-empty if compute_exact = True. 
        - A list of quantum statevector solutions for each value in 'times'. Only non-empty if sim_type != "meas".
        - A list of max-error entries for each value in 'times'. Each entry contains max errors from measurement and/or statevector simulation.
          Only non-empty if exact_sol = True.
        - A list of postselection success rates for each value in 'times'.
        - A list of complexity entries for each in value in 'times'. Each entry contains the number of 1-qubit gates, number of cnot-gates, 
          total number of gates, and circuit depth. Only non-empty if Complexity=True.
    """

    # exceptions and input formatting
    times = np.atleast_1d(times)  # ensure array
    times = times[times != 0]  # remove zero entries (if any)

    if adv_speed == 0 and diff_coeff == 0: sys.exit("Error: adv_speed and diff_coeff cannot both be 0")
    if order not in [2, 4, 6, 14]: sys.exit("Error: The order must be either 2, 4, 6 or 14")
    if sim_type not in ["sv", "meas", "both"]: sys.exit("Error: sim_type must be either sv', 'meas' or 'both'")

    # Determine PDE evolution type
    method = "pure_diff" if adv_speed == 0 else "pure_adv" if diff_coeff == 0 else "adv_diff"

    # Prepare containers for results
    meas_results, fourier_results, statevec_results = [], [], []
    num_qubits_totals, max_errors, success_rates, complexities = [], [], [], []

    # Choose number of ancilla qubits based on method and order
    num_anc = 4 if order == 2 else 6 if order == 14 else 5 
    if method == "pure_diff": num_anc -= 1
    num_qubits_total = num_qubits + num_anc
    num_qubits_totals = [num_qubits_total] * len(times)

    # Spatial discretization and time-evolution parameters
    dx = domain_length / (2 ** num_qubits)
    dt_factors = {2: 1, 4: 3 / 2, 6: 11 / 6, 14: 363 / 140}
    M_adv = adv_speed * times * dt_factors[order] / dx
    M_diff = diff_coeff * times * dt_factors[order] ** 2 / (dx ** 2)

    # scaling factors
    adv_scale = 0.95
    diff_scale = 0.95

    # Spatial grid and normalized initial state
    x = np.linspace(0, domain_length, 2 ** num_qubits, endpoint=False)
    init_values = init_f(x)
    if not np.all(init_values >= 0): sys.exit("Error: initial function must be non-negative on the domain")
    norm_init = np.linalg.norm(init_values)
    init_normalized = init_values / norm_init

    # Classical Fourier solution function
    if compute_exact:
        fourier_func = fourier_approximation(*fourier_coefficients(init_f, 1e-8, domain_length), domain_length)

    # Plot setup
    if plot:
        num_plots = len(times) + 1
        plt.figure(figsize=(10.5, 10))

    for i in range(len(times)):
        print(f"--- RESULTS FOR ORDER {order} AT TIME {times[i]} ---\n")    

        # Construct registers and circuit
        qr_anc = QuantumRegister(num_anc)
        qr = QuantumRegister(num_qubits)
        qc = QuantumCircuit(qr_anc, qr)

        # A secondary quantum circuit to measure gate counts without state preparation included
        qr1_anc = QuantumRegister(num_anc)
        qr1 = QuantumRegister(num_qubits)
        qc1 = QuantumCircuit(qr1_anc, qr1)

        # State preparation 
        qc.initialize(Statevector(init_normalized), qr)                     # prepare_state changed to initialize due to a bug in the 2d simulation

        # Generate and append QSVT angle sequences for QSVT evolution
        print(f"-- ANGLE SEQUENCES --")
        if method == "pure_adv": 
            angle_seq_cos, angle_seq_sin = jacobi_anger_exp_angles(M_adv[i], adv_scale, tolerance)
            qc1.append(Adv_Diff_QC.qsvt(num_qubits, angle_seq_cos, angle_seq_sin, method, order), qr1_anc[:] + qr1[:])
        elif method == "pure_diff": 
            angle_seq_even = jacobi_anger_squared_exp_angles(M_diff[i], diff_scale, tolerance)
            qc1.append(Adv_Diff_QC.qsvt_single(num_qubits, angle_seq_even, order), qr1_anc[:] + qr1[:])
        else:
            angle_seq_even, angle_seq_odd = combined_exp_angles(tolerance, M_diff[i], M_adv[i])
            qc1.append(Adv_Diff_QC.qsvt(num_qubits, angle_seq_odd, angle_seq_even, method, order), qr1_anc[:] + qr1[:])

        # Printing gate counts and circuit depth without the cost of state preparation
        if report_complexity:
            tqc = transpile(qc1, basis_gates=["u", "cx"])    # Generic 1Q gate and CNOT
            gts = tqc.count_ops()                           
            gate_1q = gts["u"]
            gate_2q = gts["cx"]                             
            complexities.append([gate_1q, gate_2q, gate_1q + gate_2q, tqc.depth()])                   
            print(f"\n-- COMPLEXITY-- Total: {gate_1q + gate_2q}\nCircuit depth after transpiling:{tqc.depth()}")

        # Appending the QSVT circuit to the state preparation circuit
        qc.append(qc1,qr_anc[:] +  qr[:])
        
        # Fourier approximation
        if compute_exact:
            fourier_result = fourier_func(x, times[i], adv_speed, diff_coeff)
            fourier_results.append(fourier_result)

        # Statevector simulation
        if sim_type != "meas":
             # using Aer statevector simulator
            simulator = AerSimulator(method='statevector')
            qc_sv = qc.copy()
            qc_sv.save_statevector()
            qc_sv = transpile(qc_sv, simulator)
            gphase = qc_sv.global_phase 
            result = simulator.run(qc_sv).result()
            sv_data = result.get_statevector(qc_sv)
            statevec_result = np.asarray(sv_data).reshape(2 ** num_qubits, 2 ** num_anc)[:, 0]
            if sim_type != "both":
                success_rate_sv = np.linalg.norm(statevec_result) ** 2
                success_rates.append(success_rate_sv)
            # Reversing the global phase and rescaling 
            statevec_result *= np.exp(-1j*gphase)*2 * norm_init/(adv_scale if method == "pure_adv" else 2 * diff_scale if method == "pure_diff" else 1)
            statevec_results.append(statevec_result.real)

        # Measurement simulation
        if sim_type != "sv":
            qc.measure_all() 
            sim = AerSimulator(method = 'automatic')
            qc_comp = transpile(qc, sim) 
            sampler = SamplerV2()
            job = sampler.run([qc_comp],shots = shots)
            result = job.result() 
            print(sim.options.method)

            counts = result[0].data.meas.get_counts()

            # Postselection
            total_selected = 0        
            meas_result = np.zeros(2 ** num_qubits)
            select_str = "0" * num_anc
            for key in counts:
                if key[-num_anc:] ==  select_str:
                    idx = int(key[:-num_anc], 2)
                    meas_result[idx] = np.sqrt(counts[key]/shots)*2
                    meas_result[idx] *= norm_init / (adv_scale if method=="pure_adv" else 2 * diff_scale if method=="pure_diff" else 1)  # Reverse scaling factors
                    total_selected += counts[key] 
            meas_results.append(meas_result)
            success_rate = total_selected / shots
            success_rates.append(success_rate)
            print(f"\n-- SUCCESS RATE -- \nsucces rate of postselection: {success_rate:.4f}\n ")


        # Max error
        if compute_exact:
            print("-- MAX ERROR --")
            max_err_entry = []
            if sim_type != "sv":
                max_err_meas = np.max(np.abs(meas_result - fourier_result))
                print(f"Max error from measurement: {max_err_meas:.3e}")
                max_err_entry.append(max_err_meas)
            if sim_type != "meas":
                max_err_sv = np.max(np.abs(statevec_result - fourier_result))
                print(f"Max error from statevector: {max_err_sv:.3e}\n")
                max_err_entry.append(max_err_sv)
            max_errors.append(max_err_entry)

        # Plot
        if plot:
            if i == 0:
                plt.subplot(num_plots, 1, 1)
                plt.plot(x, init_f(x))  
                y_min, y_max = np.min(init_f(x)), np.max(init_f(x))
                plt.ylim(y_min - 0.05, y_max + 0.05)
                plt.title("Initial Condition")
            plt.subplot(num_plots, 1, i + 2)
            if sim_type != "sv": plt.plot(x, meas_result, label="Quantum Measurements")
            if sim_type != "meas": plt.plot(x, statevec_result.real, label="Quantum Statevector")
            if compute_exact: plt.plot(x, fourier_result, label="Exact Solution (Fourier)")
            plt.ylim(y_min - 0.05, y_max + 0.05)
            plt.title(rf"Results at Time $T = {times[i]}$")
            plt.legend()

    if plot:
        # Table 
        print("-- SUMMARY --")
        table, headers = [], ["time"]
        for i, t in enumerate(times):
            row = [f"{t}"]
            if compute_exact: row.append(f"{max_errors[i][0]:.3e}")
            row.append(f"{success_rates[i]:.4f}")
            if report_complexity: 
                row.append(f"{complexities[i][0]}")
                row.append(f"{complexities[i][1]}")
            table.append(row)

        if compute_exact:
            if sim_type == "meas": headers += ["meas max error"]
            else: headers += ["sv max error"]
        headers += ["success rate"]
        if report_complexity: headers += ["1-qubit gates", "2-qubit gates"]        # Removed: if sim_type != "sv" and

        print(tabulate(table, headers=headers, tablefmt="simple_grid", colalign=("center",)*len(headers)))

        # Plot
        if method == "pure_adv": 
            title_str = f"Advection Simulation of Order {order} with {num_qubits} Spatial Qubits\n" \
                        rf"Advection Speeds is $c = {adv_speed:.3g}$"
        elif method == "pure_diff": 
            title_str = f"Diffusion Simulation of Order {order} with {num_qubits} Spatial Qubits\n" \
                        rf"Diffusion Coefficient is $\nu = {diff_coeff:.3g}$"
        else: 
            title_str = f"Advection-Diffusion Simulation of Order {order} with {num_qubits} Spatial Qubits\n" \
                        rf"Diffusion Coefficient is $\nu = {diff_coeff:.3g}$, and Advection Speed is $c = {adv_speed:.3g}$"
        plt.suptitle(title_str, fontsize=15)
        plt.tight_layout()
        plt.show()
    
    return x, init_f(x), meas_results, fourier_results, statevec_results, num_qubits_totals, max_errors, success_rates, complexities

