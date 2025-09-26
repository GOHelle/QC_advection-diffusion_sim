import numpy as np
import sys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from tabulate import tabulate
from typing import Union
from Adv_Diff import Adv_Diff_QC
from Adv_Diff.Angles_QSVT import JA_exp_Angles, JA_2exp_Angles, comb_exp_Angles
from Adv_Diff.Fourier import Fourier_coef, Fourier_approx

"""
This module provides a quantum simulation method to approximate the solution of the advection-diffusion equation.

Method `Sim` utilizes QSVT methods defined in `Adv_Diff_QC.py` to simulate the evolution of an initial condition 
under the advection-diffusion PDE. It compares the quantum-based solution to a classical Fourier-based approximation 
defined in `Fourier.py`.
"""

def Sim(n: int, T: np.array, c: float, nu: float, d:float=4, init_f = lambda x: np.exp(-10*(x-4/3)**2), shots:int=10**6, 
        Complexity:bool=True, order:int = 2, eps=10**(-6), sim_type: str="both", plot:bool=True):
    """ Quantum simulation of the advection-diffusion equation via QSVT and comparison to classical Fourier approximation.

    This function constructs and runs quantum circuits simulating the evolution of an initial function under
    the advection-diffusion PDE. It supports pure diffusion, pure advection, and combined advection-diffusion evolution, and method orders 2, 4 and 6.
    The output is compared to a classical Fourier-based approximation, and both results are plotted for visual inspection.

    Args:
        n: Number of qubits to discretize the spatial domain (2^n grid points).
        T: Array of final times at which the solution is evaluated. Each entry in T triggers a separate simulation run. 
            Could also be of type int or float.
        c: Advection speed parameter.
        nu: Diffusion coefficient.
        d: Length of the spatial domain.
        init_f: Initial condition function f(x). Default is a Gaussian centered at 4/3.
        shots: Number of measurement shots for the quantum simulation.
        Complexity: If True, prints number of 1-qubit gates, number of 2-qubit gates, total number of gates, and circuit depth after transpiling.
        order: Order of the method. Supported values are 2, 4, or 6.
        eps: Tolerance parameter used for angle calculations.
             sim_type: If sim_type="sv", statevector simulation is performed, if sim_type="meas", measurement is performed, and if sim_type="both", both simulations are performed.
        plot: If True, plots the initial condition and the quantum and Fourier solutions for each value in T.

    Outputs:
        - A matplotlib figure with subplots for the initial condition and a comparison plot (quantum vs Fourier) for each final time value in T if plot = True.
        - gate counts and transpiled circuit depth if `Complexity=True`.
        - success rate of the postselection if measurement is performed.
        - Max error value(s)
        - A table summarizing all the printed outputs for each value in T.

    Returns:
        x: The space discretization
        f_Scaled(x): The scaled inital function on the discretized space
        z_list: a list of quantum solutions from measurement for each value in T. Note this is only non-empty if plot = "meas" or "both".
        w_list: a list of Fourier solutions for each value in T.
        W_list: a list of quantum solutions from statevector simulation for each value in T. Note this is only non-empty if plot = "sv" or "both".
        max_err_list: a list of max errors for each value in T. Each element is a list containing the max error found from the measurement outcomes, 
                      and/or the statevector simulation, dependent on plot.
        complexity_list: a list of lists containing the number of 1-qubit gates, number of 2-qubit gates, total number of gates, and circuit depth 
                         after transpiling for each value in T. Note this is only non-empty if Complexity=True.

    Notes:
        - If c=0, only advection is performed. If nu = 0 only diffusion is performed and if neither are 0, the QSVT for combined advection-diffusion is applied. 
          The simulation therefore selects the required number of ancilla qubits based on the simulation type and QSVT order. 
        - Postselection on ancillary measurements is performed to extract the quantum output.
    """

    # exceptions and input formatting
    T = np.atleast_1d(T)  # Ensure array
    T = T[T != 0]  # Remove any zero entries
    if c == 0 and nu == 0: sys.exit("Error: c and nu cannot both be 0")
    if order not in [2,4,6]: sys.exit("Error: The order should be either 2, 4 or 6")
    if sim_type not in ["sv", "meas", "both"]: sys.exit("Error: plot should be either sv', 'meas' or 'both'")

    # Identify which PDE evolution type applies
    method = "pure_diff" if c==0 else "pure_adv" if nu==0 else "adv_diff"

    # Choose number of ancilla qubits based on method and order
    if (method in ["adv_diff","pure_adv"]) and order>2: anc=5
    elif method=="pure_diff" and order==2: anc=3
    else: anc=4

    # Computing time-evoultion parameter M
    dx = d/(2**n)
    dt_factors = {2: 1, 4: 3/2, 6: 11/6}
    M_adv = c * T * dt_factors[order] / dx
    M_diff = nu * T * dt_factors[order]**2 / (dx**2)

    # sccaling
    adv_scale = 0.95
    diff_scale = 0.95

    # Spatial grid
    x = np.linspace(0,d,2**n,endpoint=False)
    y = init_f(x)
    if not np.all(y >= 0): sys.exit("Error: initial function not positive")
    norm_y = np.linalg.norm(y)
    y /= norm_y

    # Classical Fourier solution function
    g = Fourier_approx(*Fourier_coef(init_f,1e-5,d),d)

    # For storing solution values to return
    z_list, w_list, W_list, max_err_list, success_rate_list, complexity_list = [], [], [], [], [], []

    # Preparing plot
    if plot:
        num_plots = len(T) + 1
        plt.figure(figsize=(10, 9))

    for i in range(len(T)):
        print(f"--- RESULTS FOR T = {T[i]}, ORDER = {order} ---\n")

        qr_anc, cr_anc = QuantumRegister(anc), ClassicalRegister(anc)
        qr, cr = QuantumRegister(n), ClassicalRegister(n)
        qc = QuantumCircuit(qr_anc, qr, cr_anc, cr)

        # State preparation 
        qc.prepare_state(Statevector(y),qr)

        # Generate phase angles for QSVT evolution
        print(f"-- ANGLE SEQUENCES --")
        if method == "pure_adv": # We only have to apply the advection QSVT
            Phi_cos, Phi_sin = JA_exp_Angles(M_adv[i], adv_scale, eps)
            qc.append(Adv_Diff_QC.QSVT(n, Phi_cos, Phi_sin, method, order), qr_anc[:] + qr[:])
        elif method == "pure_diff":  # We only have to apply the diffusion QSVT
            Phi_even = JA_2exp_Angles(M_diff[i], diff_scale, eps)
            qc.append(Adv_Diff_QC.QSVT_single(n, Phi_even, order), qr_anc[:] + qr[:])
        else:
            Phi_even, Phi_odd = comb_exp_Angles(eps, M_diff[i], M_adv[i])
            qc.append(Adv_Diff_QC.QSVT(n, Phi_odd, Phi_even, method, order), qr_anc[:] + qr[:])

        # Fourier approximation
        w = g(x,T[i],c,nu)
        w_list.append(w)

        # State vector simulation
        if sim_type != "meas":
            sv = Statevector.from_instruction(qc)
            W = np.asarray(sv.data).reshape(2**n, 2**anc)[:,0]
            W *= 2*norm_y/(0.95 if method=="pure_adv" else 2*0.95 if method=="pure_diff" else 1)    
            W_list.append(W)

        # Measurement simulation
        if sim_type != "sv":
            qc.measure(qr_anc,cr_anc)
            qc.measure(qr,cr)
            sim = AerSimulator()
            qc_comp = transpile(qc,sim)
            res = sim.run(qc_comp,shots = shots).result()
            counts = res.get_counts(0)

            # Postselection
            total = 0        
            z = np.zeros(2**n)
            select = '0'*anc
            for key in counts:
                L = key.split()
                if L[1] == select:
                    z[int(L[0],2)] = np.sqrt(counts[key]/shots)*2
                    z[int(L[0],2)] *= norm_y / (adv_scale if method=="pure_adv" else 2*diff_scale if method=="pure_diff" else 1)  # Reverse scaling factors
                    total += counts[key] 
            z_list.append(z)
            success_rate = total/shots
            success_rate_list.append(success_rate)
            print(f"\n-- SUCCESS RATE -- \n succes rate of postselection: {success_rate}\n ")

            # Printing gate counts and circuit depth
            if Complexity:
                tqc = transpile(qc, basis_gates=["u", "cx"])    # Generic 1Q gate and CNOT
                gts = tqc.count_ops()
                gate_1q = gts['u']
                gate_2q = gts['cx']
                complexity_list.append([gate_1q, gate_2q, gate_1q+gate_2q, tqc.depth()])                   
                print(f"-- COMPLEXITY-- \nTotal: {gate_1q+gate_2q}\nCircuit depth after transpiling:{tqc.depth()}\n")

        # Max error
        print("-- MAX ERROR --")
        max_err = []
        if sim_type != "sv":
            max_err_meas = np.max(np.abs(z - w))
            print(f"Max error from measurement: {max_err_meas}")
            max_err.append(max_err_meas)
        if sim_type != "meas":
            max_err_sv = np.max(np.abs(W - w))
            print(f"Max error from statevector: {max_err_sv}\n")
            max_err.append(max_err_sv)
        max_err_list.append(max_err)

        # Plot
        if plot:
            if i == 0:
                plt.subplot(num_plots, 1, 1)
                plt.plot(x,init_f(x))  
                y_min, y_max = 0, np.max(init_f(x))
                plt.ylim(y_min-0.05, y_max+0.05)
                plt.title(rf"Inital Condition")
            plt.subplot(num_plots,1,i+2)
            if sim_type != "sv": plt.plot(x, z, label="Quantum measurements")
            if sim_type != "meas": plt.plot(x, W.real, label="Quantum statevector")
            plt.plot(x,w,label="Exact (fourier)")
            plt.ylim(y_min-0.05,y_max+0.05)
            plt.title(f"T={T[i]}")
            plt.legend()

    if plot:
        # Table 
        print(f"-- SUMMARY --")
        table, headers = [], ["T"]
        for i,t in enumerate(T):
            row = [t]
            row.append(max_err_list[i][0])
            if sim_type != "sv":
                row.append(success_rate_list[i])
                if Complexity: row.extend(complexity_list[i][0:2])
            table.append(row)

        if sim_type == "meas": headers += ["meas max error"]
        else: headers += ["sv max error"]
        if sim_type != "sv":
            headers += ["success rate"]
            if Complexity: headers += ["1-qubit gates","2-qubit gates"]

        print(tabulate(table,headers=headers,tablefmt="simple_grid"))

        # Plot
        if method == "pure_adv": title_str = rf"Pure Advection with Parameters; $n = {n},\ c = {c},\ order = {order}$"
        elif method == "pure_diff": title_str = rf"Pure Diffusion with Parameters; $n = {n},\ \nu = {nu},\ order = {order}$"
        else: title_str = rf"Advection-Diffusion with Parameters; $n = {n},\ \nu = {nu},\ c = {c},\ order = {order}$"
        plt.suptitle(title_str,fontsize=15)
        plt.tight_layout()
        plt.show()
    
    return x, init_f(x), z_list, w_list, W_list, max_err_list, success_rate_list, complexity_list
