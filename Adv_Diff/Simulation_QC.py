import numpy as np
import sys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from Plot_help import sci_notation
from tabulate import tabulate

# if the test file is added outside the folder
#from Adv_Diff import Adv_Diff_QC
#from Adv_Diff.Angles_QSVT import JA_exp_Angles, JA_2exp_Angles, comb_exp_Angles
#from Adv_Diff.Fourier import Fourier_coef, Fourier_approx

import Adv_Diff_QC
from Angles_QSVT import JA_exp_Angles, JA_2exp_Angles, comb_exp_Angles
from Fourier import Fourier_coef, Fourier_approx

"""
This module provides a quantum simulation method to approximate the solution of the advection-diffusion equation.

Method `Sim` utilizes QSVT methods defined in `Adv_Diff_QC.py` to simulate the evolution of an initial condition 
under the advection-diffusion PDE. It compares the quantum-based solution to a classical Fourier-based approximation 
defined in `Fourier.py`.
"""

def Sim(n: int, T: np.array, c: float, nu: float, d:float=4, init_f = lambda x: np.exp(-10*(x-4/3)**2), shots:int=10**6, 
        Complexity:bool=True, order:int = 2, eps=10**(-6), sv_sim:bool = True, plot:bool=True, max_error:bool=True):
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
        sv_sim: if True, statevector simulation is performed and the corresponding max error is shown in the plot.
        plot: if True, the plot is shown. If False, only the solutions are returned. 
        max_error: if True, the max error for each final time will be displayed in a table. 

    Outputs:
        - A matplotlib figure with subplots for the initial condition and a comparison plot (quantum vs Fourier) for each final time value in T.
        - gate counts and transpiled circuit depth if `Complexity=True`.
        - success rate of the postselection
        - A table of max error values for the different final time values.

    Returns:
        x: The space discretization
        f_Scaled(x): The scaled inital function on the discretized space
        z_list: a list of quantum solutions for each value in T
        w_list: a list of Fourier solutions for each value in T
        max_err_list: a list of max error for each value in T. Each element is a list containing the max error found from the measurement outcomes, 
                      as well as the max error found from the statevector simulation, given that sv_sim = True.


    Notes:
        - If c=0, only advection is performed. If nu = 0 only diffusion is performed and if neither are 0, the QSVT for combined advection-diffusion is applied. 
          The simulation therefore selects the required number of ancilla qubits based on the simulation type and QSVT order. 
        - A classical Fourier approximation is computed using `Fourier_coef` and `Fourier_approx` for comparison. The max error is displayed in the plot
        - Postselection on ancillary measurements is performed to extract the quantum output. The success rate is displayed in the plot.

    """

    dx = d/(2**n)

    if not isinstance(T, np.ndarray):  # Ensure T is a NumPy array for consistent indexing
        T = np.array([T]) if np.isscalar(T) else np.array(T)

    if 0 in T:
        sys.exit("Error: T cannot contain 0. The inital condition is plotted regardless of the choice of T")
    if c == 0 and nu == 0:
        sys.exit("Error: c and nu cannot both be 0")
    if order not in [2,4,6]:
        sys.exit("Error: The order should be either 2, 4 or 6")

    # Identify which PDE evolution type applies
    if c == 0:
        method = "pure_diff"
    elif nu == 0:
        method = "pure_adv"
    else:
        method = "adv_diff"

    # Computing time-evoultion parameter M
    dt_factors = {2: 1, 4: 3/2, 6: 11/6}
    M_adv = c * T * dt_factors[order] / dx
    M_diff = nu * T * dt_factors[order]**2 / (dx**2)

    if plot:
        num_plots = len(T) + 1
        plt.figure(figsize=(10, 9))

    adv_scale = 0.95
    diff_scale = 0.95

    # Choose number of ancilla qubits based on method and order
    if (method == "adv_diff" or method=="pure_adv") and order >2:
        anc = 5
    elif method == "pure_diff" and order==2:
        anc = 3
    else: 
        anc = 4

    # For storing solution values to return
    z_list = []
    w_list = []
    W_list = []
    max_err_list = []

    for i in range(len(T)):
        print(f"--- RESULTS FOR T = {T[i]}, ORDER = {order} ---\n")

        qr_anc = QuantumRegister(anc, name = 'Ancilla')
        cr_anc = ClassicalRegister(anc)
        qr = QuantumRegister(n)
        cr = ClassicalRegister(n)
        qc = QuantumCircuit(qr_anc, qr, cr_anc, cr)

        # State preparation 
        x = np.linspace(0, d, 2**n, endpoint=False)
        y = init_f(x)
        if not np.all(y >= 0):
            sys.exit("Error: function is not positive on the domain.")
        norm_y = np.linalg.norm(y)
        y = y/norm_y
        qc.prepare_state(Statevector(y),qr)

        # Generate phase angles for QSVT evolution
        print(f"-- ANGLE SEQUENCES --")
        if method == "pure_adv": # We only have to apply the advection QSVT
            Phi_cos, Phi_sin = JA_exp_Angles(M_adv[i], adv_scale, eps)
            adv_qc = Adv_Diff_QC.QSVT(n, Phi_cos, Phi_sin, method, order)
            qc.append(adv_qc, qr_anc[:] + qr[:])

        elif method == "pure_diff":  # We only have to apply the diffusion QSVT
            Phi_even = JA_2exp_Angles(M_diff[i], diff_scale, eps)
            diff_qc = Adv_Diff_QC.QSVT_single(n, Phi_even, order)
            qc.append(diff_qc, qr_anc[:] + qr[:])

        else:
            Phi_even, Phi_odd = comb_exp_Angles(eps, M_diff[i], M_adv[i])
            QSVT_qc = Adv_Diff_QC.QSVT(n, Phi_odd, Phi_even, method, order)

            qc.append(QSVT_qc, qr_anc[:] + qr[:])

        # Fourier approximation
        def f_scaled(x):
            return init_f(x)/norm_y

        g = Fourier_approx(*Fourier_coef(f_scaled, 1e-5, d),d)
        w = g(x,T[i],c,nu)
        w_list.append(w)

        # State vector simulation
        if sv_sim:
            sv = Statevector.from_instruction(qc)
            sv_array = np.asarray(sv.data).reshape(2**n, 2**anc)
            W = sv_array[:, 0]  # Postselection

            # Reverse scaling factors
            W *= 2
            if method == "pure_adv":
                W *= 1/adv_scale
            elif method == "pure_diff":
                W *= 1/(2*diff_scale)
            
            W_list.append(W)

        # Measurements
        qc.measure(qr_anc,cr_anc)
        qc.measure(qr,cr)

        # Running the circuit 
        sim = AerSimulator()
        qc_comp = transpile(qc,sim)
        res = sim.run(qc_comp,shots = shots).result()
        counts = res.get_counts(0)

        # Postselection
        total = 0                      # Tracks the number of successfull outcomes
        N = 2**n
        z = np.zeros(N)
        select = '0'*anc

        for key in counts:
            L = key.split()
            if L[1] == select:
                z[int(L[0],2)] = np.sqrt(counts[key]/shots)*2
                # Reverse scaling factors
                if method == "pure_adv":
                    z[int(L[0],2)] *= 1/adv_scale
                elif method == "pure_diff":
                    z[int(L[0],2)] *= 1/(2*diff_scale)
                total += counts[key] 
        z_list.append(z)

        success_rate = total/shots
        print(f"-- SUCCESS RATE -- \n succes rate of postselection: {success_rate}\n ")

        # Printing gate counts and circuit depth
        if Complexity:
            dict = qc_comp.count_ops()
            gate_1q = 0
            gate_2q = 0
            for key in dict:
                if key[0] == 'c':
                    gate_2q += dict[key]
                elif key != 'measure':
                    gate_1q += dict[key]
                
            print(f"-- COMPLEXITY--\n1 qubit gates: {gate_1q}\n2 qubit gates: {gate_2q}")
            print(f"Total: {gate_1q+gate_2q}\nCircuit depth after transpiling:{qc_comp.depth()}\n")

        # Plot
        max_err = [np.max(np.abs(z - w))]
        if sv_sim:
            max_err.append(np.max(np.abs(W - w)))
        max_err_list.append(max_err)
        if plot:
            if i == 0:
                plt.subplot(num_plots, 1, 1)
                plt.plot(x,f_scaled(x))  
                y_min, y_max = 0, np.max(f_scaled(x))
                plt.ylim(y_min-0.05, y_max+0.05)
                plt.title(rf"$T = 0$")

            plt.subplot(num_plots, 1, i + 2)
            plt.plot(x, z, label= f"Quantum")
            plt.plot(x, w, label= f"Fourier")
            plt.ylim(y_min-0.05, y_max+0.05)
            title_str = rf"$T = ${T[i]}"
            plt.title(title_str)
            plt.legend()

    if plot:
        plt.suptitle(
        rf"$n = {n},\ c = {c},\ \nu = {nu},\ order = {order}$",fontsize=17)
        plt.tight_layout()
        plt.show()

    if max_error:
        print("--MAX ERROR--")
        print(f"Number of shots for measurement: 10^{int(np.log10(shots))}")
        if sv_sim:
            print(tabulate([[f'T = {i}', max_err_list[i][0], max_err_list[i][1]] for i in range(len(T))], headers=['', 'measurement max error', 'statevector max error'], tablefmt="simple_grid"))
        else:
            print(tabulate([[f'T = {i}', max_err_list[i][0]] for i in range(len(T))], headers=['', 'measurement max error'], tablefmt="simple_grid"))
    
    return x, f_scaled(x), z_list, w_list, max_err_list

