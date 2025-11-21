import numpy as np
import sys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from typing import Union
from Adv_Diff import Adv_Diff_QC
from Adv_Diff.Angles_QSVT import JA_exp_Angles, JA_2exp_Angles, comb_exp_Angles
from Adv_Diff.Fourier import Fourier_coef_2d, Fourier_approx_2d


"""
This module provides a quantum simulation method to approximate the solution of the 2D advection-diffusion equation.

Method `Sim` utilizes QSVT methods defined in `Adv_Diff_QC.py` to simulate the evolution of an initial function over a 2D 
spatial domain under advection-diffusion dynamics. It compares the quantum-based solution to a classical Fourier-based approximation 
defined in `Fourier.py`. It visualizes the initial condition, exact solution, and quantum solution via 3D surface plots.
"""

def Sim(n: int, T: float, c1: float, c2:float, nu: float, d:float=4, init_f=lambda X, Y: np.sin(np.pi * (0.5 * X + Y))**2, 
        shots:int=10**7, Complexity:bool=True, order:int = 2, eps:float=10**(-6), sim_type:str="both", exact_sol:bool = True, plot:bool=True):
    """ Quantum simulation of the 2D advection-diffusion equation via QSVT and comparison to classical Fourier approximation.

    This function constructs and runs quantum circuits to simulate the evolution of a 2D initial function 
    under the advection-diffusion PDE by applying 1D advection-diffusion in both x and y directions. 
    It supports pure advection, pure diffusion, and combined advection-diffusion in each direction, for method orders 2, 4, and 6. 
    The quantum solution is compared to a classical Fourier-based approximation, and both are plotted for visual inspection.

    Args:
        n: Number of qubits per spatial dimension (total grid size is 2^n x 2^n).
        T: Final time for the evolution.
        c1: Advection velocity in the x direction.
        c2: Advection velocity in the y direction.
        nu: Diffusion coefficient.
        d: Length of each spatial dimension (domain is [0,d] x [0,d]).
        init_f: Initial 2D function of X and Y. Default is `sin(pi*(0.5*X + Y)) + 1`.
        shots: Number of measurement shots for the quantum simulation.
        Complexity: If True, prints 1-qubit, 2-qubit gate counts, total gates, and circuit depth.
        order: Order of the QSVT method (2, 4, 6 or 14).
        eps: Tolerance parameter used for angle calculations.
        sim_type: If sim_type="sv", statevector simulation is performed, if sim_type="meas", measurement is performed, and if sim_type="both", both simulations are performed.
        exact_sol: If True, computes the exact Fourier solution for comparison.
        plot: If True, plots the initial condition and the quantum and Fourier solutions.

    Outputs:
        - 3D surface plots of the initial condition, exact solution, and quantum solution at final time T if plot = True. 
        - Gate counts and circuit depth if Complexity=True.
        - Success rate of postselection
        - Max errors between quantum and exact solutions if exact_sol = True.

    Returns:
        x: Spatial grid points in x direction
        y: Spatial grid points in y direction
        init_vals: Inital condition over space discretization
        z: Measurement quantum solution. If sim_type ="sv", z = None
        exact: Fourier approximation. If exact_sol = False, exact = None
        W: Statevector quantum solution. If sim_type ="meas", W = None
        max_err: List of maximum errors from measurement and statevector solutions. If exact_sol = False, max_err = []
        success_rate: Success rate of postselection.
        complexity: List of complexity data [1-qubit gates, 2-qubit gates, total gates, circuit depth]. If Complexity=False, Complexity = None

    Notes:
        - For each direction: if c=0, only diffusion is performed; if nu=0, only advection is performed; 
          if neither is zero, the QSVT for combined advection-diffusion is applied.
        - The number of ancilla qubits is chosen automatically based on the type of evolutions in each direction and the QSVT order.
        - Postselection on ancillary measurements is performed to extract the quantum output. The success rate is printed.
    """
    N = 2**n  
    dx = d / N
    x = np.linspace(0, d, N)
    y = np.linspace(0, d, N)
    X, Y = np.meshgrid(x, y)

    # Identify which PDE evolution type applies
    method1 = "pure_adv" if nu == 0 else "pure_diff" if c1 == 0 else "adv_diff"
    method2 = "pure_adv" if nu == 0 else "pure_diff" if c2 == 0 else "adv_diff"

    if (nu == 0 and c1 == 0) or (nu == 0 and c2 == 0):
        sys.exit("Error: if nu=0, c1 or c2 should be nonzero, to avoid 1d evolution")
    if order not in [2,4,6,14]:
        sys.exit("Error: The order should be either 2, 4, 6 or 14")
    if sim_type not in ["sv", "meas", "both"]: sys.exit("Error: sim_type should be either sv', 'meas' or 'both'")

    # Computing time-evolution parameter M
    dt_factors = {2: 1, 4: 3/2, 6: 11/6, 14: 363/140}
    factor = dt_factors[order]
    M_adv1 = c1 * T * factor / dx
    M_adv2 = c2 * T * factor / dx
    M_diff = nu * T * factor**2 / (dx**2)

    adv_scale = 0.95
    diff_scale = 0.95

    anc1 = 4 if order ==2 else 6 if order == 14 else 5 
    anc2 = 4 if order ==2 else 6 if order == 14 else 5
    if method1 =="pure_diff": anc1 = anc1-1
    if method2 =="pure_diff": anc2 = anc2-1
    anc = max(anc1,anc2)+1        # Replace anc1 +anc2 by max + 1

    qr_anc = QuantumRegister(anc, name = 'Ancilla')
    cr_anc = ClassicalRegister(anc)
    qr = QuantumRegister(2*n)
    cr = ClassicalRegister(2*n)
    qc = QuantumCircuit(qr_anc, qr, cr_anc, cr)
    print("Ancillas = ", anc, "Spatial = ", 2*n, "Total = ", anc+2*n)
 
    # Initial state preparation
    init_vals = init_f(X, Y) 
    if not np.all(init_vals >= 0):
        sys.exit("Error: function is not positive on the domain.")   
    norm = np.linalg.norm(init_vals)       
    init_vals /= norm
    qc.prepare_state(Statevector(init_vals.flatten()),qr) 

    # QSVT evolution
    print(f"-- ANGLE SEQUENCES --")
    def build_qsvt(method, M_adv):                       
        if method == "pure_adv":
            Phi_cos, Phi_sin = JA_exp_Angles(M_adv, adv_scale, eps)
            return Adv_Diff_QC.QSVT(n, Phi_cos, Phi_sin, method, order)
        elif method == "pure_diff":
            Phi_even = JA_2exp_Angles(M_diff, diff_scale, eps)
            return Adv_Diff_QC.QSVT_single(n, Phi_even, order)
        else:
            Phi_even, Phi_odd = comb_exp_Angles(eps, M_diff, M_adv)
            return Adv_Diff_QC.QSVT(n, Phi_odd, Phi_even, method, order)

    qsvt1 = build_qsvt(method1, M_adv1)  # QSVT evolution in x direction
    qc.append(qsvt1, qr_anc[:anc-1] + qr[:n])
    
    qc.mcx(qr_anc[:anc-1],qr_anc[anc-1],ctrl_state = (anc-1)*'0')   # Composition trick 
    qc.x(qr_anc[anc-1])

    qsvt2= build_qsvt(method2, M_adv2)  # QSVT evolution in y direction
    qc.append(qsvt2, qr_anc[:anc-1] + qr[n:])

    # Statevector simulation
    W = None
    if sim_type != "meas": 
        sv = Statevector.from_instruction(qc)
        W = np.asarray(sv.data).reshape(N, N, 2**anc)[:, :, 0]
        if sim_type != "both":
            success_rate = np.linalg.norm(W)**2
        # Rescale according to method
        W *= 2**2
        W /= adv_scale if method1 == "pure_adv" else (2 * diff_scale if method1 == "pure_diff" else 1)
        W /= adv_scale if method2 == "pure_adv" else (2 * diff_scale if method2 == "pure_diff" else 1)

    # Measurements
    z  = None
    complexity = None
    if sim_type != "sv":
        qc.measure(qr_anc,cr_anc)
        qc.measure(qr,cr)

        # Running the circuit 
        sim = AerSimulator()
        qc_comp = transpile(qc,sim)
        res = sim.run(qc_comp,shots = shots).result()
        counts = res.get_counts(0)

        # Postselection
        z = np.zeros([N, N])
        total = 0
        anc_bits = '0' * anc

        for key, val in counts.items():
            L = key.split()
            if L[1] == anc_bits:
                bitstring = L[0]
                i = int(bitstring[:n], 2)
                j = int(bitstring[n:], 2)
                z[i, j] = np.sqrt(val / shots) * 4
                total += val

        success_rate = total / shots
        print(f"\n-- SUCCESS RATE -- \n succes rate of postselection: {success_rate} ")
        
        # Rescale
        z /= adv_scale if method1 == "pure_adv" else (2 * diff_scale if method1 == "pure_diff" else 1)
        z /= adv_scale if method2 == "pure_adv" else (2 * diff_scale if method2 == "pure_diff" else 1)

    # Printing gate counts and circuit depth
    if Complexity:
        tqc = transpile(qc, basis_gates=["u", "cx"])    # Generic 1Q gate and CNOT
        gts = tqc.count_ops()
        gate_1q = gts['u']
        gate_2q = gts['cx']                  
        print(f"\n-- COMPLEXITY-- \nTotal: {gate_1q+gate_2q}\nCircuit depth after transpiling:{tqc.depth()}\n")
        complexity = [gate_1q, gate_2q, gate_1q + gate_2q, tqc.depth()]

    # Exact solution
    exact = None
    if exact_sol:
        f_scaled = lambda x, y: init_f(x, y) / norm
        if nu != 0:
            g_exact = Fourier_approx_2d(*Fourier_coef_2d(f_scaled, 1e-5, d), d)
            exact = g_exact(X, Y, T, nu, c1, c2)
        else:  # pure advection
            exact = f_scaled((X - c1 * T) % d, (Y - c2 * T) % d)

    # Max error
    max_err = []
    if exact_sol:
        print(f"-- MAX ERROR --")
        if sim_type != "sv": 
            max_err.append(np.max(np.abs(z - exact)))
            print(f"Max error from measurement: {max_err[-1]}")
        if sim_type != "meas": 
            max_err.append(np.max(np.abs(W - exact)))
            print(f"Max error from statevector: {max_err[-1]}")

    # Plotting
    if plot:
        fig = plt.figure(figsize=(20, 8))

        z_min_init = np.min(init_vals)
        z_max_init = np.max(init_vals)
        if exact_sol:
            z_min_exact = np.min(exact)
            z_max_exact = np.max(exact)
        z_min_W, z_max_W = (np.min(W), np.max(W)) if sim_type != "meas" else (None, None)
        z_min_z, z_max_z = (np.min(z), np.max(z)) if sim_type != "sv" else (None, None)
        x_min, x_max = np.min(X), np.max(X)
        y_min, y_max = np.min(Y), np.max(Y)

        data = [init_vals, exact] if exact_sol else [init_vals]
        titles = ["Initial condition", f"Exact solution at T = {T}"] if exact_sol else ["Initial condition"]

        if sim_type != "meas": 
            data.append(W.real)
            titles.append(f"Statevector solution at T={T}")
        if sim_type != "sv":
            data.append(z)
            titles.append(f"Measurement solution at T={T}")

        for i, (title, Z) in enumerate(zip(titles, data), start=1):
            ax = fig.add_subplot(1, len(data), i, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')

            if title == "Statevector solution":
                ax.set_zlim(z_min_W, z_max_W)
            elif title == "Measurement solution":
                ax.set_zlim(z_min_z, z_max_z)
            else:
                ax.set_zlim(min(z_min_init, z_min_exact), max(z_max_init, z_max_exact)) if exact_sol else (z_min_init, z_max_init)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        if nu == 0: title_str = rf"2D Advection with Parameters; $n = {n},\ c_1 = {c1},\ c_2 = {c2},\ order = {order}$"
        elif c1 == 0 and c2 == 0: title_str = rf"2D Diffusion with Parameters; $n = {n},\ \nu = {nu},\ order = {order}$"
        else: title_str = rf"2D Advection-Diffusion with Parameters; $n = {n},\ c_1 = {c1},\ c_2 = {c2},\ \nu = {nu},\ order= {order}$"
        fig.suptitle(title_str, fontsize=16)
        plt.tight_layout()
        plt.show()

    return x, y, init_vals, z, exact, W, max_err, success_rate, complexity
