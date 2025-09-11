import numpy as np
import sys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
# if the test file is added outside the folder
#from Adv_Diff import Adv_Diff_QC
#from Adv_Diff.Angles_QSVT import JA_exp_Angles, JA_2exp_Angles, comb_exp_Angles
#from Adv_Diff.Fourier import Fourier_coef_2d, Fourier_approx_2d

import Adv_Diff_QC
from Angles_QSVT import JA_exp_Angles, JA_2exp_Angles, comb_exp_Angles
from Fourier import Fourier_coef_2d, Fourier_approx_2d

"""
This module provides a quantum simulation method to approximate the solution of the 2D advection-diffusion equation.

Method `Sim` utilizes QSVT methods defined in `Adv_Diff_QC.py` to simulate the evolution of an initial function over a 2D 
spatial domain under advection-diffusion dynamics. It compares the quantum-based solution to a classical Fourier-based approximation 
defined in `Fourier.py`. It visualizes the initial condition, exact solution, and quantum solution via 3D surface plots.
"""

def Sim(n: int, T: float, c1: float, c2:float, nu: float, d:float=4, init_f=lambda X, Y: np.sin(np.pi * (0.5 * X + Y))**2, shots:int=10**7, Complexity:bool=True, order:int = 2, eps=10**(-6)):
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
        order: Order of the QSVT method (2, 4, or 6).
        eps: Tolerance parameter used for angle calculations.

    Outputs:
        - 3D surface plots of the initial condition, exact solution, and quantum solution at final time T. 
          The sucess rate and maximal error are also displayed
        - Prints gate counts and circuit depth if `Complexity=True`.

    Returns:
        init_vals: inital condition over space discretization
        exact: fourier approximation
        z: quantum solution

    Notes:
        - For each direction: if c=0, only diffusion is performed; if nu=0, only advection is performed; 
          if neither is zero, the QSVT for combined advection-diffusion is applied.
        - The number of ancilla qubits is chosen automatically based on the type of evolutions in each direction and the QSVT order.
        - Postselection on ancillary measurements is performed to extract the quantum output. The success rate is displayed in the plot.
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
        sys.exit("Error: if nu=0, c1 or c2 should be nonzero, to avoid 1d evoultion")
    if order not in [2,4,6]:
        sys.exit("Error: The order should be either 2, 4 or 6")

    # Computing time-evoultion parameter M
    dt_factors = {2: 1, 4: 3/2, 6: 11/6}
    factor = dt_factors[order]
    M_adv1 = c1 * T * factor / dx
    M_adv2 = c2 * T * factor / dx
    M_diff = nu * T * factor**2 / (dx**2)

    adv_scale = 0.95
    diff_scale = 0.95

    anc1 = 3 if (method1 == "pure_diff" and order == 2) else 5 if (method1 != "pure_diff" and order > 2) else 4
    anc2 = 3 if (method2 == "pure_diff" and order == 2) else 5 if (method2 != "pure_diff" and order > 2) else 4
    anc = anc1 + anc2

    qr_anc = QuantumRegister(anc, name = 'Ancilla')
    cr_anc = ClassicalRegister(anc)
    qr = QuantumRegister(2*n)
    cr = ClassicalRegister(2*n)
    qc = QuantumCircuit(qr_anc, qr, cr_anc, cr)
 
    # Initial state preparation
    init_vals = init_f(X, Y) 
    if not np.all(init_vals >= 0):
        sys.exit("Error: function is not positive on the domain.")   
    norm = np.linalg.norm(init_vals)       
    init_vals /= norm
    qc.prepare_state(Statevector(init_vals.flatten()),qr) 

    # QSVT evoultion
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

    qsvt1 = build_qsvt(method1, M_adv1)  # QSVT evoultion in x direction
    qc.append(qsvt1, qr_anc[:anc1] + qr[:n])

    qsvt2= build_qsvt(method2, M_adv2)  # QSVT evoultion in y direction
    qc.append(qsvt2, qr_anc[anc1:] + qr[n:])

    # Measurements
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
    print(f"-- SUCCESS RATE -- \n succes rate of postselection: {success_rate}\n ")
    
    # Rescale
    z /= adv_scale if method1 == "pure_adv" else (2 * diff_scale if method1 == "pure_diff" else 1)
    z /= adv_scale if method2 == "pure_adv" else (2 * diff_scale if method2 == "pure_diff" else 1)

    if Complexity:
        dict = qc_comp.count_ops()
        gate_1q = sum(v for k, v in dict.items() if k[0] != 'c' and k != 'measure')
        gate_2q = sum(v for k, v in dict.items() if k[0] == 'c')
        print(f"\n-- COMPLEXITY--\n1 qubit gates: {gate_1q}\n2 qubit gates: {gate_2q}")
        print(f"Total: {gate_1q+gate_2q}\nCircuit depth after transpiling:{qc_comp.depth()}\n")

    # Exact solution
    f_scaled = lambda x, y: init_f(x, y) / norm
    if nu != 0:
        g_exact = Fourier_approx_2d(*Fourier_coef_2d(f_scaled, 1e-5, d), d)
        exact = g_exact(X, Y, T, nu, c1, c2)
    else:  # pure advection
        exact = f_scaled((X - c1 * T) % d, (Y - c2 * T) % d)

    max_err = np.max(np.abs(z - exact))  # compute maximal error between quantum result and exact solution

    # Plotting
    fig = plt.figure(figsize=(20, 8))

    z_min = min(np.min(init_vals), np.min(exact), np.min(z))  # compute global min/max
    z_max = max(np.max(init_vals), np.max(exact), np.max(z))
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)

    titles = ["T = 0", f"Exact solution at T = {T}", f"Quantum result at T = {T}\nSuccess rate = {success_rate:.3f}\nMax error = {max_err:.6e}", ]
    data = [init_vals, exact, z]

    for i, (title, Z) in enumerate(zip(titles, data), start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        # Fix axes limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    fig.suptitle(f"n = {n},    c1 = {c1},    c2 = {c2},    nu = {nu},    shots = 10^{int(np.log10(shots))}")
    plt.tight_layout()
    plt.show()

    return init_vals, exact, z

# Run 2D simulation with specified parameters. 
#Sim(5, 1, 1, 2, 0.1, order = 6)