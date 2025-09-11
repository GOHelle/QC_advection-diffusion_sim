from pyqsp import angle_sequence, response
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebmul
from scipy.special import jv
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from typing import Callable

""" 
This module provides tools for generating angle sequences for 
approximating functions such as exp(iMx), exp(-Mx²), and exp(-Mx² + iMx) using 
Chebyshev polynomial expansions

The module contains functions Angles_symQSP, JA_exp, JA_exp_Angles, JA_2exp, JA_2exp_Angles, err_deg, min_R, comb_exp, and comb_exp_Angles.

Angles_symQSP uses the pyqsp implementation of qsp_sym to find an optimal angle sequence for a
given Chebyshev coefficient sequence. 

JA_exp gives the Jacobi-Anger expansion for exp(iMx)

JA_exp_Angles generates angles for exp(iMx) using JA_exp

JA_2exp gives the Jacobi-Anger expansion for exp(-Mx²)

JA_2exp_Angles generates angles for exp(-Mx²) using JA_2exp

err_deg exp(-Mx²) plots max error vs. degree for approximation

min_R determines the minimum required expansion degree for a given error tolerance

comb_exp computes Chebyshev coefficients for exp(-Mx² + iMx)

comb_exp_angles generates angles for exp(-Mx² + iMx) using comb_exp

"""  
    
def Angles_symQSP(coef: np.array, resp_plot: bool=False, targ_f: Callable=None, QSVT_format: bool=True) -> np.array:
    """ Return the angle sequence for a given Chebyshev coefficient sequence using the pyqsp implementation of qsp_sym.

    Args:
        coef: Array of Chebyshev coefficients representing the target polynomial
        resp_plot: If True and a target function is provided, the function plots the QSP response against the target.
        targ_f: The target function approximated by the Chebyshecv coefficients. Used for plotting comparison.
        QSVT_format: If True, convert the angle sequence into QSVT-compatible format.
                     If False, return the raw phase sequence from the solver.

    Returns:
        Phi: Array of rotation angles (phases) for use in a quantum circuit.
    """

    # Compute symmetric QSP phase sequence using pyqsp's solver
    (phiset, red_phiset, parity) = angle_sequence.QuantumSignalProcessingPhases(
        poly=coef,
        eps=1e-6,
        suc=1 - 1e-6,
        signal_operator="Wx",
        measurement=None,
        tolerance=0.01,
        method="sym_qsp",
        chebyshev_basis=True
    )

    # Plot the angle response against the target function if requested
    if resp_plot and targ_f:
        response.PlotQSPResponse(
            phiset,
            target=targ_f,
            signal_operator="Wx",
            measurement="z",
            sym_qsp=True,
            simul_error_plot=True,
            title="Comparison"
        )
    
    # Transform the phase sequence into QSVT-compatible format if requested.
    if QSVT_format:
        n = len(phiset)-1
        Phi = np.zeros(n)
        Phi[1:n] = phiset[1:n]-np.pi/2
        Phi[0] = phiset[0]+phiset[-1]+((n-2)%4)*np.pi/2
        # In the ususal QSP to QSVT one adds (n-1) pi/2. Here (n-2) is needed. 
        # In the symmetric qsp protocol the target polynomial is encoded as Im P wrt. the standard pair (P,Q)
        # where the symmetry forces Q = Q* to be real. In my previous work the target function is Re P instead
        # and in that case one uses n-1 instead. 
    else: 
        Phi = phiset 
    return Phi

def JA_exp(M: float, deg: int):
    """
    Computes Chebyshev coefficients of exp(i*M*x) using the Jacobi-Anger expansion.

    Args:
        M : Parameter in exp(i M x).
        deg: Degree of the Chebyshev expansion.

    Returns:
        coef: Array of Chebyshev coefficients.
    """

    coef = np.zeros(deg + 1, dtype=complex)
    for n in range(deg + 1):
        coef[n] = 2 * (1j)**n * jv(n, M)
    coef[0] /= 2  # correct the first term since J_0 is not doubled
    return coef

def JA_exp_Angles(M: float, scale: float, eps: float = 1e-6):
    """
    Computes QSP angle sequences for the real and imaginary parts of exp(i*M*x)

    Args:
        M: Parameter in the exponential.
        scale: Scaling factor applied to the coefficients.
        eps: Desired max approximation error.

    Returns:
        (Phi_even, Phi_odd): QSP angles for real and imaginary components.
    """
    deg = 2 * min_R(eps, M, func_type='exp') + 1
    
    if deg % 2 == 0:
        deg += 1  # ensure odd degree
    
    coef = scale * JA_exp(M, deg)
    Phi_even = Angles_symQSP(np.real(coef))
    Phi_odd = Angles_symQSP(np.imag(coef))
    
    return Phi_even, Phi_odd

def JA_2exp(M:float,deg):
    """
    Computes Chebyshev coefficients of exp(-M*x²) via the Jacobi-Anger expansion.

    Args:
        M: Parameter in exp(-M*x²).
        deg: Degree of the Chebyshev expansion.

    Returns:
        coef: Array of Chebyshev coefficients.
    """
    coef = np.zeros(deg +1)
    z = 1j*M/2
    coef[0] = jv(0,z).real
    for n in range(2,deg+1,2):
        c = 2*1j**(n/2)*jv(int(n/2),z)
        coef[n] = c.real
    coef = np.exp(-M/2)*coef 
    return coef 

def JA_2exp_Angles(M: float, scale: float = 1.0, eps: float = 1e-6):
    """
    Computes QSP angle sequence for exp(-M*x²) with minimal degree to meet eps.

    Args:
        M: Parameter in the exponential.
        scale: Scaling factor for the coefficients.
        eps: Desired max approximation error.

    Returns:
        Array of QSP angles for approximating exp(-M*x²).
    """
    deg = 2 * min_R(eps, M, func_type='2exp') + 1  # ensure odd degree
    
    if deg % 2 == 0:
        deg += 1
    
    coef = scale * JA_2exp(M, deg)
    Phi_even = Angles_symQSP(coef)
    
    return Phi_even

def err_deg(M: int, R: tuple, func_type='exp'):
    """
    Plots the maximum error of Chebyshev polynomial approximations over a range of degrees.

    Args:
        M: Parameter in the exponential.
        R: A tuple (start, stop, step) for degrees to evaluate.
        func_type: Type of function to approximate; either 'exp' (for exp(i*M*x)) or '2exp' (for exp(-M*x²)).

    Returns:
        None: Displays a plot of max error vs. degree.
    """
    start, stop, step = R
    x = np.linspace(-1, 1, 1000)
    
    if func_type == 'exp':
        f = lambda x: np.exp(1j* M * x)
        coef_fn = JA_exp
        title = f"exp(iMx, M = {M}"
    else:
        f = lambda x: np.exp(-M*x**2)
        coef_fn = JA_2exp
        title = f"exp(-Mx²), M = {M}"
    
    degrees = range(start, stop, step)
    errors = []
    for deg in degrees:
        coef = coef_fn(M, deg)
        P = Chebyshev(coef, domain=[-1, 1])
        err = np.max(np.abs(f(x) - P(x)))
        errors.append(err)
    
    # Plotting
    plt.figure()
    plt.plot(degrees, errors, marker='o')
    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Max Error')
    plt.title(f"Max Error vs Degree for {title}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# test
#for M in [5, 10, 20]:
#    err_deg(M, R=(10, 75, 5), func_type='exp')
#    err_deg(M, R=(10, 75, 5), func_type='2exp')

def min_R(eps:float, M:float, func_type:str='exp'):
    """
    Estimates the minimum expansion degree required to approximate a function within given error tolerance.

    Args:
        eps: Target approximation error.
        M: Parameter in the exponential.
        func_type: Either 'exp' or '2exp', determining which function to approximate.

    Returns:
        Minimum value of R such that the Chebyshev approximation error is ≤ eps.
    """
    def r(eps,M):
        r_val = fsolve(lambda r: (M/r)**r - eps,M)[0]
        return r_val
    R_init = int(np.floor(r(3/2*eps, np.e*abs(M)/4)))

    x_vals = np.linspace(-1, 1, 1000)
    if func_type == 'exp':
        f_vals = np.exp(1j*M*x_vals)
    else:
        f_vals = np.exp(-M*(x_vals)**2)

    for R in range(R_init, 0, -1):
        if func_type == 'exp':
            coeffs = JA_exp(M, 2*R+1)
        else:
            coeffs = JA_2exp(M, 2*R+1)
        approx = np.polynomial.chebyshev.chebval(x_vals, coeffs)
        max_error = np.max(np.abs(f_vals - approx))
        if max_error > eps:
            return R+1      

def comb_exp(eps:float, M1:float, M2:float):
    """
    Computes Chebyshev coefficients of the function exp(-M1*x² + i*M2*x)
    such that the max approximation error is within eps.

    Args:
        eps: Desired approximation error.
        M1: First parameter in the exponential.
        M2: Second parameter in the exponential.

    Returns:
        Array of Chebyshev coefficients of the approximated function.
    """

    R_f = min_R(eps/2, M1, func_type="2exp")
    coeffs_f = JA_2exp(M1,2*R_f+1)

    R_g = min_R(eps/2, M2, func_type="exp")
    coeffs_g = JA_exp(M2,2*R_g+1)

    coeffs = chebmul(coeffs_f, coeffs_g)

    x_vals = np.linspace(-1, 1, 1000)
    f_vals = np.exp(-M1*x_vals**2 + 1j*M2*x_vals)

    for i in range(len(coeffs), 0, -1):
        approx = np.polynomial.chebyshev.chebval(x_vals, coeffs[:i])
        max_error = np.max(np.abs(f_vals - approx))
        if max_error > eps:
            if (i+1) % 2 == 0:    # Ensure odd number of coefficients
                return coeffs[:i+1]
            else:
                return coeffs[:i+2]

def comb_exp_Angles(eps:float, M1:float, M2:float):
    """
    Computes QSP angles for the function exp(-M1 x² + i M2 x).

    Args:
        eps: Target Approximation error.
        M1: First parameter in the exponential.
        M2: Second parameter in the exponential.

    Returns:
        tuple: (Phi_even, Phi_odd) where:
            - Phi_even: Angle sequence for the real part.
            - Phi_odd: Angle sequence for the imaginary part.
    """

    coef = comb_exp(eps, M1, M2)
    coef_real = np.real(coef)
    coef_imag = np.imag(coef)
    coef_even = [val if i % 2 == 0 else 0 for i, val in enumerate(coef_real)]
    coef_odd  = [val if i % 2 != 0 else 0 for i, val in enumerate(coef_imag)]
    Phi_even = Angles_symQSP(coef_even)
    Phi_odd = Angles_symQSP(coef_odd)

    return Phi_even, Phi_odd 
