from pyqsp import angle_sequence, response
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebmul
from scipy.special import jv, jve
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from typing import Callable, Tuple

""" 
This module provides utilities for generating QSP angle sequences for Chebyshev-based approximations of exp(iMx), exp(-Mx²), and exp(-Mx²+iMx)

Functions:
- symmetric_qsp_angles: Uses the pyqsp implementation of qsp_sym to find an optimal angle sequence for a
                        given Chebyshev coefficient sequence.
- jacobi_anger_exp, jacobi_anger_exp_angles: Jacobi-Anger expansion and angle generation for exp(iMx)
- jacobi_anger_squared_exp, jacobi_anger_squared_exp_angles: Jacobi-Anger expansion and angle generation for exp(-Mx²)
- plot_max_error, min_expansion_degree: Error plotting and minimum degree determination for approximations
- combined_exp, combined_exp_angles: Chebyshev coefficients and angle generation for exp(-Mx² + iMx)
"""
    
def symmetric_qsp_angles(cheb_coeffs: np.array, plot_response: bool = False, target_function: Callable | None = None, qsvt_format: bool = True) -> np.array:
    """ Compute symmetric QSP angle sequence for a given Chebyshev coefficient array.

    Args:
        cheb_coeffs: Array of Chebyshev coefficients representing the target polynomial
        plot_response: If True and a target_function is provided, the function plots the QSP response against the target.
        target_function: Target function to compare response.
        qsvt_format: If True, convert angles to QSVT-compatible format.
                     If False, return the raw phase sequence from the solver.

    Returns:
        Array of Phase angles for use in a quantum circuit.
    """

    # Compute symmetric QSP phase sequence using pyqsp's solver
    phiset, _, _ = angle_sequence.QuantumSignalProcessingPhases(
        poly=cheb_coeffs,
        eps=1e-6,
        suc=1 - 1e-6,
        signal_operator="Wx",
        measurement=None,
        tolerance=0.01,
        method="sym_qsp",
        chebyshev_basis=True
    )

    # Plot the angle response against the target function if requested
    if plot_response and target_function is not None:
        response.PlotQSPResponse(
            phiset,
            target=target_function,
            signal_operator="Wx",
            measurement="z",
            sym_qsp=True,
            simul_error_plot=True,
            title="Comparison"
        )
    
    # Transform the phase sequence into QSVT-compatible format if requested.
    if not qsvt_format:
        return phiset
    else:
        n = len(phiset) - 1
        qsvt_angles = np.zeros(n)
        qsvt_angles[1:n] = phiset[1:n] - np.pi / 2
        qsvt_angles[0] = phiset[0] + phiset[-1] + ((n - 2) % 4) * np.pi / 2
        # In the ususal QSP to QSVT one adds (n-1) pi/2. Here (n-2) is needed. 
        # In the symmetric qsp protocol the target polynomial is encoded as Im P wrt. the standard pair (P,Q)
        # where the symmetry forces Q = Q* to be real. In my previous work the target function is Re P instead
        # and in that case one uses n-1 instead.
        return qsvt_angles


def jacobi_anger_exp(frequency: float, degree: int) -> np.ndarray:
    """Compute Chebyshev coefficients of exp(i * frequency * x) using the Jacobi-Anger expansion.

    Args:
        frequency : Parameter in exp(i * frequency * x).
        deg: Degree of the Chebyshev expansion.

    Returns:
        Array of Chebyshev coefficients.
    """

    cheb_coeffs = np.zeros(degree + 1, dtype=complex)
    for d in range(degree + 1):
        cheb_coeffs[d] = 2 * (1j ** d) * jv(d, frequency)
    cheb_coeffs[0] /= 2  # correct the first term since J_0 is not doubled
    return cheb_coeffs

def jacobi_anger_exp_angles(frequency: float, scale: float, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Generate QSP angle sequences for the real and imaginary parts of exp(i * frequency * x)

    Args:
        frequency: Parameter in the exponential.
        scale: Scaling factor applied to the coefficients.
        eps: Desired max approximation error.

    Returns:
        Tuple of QSP angles for real and imaginary components.
    """
    degree = 2 * min_expansion_degree(tolerance, frequency, func_type="exp") + 1
    
    if degree % 2 == 0:
        degree += 1  # ensure odd degree
    
    cheb_coeffs = scale * jacobi_anger_exp(frequency, degree)
    
    return (symmetric_qsp_angles(np.real(cheb_coeffs)), symmetric_qsp_angles(np.imag(cheb_coeffs)))


def jacobi_anger_squared_exp(frequency: float, degree: int) -> np.ndarray:
    """
    Compute Chebyshev coefficients for exp(-frequency * x²) via the Jacobi-Anger expansion.

    Args:
        frequency: Parameter in exp(-frequency * x²).
        degree: Degree of the Chebyshev expansion.

    Returns:
        Array of Chebyshev coefficients.
    """
    cheb_coeffs = np.zeros(degree + 1)
    z = 1j * frequency / 2
    cheb_coeffs[0] = jve(0, z).real
    for n in range(2, degree + 1, 2):
        cheb_coeffs[n] = (2 * 1j ** (n / 2) * jve(int(n / 2),z)).real
    return cheb_coeffs 

def jacobi_anger_squared_exp_angles(frequency: float, scale: float = 1.0, tolerance: float = 1e-6) -> np.ndarray:
    """Generate QSP angles for exp(-frequency * x²) with minimal degree to meet the tolerance.

    Args:
        frequency: Parameter in the exponential.
        scale: Scaling factor for the coefficients.
        tolerance: Desired max approximation error.

    Returns:
        Array of QSP angles for approximating exp(-frequency * x²).
    """
    degree = 2 * min_expansion_degree(tolerance, frequency, func_type='squared_exp') + 1 
    
    if degree % 2 == 0: # ensure odd degree
        degree += 1
    
    return symmetric_qsp_angles(scale * jacobi_anger_squared_exp(frequency, degree))


def plot_max_error(frequency: int, degree_range: tuple[int, int, int], func_type: str = 'exp') -> None:
    """Plot maximum error of Chebyshev polynomial approximations over a range of degrees.

    Args:
        frequency: Parameter in the exponential.
        degree_range: A tuple (start, stop, step) for degrees to evaluate.
        func_type: Type of function to approximate; either 'exp' (for exp(i * frequency * x)) or 'squared_exp' (for exp(-M*x²)).
    """
    start, stop, step = degree_range
    x_vals = np.linspace(-1, 1, 1000)
    
    if func_type == 'exp':
        target_fn = lambda x: np.exp(1j * frequency * x)
        coeff_fn = jacobi_anger_exp
        title = f"exp(iMx, M = {frequency}"
    else:
        target_fn = lambda x: np.exp(-frequency * x ** 2)
        coeff_fn = jacobi_anger_squared_exp
        title = f"exp(-Mx²), M = {frequency}"
    
    errors = []
    for degree in range(start, stop, step):
        cheb_coeffs = coeff_fn(frequency, degree)
        poly = Chebyshev(cheb_coeffs, domain=[-1, 1])
        errors.append(np.max(np.abs(target_fn(x_vals) - poly(x_vals))))
    
    # Plotting
    plt.figure()
    plt.plot(range(start, stop, step), errors, marker='o')
    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Max Error')
    plt.title(f"Max Error vs Degree for {title}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# test
#for frequency in [5, 10, 20]:
#    err_deg(frequency, R=(10, 75, 5), func_type='exp')
#    err_deg(frequency, R=(10, 75, 5), func_type='squared_exp')

def min_expansion_degree(tolerance: float, frequency: float, func_type: str = 'exp') -> int:
    """Estimate minimal expansion degree for a given approximation tolerance.

    Args:
        tolerance: Target approximation error.
        frequency: Parameter in the exponential.
        func_type: Either 'exp' or 'squared_exp', determining which function to approximate.

    Returns:
        Minimum value of R such that the Chebyshev approximation error is ≤ eps.
    """

    def solve_r(eps, frequency):
        return fsolve(lambda r: (frequency / r) ** r - eps, frequency)[0]
    initial_degree = int(np.floor(solve_r(1.5 * tolerance, np.e * abs(frequency) / 4)))

    x_vals = np.linspace(-1, 1, 1000)
    f_vals = np.exp(1j * frequency * x_vals) if func_type == 'exp' else np.exp(-frequency * x_vals ** 2)

    for R in range(initial_degree, 0, -1):
        cheb_coeffs = jacobi_anger_exp(frequency, 2 * R + 1) if func_type == 'exp' else jacobi_anger_squared_exp(frequency, 2 * R + 1)
        approx_vals = np.polynomial.chebyshev.chebval(x_vals, cheb_coeffs)
        max_error = np.max(np.abs(f_vals - approx_vals))
        if max_error > tolerance or (R == initial_degree and max_error <= tolerance):
            return R+1      
      
        
def combined_exp(tolerance: float, frequency1: float, frequency2:float):
    """Compute Chebyshev coefficients for exp(-frequency1 * x² + i * frequency2 * x) with minimal degree to meet the tolerance.

    Args:
        tolerance: Desired approximation error.
        frequency1: First parameter in the exponential.
        frequency2: Second parameter in the exponential.

    Returns:
        Array of Chebyshev coefficients of the approximated function.
    """
    R_f = min_expansion_degree(tolerance / 2, frequency1, func_type="2exp")
    coeffs_f = jacobi_anger_squared_exp(frequency1, 2 * R_f + 1)

    R_g = min_expansion_degree(tolerance / 2, frequency2, func_type="exp")
    coeffs_g = jacobi_anger_exp(frequency2, 2 * R_g + 1)

    coeffs = chebmul(coeffs_f, coeffs_g)

    x_vals = np.linspace(-1, 1, 1000)
    f_vals = np.exp(-frequency1 * x_vals ** 2 + 1j * frequency2 * x_vals)

    for i in range(len(coeffs), 0, -1):
        approx = np.polynomial.chebyshev.chebval(x_vals, coeffs[:i])
        max_error = np.max(np.abs(f_vals - approx))
        if max_error > tolerance:
            if (i + 1) % 2 == 0:    # Ensure odd number of coefficients
                return coeffs[:i + 1]
            else:
                return coeffs[:i + 2]

def combined_exp_angles(tolerance: float, frequency1: float, frequency2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute QSP angles for exp(-frequency1 * x² + i * frequency2 * x).

    Args:
        tolerance: Target Approximation error.
        frequency1: First parameter in the exponential.
        frequency2: Second parameter in the exponential.

    Returns:
        Tuple containing angle sequences for real and imaginary parts:
    """

    combined_coeffs = combined_exp(tolerance, frequency1, frequency2)
    even_coeffs = [val if i % 2 == 0 else 0 for i, val in enumerate(np.real(combined_coeffs))]
    odd_coeffs = [val if i % 2 == 1 else 0 for i, val in enumerate(np.imag(combined_coeffs))]

    return symmetric_qsp_angles(even_coeffs), symmetric_qsp_angles(odd_coeffs) 



