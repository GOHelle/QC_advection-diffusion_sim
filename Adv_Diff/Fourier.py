from scipy.integrate import quad, dblquad
import numpy as np
from typing import Callable, List, Tuple

""" 
This module provides tools for computing one-dimensional and two-dimensional Fourier coefficients,
as well as Fourier series solutions to the one-dimensional and two-dimensional advection–diffusion equation
 contains functions Fourier_coef, Fourier_approx, Fourier_coef_2d, and Fourier_approx_2d.

Functions:
fourier_coefficients, fourier_coefficients_2d: Return the Fourier coefficients for a given 1-dimensional or 2-dimensional function, respectively.
fourier_approximation, fourier_approximation_2d: Construct functions that provide Fourier series solutions to the 1-dimensional or 2-dimensional 
                                                 advection–diffusion equation, respectively.
"""  

def fourier_coefficients(f: Callable, tolerance: float, domain_length: float, max_order: int | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """ Compute the Fourier coefficients of a function f(x) defined on the interval [0, domain_length]

    If `max_order` is not provided, coefficients are computed adaptively until
    the maximum error satisfies `tolerance`. The number of fourier terms needed to converge is printed

    The approximation is constructed as:
    f(x) ≈ c + Σ_{k=1}^N  a_k cos(2π k x / L) + Σ_{k=1}^N  b_k sin(2π k x / L)
    where L is the domain length.

    Args:
        f: Function f(x) defined on [0, domain_length].
        tolerance: Maximum absolute approximation error for adaptive mode.
        domain_length: Length of the interval [0, domain_length] over which f is defined.
        Optional maximum Fourier mode. If None, adaptive detection
            is used.
        max_order: Optional maximum Fourier mode. If None, adaptive detection is used.

    Returns:
        A tuple containing:
        - The constant Fourier coefficient.
        - Array of cosine coefficients a_k for k = 1, ..., N
        - Array of sine coefficients b_k for k = 1, ..., N
    """

    def compute_mode_coefficients(k):
        """Compute the cosine and sine coefficients for mode k."""

        # Define the integrand for cosine and sine terms
        def integrand_cos(x):
            return f(x) * np.cos(2 * np.pi * k * x / domain_length)

        def integrand_sin(x):
            return f(x) * np.sin(2 * np.pi * k * x / domain_length)

        # Compute integrals
        cos_coeff_k = quad(integrand_cos, 0, domain_length)[0] * (2.0 / domain_length)
        sin_coeff_k = quad(integrand_sin, 0, domain_length)[0] * (2.0 / domain_length)

        return cos_coeff_k, sin_coeff_k

    # Compute zeroth coefficient.
    constant_coeff = quad(f, 0, domain_length)[0] / domain_length
    cos_coeffs, sin_coeffs = [], []

    # --- Non-adaptive mode ---
    if max_order is not None:
        for k in range(1, max_order + 1):
            cos_coeff_k, sin_coeff_k = compute_mode_coefficients(k)
            cos_coeffs.append(cos_coeff_k)
            sin_coeffs.append(sin_coeff_k)
        return constant_coeff, np.array(cos_coeffs), np.array(sin_coeffs)

    # --- Adaptive mode ---
    # Discretize the interval for error estimation
    x_grid = np.linspace(0, domain_length, 1000, endpoint=False)
    f_grid = np.array([f(x) for x in x_grid])

    # Initialize approximation with just the constant coefficient
    approximation = np.full_like(x_grid, constant_coeff, dtype=float)

    max_iter = 500
    k = 1
    while k <= max_iter:
        #compute new pair of coefficients
        cos_coeff_k, sin_coeff_k = compute_mode_coefficients(k)
        cos_coeffs.append(cos_coeff_k)
        sin_coeffs.append(sin_coeff_k)

        # update the approximation values with this new term
        freq = 2 * np.pi * k / domain_length
        approximation += cos_coeff_k * np.cos(freq * x_grid) + sin_coeff_k * np.sin(freq * x_grid)

        # check if we need more iterations to reach the error threshold.
        error = np.max(np.abs(f_grid - approximation))
        if error < tolerance:
            print(f"\n-- FOURIER --\nConverged with {k} Fourier modes\n")
            return constant_coeff, np.array(cos_coeffs), np.array(sin_coeffs)
        k += 1

    print(f"\n-- FOURIER --\nFourier series did not converge within {max_iter} steps\n")
    return constant_coeff, np.array(cos_coeffs), np.array(sin_coeffs)  

def fourier_approximation(constant_coeff: float, cos_coeffs: np.array, sin_coeffs: np.array, domain_length: float) -> Callable:
    """ Construct a callable evaluating the Fourier-series solution of the 1D advection–diffusion equation: u_t + c_adv * u_x = nu * u_xx
        The solution is constructed as:
        u(x, t) = c + Σ_{k=1}^N e^{-nu * n_k^2 * t} * [a_k * cos(n_k * (x − c_adv * t)) + b_k * sin(n_k * (x − c_adv * t))]
        where n_k = 2πk / domain_length.

    Args:
        constant_coeff: Zeroth Fourier coefficient c.
        cos_coeffs: Array of cosine coefficients a_k for k = 1, ..., N.
        sin_coeffs: Array of sine coefficients b_k for k = 1, ..., N.
        domain_length: Length of the interval [0, domain_length] over which the Fourier series is defined.

    Returns:
        A function u(x, t, c_adv, nu) evaluating the solution.
    """

    def solution(x, t, c_adv, nu):
        u = constant_coeff
        for k in range(len(cos_coeffs)):
            n_k = 2 * np.pi * (k + 1) / domain_length
            u += np.exp(-nu * (n_k**2) * t) * (cos_coeffs[k] * np.cos(n_k * (x - c_adv * t)) + sin_coeffs[k] * np.sin(n_k * (x - c_adv * t))) 
        return u
    return solution


def fourier_coefficients_2d(f: Callable, tolerance: float, domain_length: float, max_order: int | None = None) -> Tuple[float, np.array, np.array, List[Tuple[int, int]]]:
    """ Compute 2D Fourier coefficients of f(x, y) on [0, domain_length] × [0, domain_length].

    The representation is:
    (x, y) ≈ c + Σ A_{n1,n2} * cos(ω(n1 * x + n2 * y)) + Σ B_{n1,n2} * sin(ω(n1 * x + n2 * y))

    with ω = 2π / domain_length and modes (n1, n2) selected so that n1 ≥ 0 and if n1 = 0 then n2 > 0

    Args:
        f: Function f(x, y).
        tolerance: Target maximum absolute error for adaptive mode
        domain_length: Domain size in both x and y directions.
        max_order: Optional integer specifying the number of Fourier terms to compute. If omitted, the function adaptively determines
           the number of terms required to achieve the desired tolerance.
    
    Outputs:
        The number of fourier terms needed to converge.       

    Returns:
    Tuple containing:
        - The constant Fourier coefficient, c.
        - Array of Cosine coefficients, A.
        - Array of Sine coefficients, B.
        - List of (n1, n2) mode indices.
    """

    omega = 2 * np.pi / domain_length

    # Compute the zeroth coefficient.
    constant_coeff = dblquad(lambda y, x: f(x, y), 0, domain_length, lambda _: 0, lambda _: domain_length)[0] / (domain_length ** 2)

    def compute_mode_coefficients(n1: int, n2: int) -> Tuple[float, float]:
        """Compute the cosine and sine coefficients for mode (n1, n2)."""
        integrand_cos = lambda y, x: f(x, y) * np.cos(omega * (n1 * x + n2 * y))
        integrand_sin = lambda y, x: f(x, y) * np.sin(omega * (n1 * x + n2 * y))
        cos_coeff_n = dblquad(integrand_cos, 0, domain_length, lambda _: 0, lambda _: domain_length)[0] * (2 / (domain_length ** 2))
        sin_coeff_n = dblquad(integrand_sin, 0, domain_length, lambda _: 0, lambda _: domain_length)[0] * (2 / (domain_length ** 2))
        return cos_coeff_n, sin_coeff_n

    # --- Non-adaptive mode: fixed number of coefficients ---
    cos_coeffs, sin_coeffs, modes = [], [], []
    if max_order is not None:
        for n1 in range(0, max_order + 1):
            for n2 in range(-max_order, max_order + 1):
                if n1 == 0 and n2 <= 0:
                    continue
                cos_coeff_n, sin_coeff_n = compute_mode_coefficients(n1, n2)
                cos_coeffs.append(cos_coeff_n)
                sin_coeffs.append(sin_coeff_n)
                modes.append((n1, n2))
        return constant_coeff, np.array(cos_coeffs), np.array(sin_coeffs), modes

    # --- Adaptive mode: increase N until error < eps ---
    x_grid = np.linspace(0, domain_length, 200, endpoint=False)
    y_grid = np.linspace(0, domain_length, 200, endpoint=False)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")
    f_grid = f(X, Y)

    max_iter = 25
    approximation = np.full_like(f_grid, constant_coeff, dtype=float)

    # storage for final result only
    cos_coeffs, sin_coeffs, modes = [], [], []

    for N in range(1, max_iter + 1):
        new_modes = []
        # horizontal strip
        for n2 in range(-N, N + 1):
            if N == 0 and n2 <= 0:
                continue
            new_modes.append((N, n2))
        # vertical strip
        for n1 in range(0, N):
            for n2 in [-N, N]:
                if n1 == 0 and n2 <= 0:
                    continue
                new_modes.append((n1, n2))

        # compute coefficients and update approximation incrementally
        for (n1, n2) in new_modes:
            cos_coeff, sin_coeff = compute_mode_coefficients(n1, n2)

            cos_coeffs.append(cos_coeff)
            sin_coeffs.append(sin_coeff)
            modes.append((n1, n2))

            phase = omega * (n1 * X + n2 * Y)
            approximation += cos_coeff * np.cos(phase) + sin_coeff * np.sin(phase)

        # convergence check
        if np.max(np.abs(f_grid - approximation)) < tolerance:
            print(f"-- FOURIER --\nNumber of Fourier coefficients needed: {len(modes)}\n")
            return constant_coeff, np.array(cos_coeffs), np.array(sin_coeffs), modes
        
    print(f"\n-- FOURIER --\n2D Fourier series did not reach error < {tolerance} within N={max_iter}.\n")
    return constant_coeff, np.array(cos_coeffs), np.array(sin_coeffs), modes

def fourier_approximation_2d(constant_coeff: float, cos_coeffs: np.array, sin_coeffs: np.array, modes: List[Tuple[int, int]], domain_length: float) -> Callable:
    """Construct the Fourier-series solution of the 2D advection–diffusion equation u_t + v1 * u_x + v2 * u_y = nu * Δu

    The solution is constructed as:
        u(x, y, t) = c + Σ exp(-nu * ω² * (n1² + n2²) * t) * [A_{n1, n2} * cos(φ) + B_{n1, n2} * sin(φ)]
        where φ = ω * ((n1 * x + n2 * y) - (n1 * v1 + n2 * v2) * t) and ω = 2π / domain_length.

    Args:
        constant_coeff: Zeroth coefficient, c.
        cos_coeffs: Cosine coefficients A.
        sin_coeffs: Sine coefficients B.
        modes: List of index pairs (n1, n2).
        domain_length: Domain size.

    Returns:
        A function u(X, Y, t, nu, v1, v2) evaluating the solution.
    """
    omega = 2 * np.pi / domain_length
    def solution(X, Y, t, nu, v1, v2):
        u = np.full_like(X, constant_coeff, dtype=float)
        for (n1, n2), cos_coeff_n, sin_coeff_n in zip(modes, cos_coeffs, sin_coeffs):
            exp_factor = np.exp(-nu * (omega ** 2) * (n1 ** 2 + n2 ** 2) * t)
            phase = omega * ((n1 * X + n2 * Y) - (n1 * v1 + n2 * v2) * t)
            u += exp_factor * (cos_coeff_n * np.cos(phase) + sin_coeff_n * np.sin(phase))
        return u
    return solution
