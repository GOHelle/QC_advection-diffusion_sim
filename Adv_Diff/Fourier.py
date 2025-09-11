from scipy.integrate import quad, dblquad
import numpy as np
from typing import Callable, List, Tuple

""" 
This module contains functions Fourier_coef, Fourier_approx, Fourier_coef_2d, and Fourier_approx_2d.

Fourier_coef and Fourier_coef_2d return the Fourier coefficients for a given 1-dimensional or 2-dimensional function, respectively.

Fourier_approx and Fourier_approx_2d construct functions that provide Fourier series solutions to the 1-dimensional or 2-dimensional 
    advection–diffusion equation, respectively.

"""  

def Fourier_coef(f: Callable, eps: float, d: float, N:int=None) -> tuple[float, list, list]:
    """ Return the Fourier coefficients for a function f defined on the interval [0, d]

    This function calculates the Fourier series coefficients up to the N-th Fourier mode or,
    if `N` is not provided, it adaptively determines the number of terms required to achieve a
    given approximation error `eps`.

    Args:
        f: A function defined on [0, d].
        eps: Desired maximum absolute error between the function and its truncated Fourier approximation.
        d: Length of the interval [0, d] over which f is defined.
        N: Optional integer specifying the number of Fourier terms to compute. If omitted, the function adaptively determines
           the number of terms required to achieve the desired error `eps`.

    Returns:
        c: The constant Fourier coefficient.
        a: List of cosine coefficients a_k for k = 1, ..., N-1 (or adaptively chosen N).
        b: List of sine coefficients b_k for k = 1, ..., N-1 (or adaptively chosen N).

    Outputs:
        The number of fourier terms needed to converge.

    Notes:
        - The approximation f_N(x) is constructed as:
              f_N(x) = c + sum_{k=1}^{N-1} [a_k * cos(2πkx/d) + b_k * sin(2πkx/d)]
    """

    def compute_coef(k):
        """Compute the k-th cosine and sine Fourier coefficients a_k and b_k."""

        # Define the integrand for cosine and sine terms
        def g(x, t):
            if t == 0:
                return f(x) * np.cos(2 * np.pi * k * x / d)
            else:
                return f(x) * np.sin(2 * np.pi * k * x / d)

        # Compute integrals
        a_k = quad(g, 0, d, args=(0,))[0] / (d / 2)
        b_k = quad(g, 0, d, args=(1,))[0] / (d / 2)

        return a_k, b_k

    # Compute the zeroth coefficient.
    c = quad(f, 0, d)[0] / d
    a = []
    b = []

    # === Non-adaptive mode: fixed number of coefficients ===
    if N:
        for k in range(1, N):
            a_k, b_k = compute_coef(k)
            a.append(a_k)
            b.append(b_k)
        return c, a, b

    # === Adaptive mode: increase N until error < eps ===
    N = 1

    # Discretize the interval for error estimation
    x_vals = np.linspace(0, d, 1000, endpoint=False)
    f_vals = np.array([f(x) for x in x_vals])

    # Initialize approximation with just the constant coefficient
    approx_vals = np.full_like(x_vals, fill_value=c, dtype=float)


    max_iter = 500
    while N <= max_iter :
        #compute new pair of coefficients
        a_k, b_k = compute_coef(N)
        a.append(a_k)
        b.append(b_k)

        # update the approximation values with this new term
        n = 2 * np.pi * N / d
        approx_vals += a_k * np.cos(n * x_vals) + b_k * np.sin(n * x_vals)

        # check if we need more iterations to reach the error threshold.
        error = np.max(np.abs(f_vals - approx_vals))
        if error < eps:
            print(f"\n-- FOURIER --\nNumber of Fourier coefficients needed: {N}\n")
            return c, np.array(a), np.array(b)
        
        N += 1

    raise RuntimeError(f"Fourier series did not converge within {max_iter} steps.")
    

def Fourier_approx(c:float, a:list, b:list, d:float) -> Callable:
    """ Constructs a time-evolving Fourier approximation function for a solution to the advection-diffusion equation.

    This function returns a callable `g(x, t, c_adv, nu)` that evaluates the Fourier series solution at a given position `x`
    and time `t`, for specified advection speed `c_adv` and diffusion coefficient `nu`. The approximation is based on the 
    provided Fourier coefficients.

    Args:
        c: Constant Fourier coefficient.
        a: List of cosine coefficients a_k for k = 1, ..., N.
        b: List of sine coefficients b_k for k = 1, ..., N.
        d: Length of the interval [0, d] over which the Fourier series is defined.

    Returns:
        g: A function g(x, t, c_adv, nu) representing the Fourier approximation to the solution of the advection-diffusion
           equation at position x and time t. x can be an array of positions. 

    Notes:
        - The solution is constructed as:
              u(x, t) = c + sum_{k=1}^{N-1} exp(-nu * n_k² * t) * [a_k * cos(n_k * (x - c_adv * t)) + b_k * sin(n_k * (x - c_adv * t))]
          where n_k = 2πk / d.
        - This representation assumes that the Fourier coefficients (c, a, b) describe the initial condition u(x, 0).
    """

    def g(x, t, c_adv, nu):
        u = c
        for k in range(len(a)):
            n = 2 * np.pi * (k + 1) / d
            u += np.exp(-nu * (n**2) * t) * (a[k] * np.cos(n * (x - c_adv * t)) + b[k] * np.sin(n * (x - c_adv * t))) 
        return u
    return g

def Fourier_coef_2d(f: Callable, eps: float, d: float, N: int = None) -> Tuple[float, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """ Return the Fourier coefficients for a function f defined on the interval [0, d]×[0, d]

    This function calculates the Fourier series coefficients up to the N-th Fourier mode or,
    if `N` is not provided, it adaptively determines the number of terms required to achieve a
    given approximation error `eps`.

    Args:
        f: A function defined on [0, d]×[0, d].
        eps: Target maximum absolute error in adaptive mode
        d: Domain size in both x and y directions.
        N: Optional integer specifying the number of Fourier terms to compute. If omitted, the function adaptively determines
           the number of terms required to achieve the desired error `eps`.
    
    Outputs:
        The number of fourier terms needed to converge.       

    Returns:
        c: The constant Fourier coefficient.
        A: Cosine coefficients.
        B: Sine coefficients.
        modes: List of (n1, n2) mode indices.
    """

    omega = 2 * np.pi / d

    # Compute the zeroth coefficient.
    c = dblquad(lambda y, x: f(x, y), 0.0, d, lambda _: 0.0, lambda _: d)[0] / (d * d)

    def compute_coef(n1: int, n2: int) -> Tuple[float, float]:
        A_n = dblquad(lambda y, x: f(x, y) * np.cos(omega * (n1 * x + n2 * y)), 0.0, d, lambda _: 0.0, lambda _: d)[0] * (2.0 / (d * d))
        B_n = dblquad(lambda y, x: f(x, y) * np.sin(omega * (n1 * x + n2 * y)), 0.0, d, lambda _: 0.0, lambda _: d)[0] * (2.0 / (d * d))
        return A_n, B_n

    # === Non-adaptive mode: fixed number of coefficients ===
    if N:
        A_list, B_list, modes = [], [], []
        for n1 in range(0, N + 1):
            for n2 in range(-N, N + 1):
                if n1 == 0 and n2 <= 0:
                    continue
                A_n, B_n = compute_coef(n1, n2)
                A_list.append(A_n)
                B_list.append(B_n)
                modes.append((n1, n2))
        return c, np.array(A_list), np.array(B_list), modes

    # === Adaptive mode: increase N until error < eps ===
    x_vals = np.linspace(0.0, d, 1000, endpoint=False)
    y_vals = np.linspace(0.0, d, 1000, endpoint=False)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
    f_vals = f(X, Y)

    max_iter = 500
    for N in range(1, max_iter + 1):
        approx_vals = np.full_like(f_vals, fill_value=c, dtype=float)
        modes = []
        A_list, B_list = [], []

        for n1 in range(0, N + 1):
            for n2 in range(-N, N + 1):
                if n1 == 0 and n2 <= 0:
                    continue
                A_n, B_n = compute_coef(n1, n2)
                A_list.append(A_n)
                B_list.append(B_n)
                modes.append((n1, n2))
                phase = omega * (n1 * X + n2 * Y)
                approx_vals += A_n * np.cos(phase) + B_n * np.sin(phase)

        if np.max(np.abs(f_vals - approx_vals)) < eps:
            print(f"-- FOURIER --\nNumber of Fourier coefficients needed: {N}\n")
            return c, np.array(A_list), np.array(B_list), modes

    raise RuntimeError(f"2D Fourier series did not reach error < {eps} within N={max_iter}.")

def Fourier_approx_2d(c: float, A: np.ndarray, B: np.ndarray, modes: List[Tuple[int, int]], d: float) -> Callable:
    """
    Constructs a time-evolving Fourier approximation function for a solution to the 2-dimensional advection-diffusion equation.

    This function returns a callable `g(x, y, t, nu, v1, v2)` that evaluates the Fourier series solution at a given position `(x,y)`
    and time `t`, for specified advection speeds `v1`, `v2` and diffusion coefficient `nu`. The approximation is based on the 
    provided Fourier coefficients.

    Args:
        c: Constant Fourier coefficient.
        A: Cosine coefficients.
        B: Sine coefficients.
        modes: List of (n1, n2) mode indices.
        d: Domain size.

    Returns:
        g: Function that evaluates the Fourier approximation at given points.

    Notes:
        - The solution is constructed as:
              u(X, Y, t) = c + Σ exp(-nu * ω² * (n1² + n2²) * t) * [A_n cos(φ) + B_n sin(φ)]
          where φ = ω * ((n1 * X + n2 * Y) - (n1 * v1 + n2 * v2) * t)
                ω = 2π / d

    """
    omega = 2 * np.pi / d
    def g(X, Y, t, nu, v1, v2):
        u = np.full_like(X, fill_value=c, dtype=float)
        for (n1, n2), A_n, B_n in zip(modes, A, B):
            exp_factor = np.exp(-nu * (omega ** 2) * (n1 ** 2 + n2 ** 2) * t)
            phase = omega * ((n1 * X + n2 * Y) - (n1 * v1 + n2 * v2) * t)
            u += exp_factor * (A_n * np.cos(phase) + B_n * np.sin(phase))
        return u
    return g