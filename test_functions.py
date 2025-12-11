import numpy as np
from Adv_Diff import Simulation_QC, Simulation_QC_2D
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.interpolate import interp1d
from typing import Callable

""" This file contains several test functions to be used as initial conditions for the advection-diffusion equation.
    It also contains functions plot_simulations, plot_simulations_2d, and run_examples which run and plot simulations using these test functions. 
    These functions run simulations for different sets of orders and number of qubits, allowing for comparison of results.
"""

# Test functions for initial conditions

def gaussian(x, c = 5/3, scale = 10):
    return np.exp(-scale * (x - c) ** 2)

def gaussian_2d(x, y, c = (5/3, 2), scale = 7):
    return np.exp(-scale * ((x - c[0]) ** 2 + (y - c[1]) ** 2))

def sine_squared_2d(x, y, c = 0.5):
    return np.sin(np.pi * (c * x + y)) ** 2

def sine_sum(x, k1 = 3, k2 = 11):
    return 0.5 * np.sin((k1 / 2) * np.pi * x) + 0.5 * np.sin((k2 / 2) * np.pi * x) + 1
    
def wave_pack(x, k = 17):
    return np.exp(-5 * (x - 2) ** 2) * 0.5 * np.cos((k / 2) * np.pi * (x - 2)) + 0.6

def b(x):
  # helper function used to define Bump(x)
    x = np.asarray(x)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    mask_y = x > 0
    mask_z = x < 1
    y[mask_y] = np.exp(-1 / x[mask_y])
    z[mask_z] = np.exp(1 / (x[mask_z] - 1))
    return y / (y + z)

def bump(x, I = [1, 1.5, 2.5, 3]):
    """
    I = (a, b, c, d), a < b < c < d 
    Bump(x) = 1 for x in [b, c]
    Bump(x) = 0 for x not in (a, d)
    Bump(x) in (0, 1) for x in (a, b) union (c, d)
    """
    
    return b((x - I[0]) / (I[1] - I[0])) * b((I[3] - x) / (I[3] - I[2]))
    
def rec(x):
    x = np.asarray(x)
    y = np.zeros_like(x)
    y[x < 2] = 1
    return y


def plot_simulations(
        num_qubits_order: list[tuple] = [(6, 2), (6, 4), (6, 6)], 
        time: float = 0.5, 
        adv_speed: float = 1, 
        diff_coef: float = 0.1, 
        domain_length: float = 4, 
        init_f: Callable = lambda x: np.exp(-10 * (x - 4 / 3) ** 2), 
        shots: int = 10 ** 6, 
        tolerance: float = 1e-8, 
        sim_type: str = "both"
        ):
    
    """ Run and visualize 1D advection–diffusion simulations for multiple (number of spatial qubits, order) configurations.
        
        Plots the initial condition and the final states obtained from quantum simulations (measurements and/or statevector) alongside the exact solution.
        All solutions are interpolated onto a common grid for comparison.
        A summary table reporting max error, qubit counts, success rate, and gate complexity is printed and included in the figure.

    Args:
        num_qubits_order: List of (num_spatial_qubits, order) pairs. A separate simulation is run for each tuple.
        time: Final simulation time.
        adv_speed: Advection speed.
        diff_coef: Diffusion coefficient.
        domain_length: Length of the spatial domain.
        init_f: Initial condition function f(x).
        shots: Number of measurement shots (for sim_type = "meas").
        tolerance: Tolerance passed to the quantum simulation backend.
        sim_type: Specifies whether to plot statevector results ('sv'), measurement results ('meas), or both ('both').
    """

    if sim_type not in ["sv", "meas", "both"]:  
        raise ValueError("sim_type must be 'sv', 'meas', or 'both'.")
    
    results = []
    orders = [order for (_, order) in num_qubits_order]
    num_qubits = [num_qubits for (num_qubits, _) in num_qubits_order]

    # Run simulations
    for i in range(len(orders)):
        x, init_fx, meas_results, fourier_results, statevec_results, num_qubits_total, max_errors, success_rates, complexities = Simulation_QC.simulate_adv_diff(
            num_qubits[i], time, adv_speed, diff_coef, domain_length, init_f, shots, True, orders[i], tolerance, sim_type, True, False)

        meas_result = meas_results[0] if (sim_type != "sv") else None
        statevec_result = statevec_results[0] if (sim_type != "meas") else None
        fourier_result = fourier_results[0]

        results.append((x, init_fx, meas_result, fourier_result, statevec_result, num_qubits_total, max_errors, success_rates, complexities))

    # Define a common grid (based on the finest resolution)
    max_num_qubits = max(num_qubits)
    xmin = max([x.min() for (x, _, _, _, _, _, _, _, _) in results])
    xmax = min([x.max() for (x, _, _, _, _, _, _, _, _) in results])
    x_common = np.linspace(xmin, xmax, 2 ** max_num_qubits)

    # Interpolate results to common grid
    interp_results = []
    for (x, init_fx, meas_result, fourier_result, statevec_result, num_qubits_total, max_errors, success_rates, complexities) in results:
        init_fx_interp = interp1d(x, init_fx, kind="linear")(x_common)
        fourier_result_interp = interp1d(x, fourier_result, kind="linear")(x_common)
        meas_result_interp = interp1d(x, meas_result, kind="linear")(x_common) if meas_result is not None else None
        statevec_result_interp = interp1d(x, statevec_result, kind="linear")(x_common) if statevec_result is not None else None
        interp_results.append((init_fx_interp, fourier_result_interp, meas_result_interp, statevec_result_interp, max_errors, success_rates, complexities))

    # Create summary table
    table = [
        [f"order {orders[i]}", num_qubits[i], num_qubits_total[0], f"{max_errors[0][0]:.3e}", f"{success_rates[0]:.4f}", complexities[0][0], complexities[0][1]]
        for i, (_, _, _, _, _, num_qubits_total, max_errors, success_rates, complexities) in enumerate(results)]
    print(f"\n-- SUMMARY --")
    error_type = "meas." if sim_type == "meas" else "sv"
    headers=['', 'spatial qubits', 'total qubits', f'error ({error_type})', 'success rate', '1-qubit gates', 'CNOT gates']
    print(tabulate(table, headers=headers, tablefmt="simple_grid", colalign=("center",)*len(headers)))
    
    # Plot
    _, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True, gridspec_kw={'height_ratios': [1, 1, 0.2*len(table)]})

    # Plot initial condition
    axes[0].plot(x_common, interp_results[0][0], lw=1, color="b")
    axes[0].set_title(rf'Initial Condition', fontsize=14)

    # Plot all orders together
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(len(orders)):
        init_fx_interp, fourier_result_interp, meas_result_interp, statevec_result_interp, _, _, _ = interp_results[i]
        if i == 0:
            axes[1].plot(x_common, fourier_result_interp, color="b", lw=1, label=f'exact solution (Fourier)')
        if sim_type != "sv":
            axes[1].plot(x_common, meas_result_interp, color = colors[i], lw=1, label=f'measurements, order {orders[i]} ({num_qubits[i]} spatial qubits)')
        if sim_type != "meas":
            axes[1].plot(x_common, statevec_result_interp.real, '--', color=colors[i], lw=1, label=f'statevector, order {orders[i]} ({num_qubits[i]} spatial qubits) ')

    time_str = str(time)
    if len(time_str) > 4: time_str = time_str[:4]
    axes[1].set_title(rf'Results at Time $T = {time_str}$', fontsize=14)
    axes[1].legend()

    # Determine min/max y-values across both plots
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

    # Apply the same y-limits to both axes
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

    # Adding the table to the plot
    headers = ['', 'spatial qubits', 'total qubits', f'error ({error_type})', 'success rate', '1-qubit gates', 'CNOT gates']
    axes[2].axis('off')
    tbl = axes[2].table(cellText = table, colLabels = headers, loc = 'center')
    tbl.scale(1,2)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    _ = [cell.set_text_props(ha='center', va='center') for cell in tbl.get_celld().values()]
    axes[2].set_title('Data Table', fontsize=14)

    # Set super title
    if diff_coef == 0:
        super_title = rf"Advection Simulation with Advection Speed $c =$ {adv_speed:.3g}"
    elif adv_speed == 0:
        super_title = rf"Diffusion Simulation with Diffusion Coefficient $\nu =$ {diff_coef:.3g}"
    else:
        super_title = rf"Advection-Diffusion Simulation with Advection Speed $c =$ {adv_speed:.3g} and Diffusion Coefficient $\nu =$ {diff_coef:.3g}"
    plt.suptitle(super_title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_simulations_2d(
        num_qubits_order: list[tuple] = [(6,2),(6,4)], 
        time: float = 0.5, 
        adv_speed_x: float = 1.0, 
        adv_speed_y: float = 1.0, 
        diff_coef: float = 0.1, 
        domain_length: float = 4.0,
        init_f: Callable = lambda X, Y: np.sin(np.pi * (0.5 * X + Y)) ** 2,
        shots: int = 10 ** 7,
        tolerance: float = 1e-6, 
        sim_type: str = "sv"
        ):
    
    """Run and visualize 2D advection–diffusion simulations for two (spatial qubits per dimension, order) configurations.

    Plots the initial condition and the final states obtained from quantum simulations (measurements or statevector) alongside the exact solution.
    A summary table reporting max error, qubit counts, success rate, and gate complexity is printed and included in the figure.

    Args:
    num_qubits_order: Exactly two entries (num_qubits, order), specifying spatial qubits per dimension and method order.
    time: Final simulation time.
    adv_speed_x, adv_speed_y: Advection velocities in the x and y directions.
    diff_coef: Diffusion coefficient.
    domain_length: Length of each spatial dimension.
    init_f: Initial condition function f(x, y).
    shots: Number of measurement shots (for sim_type = "meas").
    tolerance: Tolerance passed to the quantum simulation backend.
    sim_type : Whether to visualize statevector ('sv') or measurement ('meas') results.
    """

    if len(num_qubits_order) != 2:
        raise ValueError("This function only supports two simulations (two entries in num_qubits_order).")

    if sim_type not in ["sv", "meas"]:
        raise ValueError("sim_type must be 'sv' or 'meas'")

    results = []
    orders = [order for (_, order) in num_qubits_order]
    num_qubits_list = [num_qubits for (num_qubits, _) in num_qubits_order]

    # Run simulations
    for i in range(2):
        x, y, init_fxy, meas_result, fourier_result, statevec_result, num_qubits_total, max_errors, success_rates, complexities = Simulation_QC_2D.simulate_adv_diff_2d(
            num_qubits_list[i], time, adv_speed_x, adv_speed_y, diff_coef, domain_length, init_f, shots, True, orders[i], tolerance, sim_type, True, False)
        results.append((x, y, init_fxy, meas_result, fourier_result, statevec_result, num_qubits_total, max_errors, success_rates, complexities))

    # Create summary table
    table = [
        [f"order {orders[i]}", num_qubits_list[i] * 2, num_qubits_total, f"{max_errors[0]:.3e}", f"{success_rates:.4f}", complexities[0], complexities[1]]
        for i, (_, _, _, _, _, _, num_qubits_total, max_errors, success_rates, complexities) in enumerate(results)
    ]
    print(f"\n-- SUMMARY --")
    headers=['', f'spatial qubits', 'total qubits', f'error', 'success rate', '1-qubit gates', 'CNOT gates']
    print(tabulate(table, headers=headers, tablefmt="simple_grid", colalign=("center",)*len(headers)))

    # Set up figure and axes
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 1])
    ax_init = fig.add_subplot(gs[0, 0], projection='3d')  # initial
    ax_exact = fig.add_subplot(gs[0, 1], projection='3d')  # exact
    ax_q1 = fig.add_subplot(gs[1, 0], projection='3d')  # quantum plot first given order
    ax_q2 = fig.add_subplot(gs[1, 1], projection='3d')  # quantum plot second given order
    ax_table = fig.add_subplot(gs[2, :])  # table

    # Determine global z-limits
    z_values = []
    z_values.append(results[0][2])      # Initial
    z_values.append(results[0][4])      # Exact
    for i in range(2):
        z_values.append(results[i][5].real if sim_type=="sv" else results[i][3])
    z_min = np.min([np.min(z) for z in z_values if z is not None])
    z_max = np.max([np.max(z) for z in z_values if z is not None])

    x = results[0][0]
    y = results[0][1]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Plot initial condition
    ax_init.plot_surface(X, Y, results[0][2], cmap='viridis')
    ax_init.set_title('Initial Condition', fontsize=14)
    ax_init.set_zlim(z_min, z_max)
    ax_init.set_xlabel('x')
    ax_init.set_ylabel('y')

    # Plot exact solution
    ax_exact.plot_surface(X, Y, results[0][4], cmap='viridis')
    time_str = str(time)
    if len(time_str) > 4: time_str = time_str[:4]
    ax_exact.set_title(f'Exact Solution at Time $T = {time_str}$', fontsize=14)
    ax_exact.set_zlim(z_min, z_max)
    ax_exact.set_xlabel('x')
    ax_exact.set_ylabel('y')

    # Plot quantum results for each order
    for i, ax in zip(range(2), [ax_q1, ax_q2]):
        x = results[i][0]
        y = results[i][1]
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = results[i][5].real if sim_type=="sv" else results[i][3]
        ax.plot_surface(X, Y, Z, cmap='plasma')
        if sim_type == "sv":
            ax.set_title(rf'Statevector at Time $T = {time_str}$ (Order {orders[i]})', fontsize=14)
        else:
            ax.set_title(rf'Measurements at Time $T = {time_str}$ (Order {orders[i]})', fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlim(z_min, z_max)

    # Table
    ax_table.axis('off')
    tbl = ax_table.table(cellText=table,
                         colLabels=['', f'spatial qubits', 'total qubits', f'error', 'success rate', '1-qubit gates', 'CNOT gates'],
                         loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.6)
    _ = [cell.set_text_props(ha='center', va='center') for cell in tbl.get_celld().values()]
    ax_table.set_title('Data Table', fontsize=14)

    # Set super title
    if diff_coef == 0:
        super_title = rf"2D Advection Simulation with Advection Speeds $c_x = {adv_speed_x:.3g}, \; c_y = {adv_speed_y:.3g}$"
    elif adv_speed_x == 0 and adv_speed_y == 0:
        super_title = rf"2D Diffusion Simulation with Diffusion Coefficient $\nu={diff_coef:.3g}$"
    else:
        super_title = (
        rf"2D Advection–Diffusion Simulation with Diffusion Coefficient $\nu={diff_coef:.3g}$"
        f"\nand Advection Speeds $c_x = {adv_speed_x:.3g}, c_y = {adv_speed_y:.3g}$"
    )
        #super_title = rf"2D Advection–Diffusion Simulation with Diffusion Coefficient $\nu={diff_coef}$ and Advection Speeds $c_x = {adv_speed_x}, \; c_y = {adv_speed_y}$"
    fig.suptitle(super_title, fontsize=16)
    plt.tight_layout()
    plt.show()


def run_examples(examples = [gaussian, sine_sum, wave_pack, bump, rec, gaussian_2d, sine_squared_2d], sim_type="sv", shots=10**6):
    """ Run and plot a predefined collection of example simulations.

    This function dispatches to either `plot_simulations` (1D) or `plot_simulations_2d` (2D) depending on the example.
    Each example uses preset parameters for number of qubits, order, time, advection speed(s), diffusion coefficient and domain length.

    Args:
    examples: List of initial-condition functions to simulate. Each entry must be one of:
              {gaussian, sine_sum, wave_pack, bump, rec, gaussian_2d, sine_squared_2d}.
    sim_type: Simulation type passed to the plotting routines. 
              Note that for 2D examples, sim_type="both" is not supported. instead, both types will be run sequentially.
    shots: Number of measurement shots used for example runs if sim_type = "meas".
    """

    for example in examples:

        if example not in [gaussian, sine_sum, wave_pack, bump, rec, gaussian_2d, sine_squared_2d]:
            raise ValueError(f"Example {example} not recognized.")
        
        if example == gaussian:
            # plot_simulations(num_qubits_order=[(7,4),(6,6)], time=2, adv_speed=0, diff_coef=0.02, init_f=Gaussian, shots = shots, sim_type=sim_type)
            plot_simulations(num_qubits_order=[(8,2),(6,6),(9,2),(7,6)], time=4, adv_speed=1, diff_coef=0, domain_length=4, init_f=gaussian, shots = shots, tolerance=1e-8, sim_type=sim_type)
        
        elif example == sine_sum:
            plot_simulations(num_qubits_order=[(8,6),(6,14)], time=0.5, adv_speed=0, diff_coef=0.1, domain_length=4, init_f=sine_sum, shots=shots, tolerance=1e-8, sim_type = sim_type)
        
        elif example == wave_pack:
            plot_simulations(num_qubits_order=[(10,6),(8,14)], time=1, adv_speed=1, diff_coef=0, domain_length=4, init_f=wave_pack, shots=shots, tolerance=1e-8, sim_type = sim_type)
        
        elif example == bump:
            plot_simulations(num_qubits_order=[(9,2),(7,6)], time=0.6, adv_speed=2, diff_coef=0, domain_length=4, init_f=bump, shots=shots, tolerance=1e-8, sim_type = sim_type)
        
        elif example == rec:
            plot_simulations(num_qubits_order=[(8,2),(7,6)], time=1, adv_speed=1, diff_coef=0.02, domain_length=4, init_f=rec, shots=shots, tolerance=1e-8, sim_type=sim_type)
        
        elif example == gaussian_2d:
            if sim_type == 'both':
                sim_type = 'sv'
                plot_simulations_2d(num_qubits_order=[(7,2),(6,4)], time=0.5, adv_speed_x=0, adv_speed_y=0, diff_coef=0.05, init_f=gaussian_2d, shots=shots, tolerance=1e-6, sim_type="meas")
            plot_simulations_2d(num_qubits_order=[(7,2),(6,4)], time=0.5, adv_speed_x=0, adv_speed_y=0, diff_coef=0.05, init_f=gaussian_2d, shots=shots, tolerance=1e-6, sim_type=sim_type)
        
        elif example == sine_squared_2d:
            if sim_type == 'both':
                sim_type = 'sv'
                plot_simulations_2d(num_qubits_order=[(6,2),(5,4)], time=0.5, adv_speed_x=1, adv_speed_y=1, diff_coef=0.1, init_f=sine_squared_2d, shots=shots, tolerance=1e-6, sim_type='meas')
            plot_simulations_2d(num_qubits_order=[(6,2),(5,4)], time=0.5, adv_speed_x=1, adv_speed_y=1, diff_coef=0.1, init_f=sine_squared_2d, shots=shots, tolerance=1e-6, sim_type=sim_type)

run_examples()

