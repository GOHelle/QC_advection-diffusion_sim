import numpy as np
from Adv_Diff import Simulation_QC, Simulation_QC_2D
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.interpolate import interp1d
from typing import Callable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from scipy.interpolate import RegularGridInterpolator
from matplotlib.lines import Line2D

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

def mixed_wave_2d(x, y):
    return np.exp(-7 * (x - 2) ** 2) * (1 + np.sin(5 * np.pi * y / 2))

def sine_sum(x, k1 = 3, k2 = 11):
    return 0.5 * np.sin((k1 / 2) * np.pi * x) + 0.5 * np.sin((k2 / 2) * np.pi * x) + 1
    
def wave_pack(x, k = 17):
    return np.exp(-5 * (x - 2) ** 2) * 0.5 * np.cos((k / 2) * np.pi * (x - 2)) + 0.6
    
def rec(x):
    x = np.asarray(x)
    y = np.zeros_like(x)
    y[x < 2] = 1
    return y

def zoom_window(
    ax,
    x,
    y_series,
    window_width_frac=0.04,
    window_height_frac=0.2,
):
    """Determine the coordinates for a zoom window that focuses on the region where the curves differ most.
    
    Args:
        ax: The matplotlib Axes object of the main plot.
        x: The common x-grid for all curves.
        y_series: List of y-value arrays corresponding to the different curves.
        window_width_frac: Desired width of the zoom window as a fraction of the x-range.
        window_height_frac: Desired height of the zoom window as a fraction of the y-range.
    """

    y_stack = np.vstack([np.asarray(y) for y in y_series])
    spread = y_stack.max(axis=0) - y_stack.min(axis=0)

    # pick x-center where curves differ most
    idx = int(np.argmax(spread))
    x0 = x[idx]
    x_range = x.max() - x.min()
    half_w = 0.5 * window_width_frac * x_range
    x1, x2 = x0 - half_w, x0 + half_w
    x1 = max(x1, x.min())
    x2 = min(x2, x.max())

    # compute y-limits in that window
    mask = (x >= x1) & (x <= x2)
    y_win = y_stack[:, mask]
    y_low = float(np.min(y_win))
    y_high = float(np.max(y_win))
    y_center = 0.5 * (y_low + y_high)

    # full y-range of original plot
    full_ymin, full_ymax = ax.get_ylim()
    full_y_range = full_ymax - full_ymin

    half_h = 0.5 * window_height_frac * full_y_range
    y1 = y_center - half_h
    y2 = y_center + half_h

    return (x1, x2), (y1, y2)

def plot_simulations(
        num_qubits_order: list[tuple] = [(6, 2), (6, 4), (6, 6)], 
        time: float = 0.5, 
        adv_speed: float = 1, 
        diff_coef: float = 0.1, 
        domain_length: float = 4, 
        init_f: Callable = lambda x: np.exp(-10 * (x - 4 / 3) ** 2), 
        shots: int = 10 ** 6, 
        tolerance: float = 1e-8, 
        sim_type: str = "both",
        window_position = None,  # "separate" OR "upper right", "lower left", etc.
        window_width_frac=0.04,
        window_height_frac=0.2,
        legend_loc = "upper left",
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
        window_position: If not None, specifies the position of a zoomed-in view of the plot. if "separate", a separate subplot is created for the zoomed view. 
                         Otherwise, should be a valid position string (e.g. "upper right").
        window_width_frac: Width of the zoom window as a fraction of the x-range.
        window_height_frac: Height of the zoom window as a fraction of the y-range.
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
    
    if window_position == "separate":
        _, axes = plt.subplots(3, 1, figsize=(6, 9), constrained_layout=True)      
    else:
        _, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)      
 
    # Plot initial condition
    axes[0].plot(x_common, interp_results[0][0], lw=1, color="b")
    axes[0].set_title(rf'Initial Condition', fontsize=13)

    # Plot all orders together
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(len(orders)):
        init_fx_interp, fourier_result_interp, meas_result_interp, statevec_result_interp, _, _, _ = interp_results[i]
        if i == 0:
            axes[1].plot(x_common, fourier_result_interp, color="b", lw=1, label=f'exact')
        if sim_type != "sv":
            axes[1].plot(x_common, meas_result_interp, color = colors[i], lw=1, label=f'ord{orders[i]}, spq={num_qubits[i]}')
        if sim_type != "meas":
            axes[1].plot(x_common, statevec_result_interp.real, '--', color=colors[i], lw=1, label=f'ord{orders[i]}, spq={num_qubits[i]}')

    time_str = str(time)
    if len(time_str) > 4: time_str = time_str[:4]
    axes[1].set_title(rf'Results at $T = {time_str}$', fontsize=13)
    axes[1].legend(loc=legend_loc, fontsize=9, labelspacing=0.25,)

    # Build list of curves for the zoom to focus on 
    y_series_for_zoom = []
    y_series_for_zoom.append(interp_results[0][1])  # fourier_result_interp
    for i in range(len(orders)):
        _, _, meas_result_interp, statevec_result_interp, _, _, _ = interp_results[i]
        if sim_type != "sv" and meas_result_interp is not None:
            y_series_for_zoom.append(meas_result_interp)
        if sim_type != "meas" and statevec_result_interp is not None:
            y_series_for_zoom.append(statevec_result_interp.real)

    ax = axes[1]
    (x1, x2), (y1, y2) = zoom_window(ax, x_common, y_series_for_zoom, window_width_frac=window_width_frac, window_height_frac=window_height_frac)     # zoom window widths and height as fraction of y-range

    if window_position:
        if window_position == "separate":
            # add separate zoomed-in plot as third subplot 
            for line in ax.lines:
                axes[2].plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    color=line.get_color(),
                    marker=line.get_marker(),
                    label=line.get_label(),
                )
            axes[2].set_xlim(x1, x2)
            axes[2].set_ylim(y1, y2)
            axes[2].set_title('Zoomed View', fontsize=13)
            axes[2].legend(loc='upper left', fontsize=9, labelspacing=0.25,)

        else:
            # create inset
            axins = inset_axes(ax, width="28%", height="40%", loc=window_position, borderpad=1.0)
            axins.patch.set_alpha(0.75)
            # split window position string by space to get e.g upper, right
            x_label_pos, y_label_pos = window_position.split()
            if x_label_pos == "upper":
                axins.xaxis.tick_bottom()
                axins.xaxis.set_label_position("bottom")
            else:
                axins.xaxis.tick_top()
                axins.xaxis.set_label_position("top")
            if y_label_pos == "right":
                axins.yaxis.tick_left()
                axins.yaxis.set_label_position("left")
            else:
                axins.yaxis.tick_right()
                axins.yaxis.set_label_position("right")

            for line in ax.lines:
                axins.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    color=line.get_color(),
                    marker=line.get_marker(),
                )

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

        # draw rectangle on main plot
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="0.3", lw=1)
        ax.add_patch(rect)

    # Determine min/max y-values across both plots
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

    # Apply the same y-limits to both axes
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

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
        sim_type: str = "sv",
        zoom_window_frac=0.25,
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
    zoom_window_frac: ratio between the size zoomed window plot and the main plot
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
        # complexities = [0, 0, 0, 0]
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
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 3)                 # Changed 2,2 to 2,3     can control width and height ratios 
    ax_init = fig.add_subplot(gs[0, 0], projection='3d')  # initial
    ax_exact = fig.add_subplot(gs[1, 0], projection='3d')  # exact
    ax_q1 = fig.add_subplot(gs[0, 1], projection='3d')  # quantum plot first given order
    ax_q2 = fig.add_subplot(gs[1, 1], projection='3d')  # quantum plot second given order
    ax_compare = fig.add_subplot(gs[0,2])               # contour plot 
    ax_compare_zoom = fig.add_subplot(gs[1,2])          # zoomed contour plot

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
    ax_init.set_title('Initial Condition', fontsize=13)
    ax_init.set_zlim(z_min, z_max)
    ax_init.set_xlabel('x')
    ax_init.set_ylabel('y')

    # Plot exact solution
    ax_exact.plot_surface(X, Y, results[0][4], cmap='viridis')
    time_str = str(time)
    if len(time_str) > 4: time_str = time_str[:4]
    ax_exact.set_title(f'Time $T = {time_str}$ (exact)', fontsize=13)
    ax_exact.set_zlim(z_min, z_max)
    ax_exact.set_xlabel('x')
    ax_exact.set_ylabel('y')

    # Plot quantum results for each order
    for i, ax in zip(range(2), [ax_q1, ax_q2]):
        x = results[i][0]
        y = results[i][1]
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = results[i][5].real if sim_type=="sv" else results[i][3]
        ax.plot_surface(X, Y, Z, cmap='viridis')                            # Changed from plasma 
        ax.set_title(rf'Time $T = {time_str}$ (ord{orders[i]})', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlim(z_min, z_max)

    
    def interp_to_grid(x_src, y_src, Z_src, x_tgt, y_tgt):
        """Interpolate Z(x_src,y_src) onto target tensor grid (x_tgt,y_tgt)."""
        interp = RegularGridInterpolator((x_src, y_src), Z_src, bounds_error=False, fill_value=np.nan)
        Xg, Yg = np.meshgrid(x_tgt, y_tgt, indexing="ij")
        pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
        Zg = interp(pts).reshape(Xg.shape)
        return Xg, Yg, Zg

    # pick a common grid (use the finest resolution)
    x0, y0 = results[0][0], results[0][1]
    x1, y1 = results[1][0], results[1][1]
    x_common = x0 if len(x0) >= len(x1) else x1
    y_common = y0 if len(y0) >= len(y1) else y1
    X, Y = np.meshgrid(x_common, y_common, indexing="ij")

    # pull surfaces
    Z_exact0 = results[0][4]  # exact for run 0
    Z_0 = results[0][5].real if sim_type == "sv" else results[0][3]
    Z_1 = results[1][5].real if sim_type == "sv" else results[1][3]

    # interpolate all onto (x_common, y_common)
    _, _, Z_exact = interp_to_grid(x0, y0, Z_exact0, x_common, y_common)
    _, _, Z0g     = interp_to_grid(x0, y0, Z_0,     x_common, y_common)
    _, _, Z1g     = interp_to_grid(x1, y1, Z_1,     x_common, y_common)

    # levels based on common-grid data
    z_min_c = np.nanmin([Z_exact, Z0g, Z1g])
    z_max_c = np.nanmax([Z_exact, Z0g, Z1g])
    levels = np.linspace(z_min_c, z_max_c, 6)

    def draw_contours(ax_):
        ax_.contour(X, Y, Z_exact, levels=levels, colors="black", linewidths=1)
        ax_.contour(X, Y, Z0g,     levels=levels, colors="red",   linewidths=1, linestyles="--")
        ax_.contour(X, Y, Z1g,     levels=levels, colors="blue",  linewidths=1, linestyles=":")
        ax_.set_xlabel("x")
        ax_.set_ylabel("y")

    draw_contours(ax_compare)
    ax_compare.set_title("Contour Comparison", fontsize=13)

    # legend via proxy artists since contour doesn't support labels
    handles = [
        Line2D([0],[0], color="black", lw=1, linestyle="-",  label="exact"),
        Line2D([0],[0], color="red",   lw=1, linestyle="--", label=f"ord{orders[0]}"),
        Line2D([0],[0], color="blue",  lw=1, linestyle=":",  label=f"ord{orders[1]}"),
    ]
    ax_compare.legend(handles=handles, loc="lower left", fontsize=9)
    ax_compare_zoom.legend(handles=handles, loc="upper right", fontsize=9)

    # --- zoom contour plot ---
    draw_contours(ax_compare_zoom)
    ax_compare_zoom.set_title("Zoomed Contours", fontsize=13)

    # auto center zoom around maximum absolute error
    err = np.abs(Z0g - Z_exact)
    idx = np.nanargmax(err)
    i0, j0 = np.unravel_index(idx, err.shape)
    xc = x_common[i0]
    yc = y_common[j0]

    x_range = x_common.max() - x_common.min()
    y_range = y_common.max() - y_common.min()
    half_wx = 0.5 * zoom_window_frac * x_range
    half_wy = 0.5 * zoom_window_frac * y_range

    zx1 = max(x_common.min(), xc - half_wx)
    zx2 = min(x_common.max(), xc + half_wx)
    zy1 = max(y_common.min(), yc - half_wy)
    zy2 = min(y_common.max(), yc + half_wy)

    ax_compare_zoom.set_xlim(zx1, zx2)
    ax_compare_zoom.set_ylim(zy1, zy2)

    # draw rectangle on the full contour plot to show zoom region
    rect = patches.Rectangle(
        (zx1, zy1), zx2 - zx1, zy2 - zy1,
        fill=False, ec="0.3", lw=1
    )
    ax_compare.add_patch(rect)

    plt.tight_layout()               
    plt.show()


def run_examples(examples = [gaussian, sine_sum, wave_pack, rec, gaussian_2d, sine_squared_2d], sim_type="sv", shots=10**4):
    """ Run and plot a predefined collection of example simulations.

    This function dispatches to either `plot_simulations` (1D) or `plot_simulations_2d` (2D) depending on the example.
    Each example uses preset parameters for number of qubits, order, time, advection speed(s), diffusion coefficient and domain length.

    Args:
    examples: List of initial-condition functions to simulate. Each entry must be one of:
              {gaussian, sine_sum, wave_pack, rec, gaussian_2d, sine_squared_2d}.
    sim_type: Simulation type passed to the plotting routines. 
              Note that for 2D examples, sim_type="both" is not supported. instead, both types will be run sequentially.
    shots: Number of measurement shots used for example runs if sim_type = "meas".
    """

    for example in examples:

        if example not in [gaussian, sine_sum, wave_pack, rec, gaussian_2d, sine_squared_2d, mixed_wave_2d]:
            raise ValueError(f"Example {example} not recognized.")
        
        if example == gaussian:
            # plot_simulations(num_qubits_order=[(7,4),(6,6)], time=2, adv_speed=0, diff_coef=0.02, init_f=Gaussian, shots = shots, sim_type=sim_type)
            plot_simulations(num_qubits_order=[(9,2),(6,6),(8,2),(7,6)], time=4, adv_speed=1, diff_coef=0, domain_length=4, init_f=gaussian, shots = shots, tolerance=1e-6, sim_type=sim_type, window_position="upper right", window_width_frac=0.03, window_height_frac=0.06)
        
        elif example == sine_sum:
            plot_simulations(num_qubits_order=[(9,2),(8,4),(7,6)], time=0.3, adv_speed=0, diff_coef=0.02, domain_length=4, init_f=sine_sum, shots=shots, tolerance=1e-6, sim_type = sim_type, window_position="lower left", window_width_frac=0.02, window_height_frac=0.09, legend_loc="upper right")
        
        elif example == wave_pack:
            plot_simulations(num_qubits_order=[(9,6),(8,6), (6,14), (7,14)], time=1.5, adv_speed=1, diff_coef=10**(-3), domain_length=4, init_f=wave_pack, shots=shots, tolerance=1e-8, sim_type = sim_type, window_position="lower center", window_width_frac=0.07, window_height_frac=1)
        
        elif example == rec:
            plot_simulations(num_qubits_order=[(8,2),(7,6)], time=1, adv_speed=1, diff_coef=0.02, domain_length=4, init_f=rec, shots=shots, tolerance=1e-8, sim_type=sim_type, window_position="lower center", window_width_frac=0.04, window_height_frac=0.1)
        
        elif example == gaussian_2d:
            if sim_type == 'both':
                sim_type = 'sv'
                plot_simulations_2d(num_qubits_order=[(8,2),(6,6)], time=0.8, adv_speed_x=3/2, adv_speed_y=2/3, diff_coef=0.0, init_f=gaussian_2d, shots=shots, tolerance=1e-6, sim_type="meas", contour_plot=True, zoom_window_frac=0.1)
            plot_simulations_2d(num_qubits_order=[(8,2),(6,6)], time=0.8, adv_speed_x=3/2, adv_speed_y=2/3, diff_coef=0.0, init_f=gaussian_2d, shots=shots, tolerance=1e-6, sim_type=sim_type, contour_plot=True, zoom_window_frac=0.1)
        
        elif example == sine_squared_2d:
            if sim_type == 'both':
                sim_type = 'sv'
                plot_simulations_2d(num_qubits_order=[(6,2),(5,4)], time=0.5, adv_speed_x=1, adv_speed_y=1, diff_coef=0.1, init_f=sine_squared_2d, shots=shots, tolerance=1e-6, sim_type='meas')
            plot_simulations_2d(num_qubits_order=[(6,2),(5,4)], time=0.5, adv_speed_x=1, adv_speed_y=1, diff_coef=0.1, init_f=sine_squared_2d, shots=shots, tolerance=1e-6, sim_type=sim_type)

        elif example == mixed_wave_2d:   
            if sim_type == 'both':
                sim_type = 'sv'
                plot_simulations_2d(num_qubits_order=[(8,2),(6,6)], time=0.4, adv_speed_x=0.5, adv_speed_y=1, diff_coef=0.2, init_f=mixed_wave_2d, shots=shots, tolerance=1e-6, sim_type='meas', contour_plot=True, zoom_window_frac=0.04)
            plot_simulations_2d(num_qubits_order=[(8,2),(6,6)], time=0.4, adv_speed_x=0.5, adv_speed_y=1, diff_coef=0.2, init_f=mixed_wave_2d, shots=shots, tolerance=1e-6, sim_type=sim_type, contour_plot=True, zoom_window_frac=0.04)


# Run the 1d examples
run_examples(examples = [gaussian, sine_sum, wave_pack, rec], sim_type="sv")

# Run the 2d examples - these are much more time consuming. 
# run_examples(examples = [gaussian_2d, sine_squared_2d, mixed_wave_2d], sim_type="sv")

 



