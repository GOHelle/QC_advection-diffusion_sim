import numpy as np
from Adv_Diff import Simulation_QC, Simulation_QC_2D
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Patch

""" This file contains several test functions to be used as initial conditions for the advection-diffusion equation.
    It also contains functions plot_simulations, plot_simulations_2D, and run_examples which run and plot simulations using these test functions. 
    These functions run simulations for different sets of orders and number of qubits, allowing for comparison of results.
"""

# Test functions for initial conditions

def Gaussian(x,c = 5/3,scale = 10):
    return np.exp(-scale*(x-c)**2)

def Gaussian_2D(x,y,c = (5/3,2),scale = 7):
    return np.exp(-scale*((x-c[0])**2+(y-c[1])**2))

def Sine_squared_2D(x, y, c=0.5):
    return np.sin(np.pi * (c * x + y))**2

def Sine_sum(x,k1 = 3,k2 = 11):
    return 0.5*np.sin((k1/2)*np.pi*x)+0.5*np.sin((k2/2)*np.pi*x)+ 1
    
def Wave_pack(x, k = 17):
    return np.exp(-5*(x-2)**2)*0.5*np.cos((k/2)*np.pi*(x-2))+0.6

def b(x):
  # helper function used to define Bump(x)
    x = np.asarray(x)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    mask_y = x>0
    mask_z = x<1
    y[mask_y] = np.exp(-1 / x[mask_y])
    z[mask_z] = np.exp(1 / (x[mask_z] - 1))
    return y/(y+z)

def Bump(x, I = [1,1.5,2.5,3]):
    """
    I = (a,b,c,d), a<b<c<d 
    Bump(x) = 1 for x in [b,c]
    Bump(x) = 0 for x not in (a,d)
    Bump(x) in (0,1) for x in (a,b) union (c,d)
    """
    
    return b((x-I[0])/(I[1]-I[0]))*b((I[3]-x)/(I[3]-I[2]))
    
def rec(x):
    x = np.asarray(x)
    y = np.zeros_like(x)
    y[x<2] = 1
    return y


def plot_simulations(n_order=[(6,2),(6,4),(6,6)], T:float=0.5, c:float=1, nu:float=0.1, d:float=4, init_f = lambda x: np.exp(-10*(x-4/3)**2), shots:int=10**6, eps:float=10**(-8), sim_type:str="both"):
    """
    each tuple in n_order is of the form (n, order). For each tuple in this list, the simulation runs with n spatial qubits and the given order.
    Quantum measurement and/or statevector results are plotted according to the `sim_type` argument ("meas", "sv", or "both"). 
    A summary table of max error, succes rate and circuit complexity for each order is printed and displayed in the plot.
    """

    if sim_type not in ["sv", "meas", "both"]:  
        raise ValueError("sim_type must be 'sv', 'meas', or 'both'.")
    
    results = []
    orders = [order for (_, order) in n_order]
    n = [n for (n, _) in n_order]

    # Run simulations
    for i in range(len(orders)):
        x, f0, z_list, w_list, W_list, max_err, succ_rate, comp = Simulation_QC.Sim(n[i], T, c, nu, d, init_f, shots, True, orders[i], eps, sim_type, True, False)

        z = z_list[0] if (sim_type != "sv" and len(z_list) > 0) else None
        W = W_list[0] if (sim_type != "meas" and len(W_list) > 0) else None
        w = w_list[0]

        results.append((x, f0, z, w, W, max_err, succ_rate, comp))

    # Define a common grid (based on the finest resolution)
    n_max = max(n)
    xmin = max([x.min() for (x, _, _, _, _, _, _, _) in results])
    xmax = min([x.max() for (x, _, _, _, _, _, _, _) in results])
    x_common = np.linspace(xmin, xmax, 2**n_max)

    # Interpolate results to common grid
    interp_results = []
    for (x, f0, z, w, W, max_err, succ_rate, comp) in results:
        f_interp = interp1d(x, f0, kind="linear")(x_common)
        w_interp = interp1d(x, w, kind="linear")(x_common)
        z_interp = interp1d(x, z, kind="linear")(x_common) if z is not None else None
        W_interp = interp1d(x, W, kind="linear")(x_common) if W is not None else None
        interp_results.append((f_interp, z_interp, w_interp, W_interp, max_err, succ_rate, comp))

    # Create summary table
    table = [
        [f"Order {orders[i]} (n={n[i]})", f"{max_err[0][0]:.3e}", f"{succ_rate[0]:.4f}", comp[0][0], comp[0][1]]
        for i, (_, _, _, _, _, max_err, succ_rate, comp) in enumerate(results)]
    print(f"\n-- SUMMARY --")
    error_type = "m" if sim_type == "meas" else "sv"
    print(tabulate(table, headers=['', f'Error ({error_type})', 'Success rate', '1Q gates', 'CNOT gates'], tablefmt="simple_grid"))
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), constrained_layout=True)

    # Plot initial condition
    axes[0].plot(x_common, interp_results[0][0], lw=1, color="b")
    axes[0].set_title(rf'Initial Condition', fontsize=14)

    # Plot all orders together
    colors = ["r", "g", "m", "c", "y"]
    for i in range(len(orders)):
        f0_interp, z_interp, w_interp, W_interp, _, _, _ = interp_results[i]
        if i == 0:
            axes[1].plot(x_common, w_interp, color="b", lw=1, label=f'Exact (fourier)')
        if sim_type != "sv":
            axes[1].plot(x_common, z_interp, color = colors[i], lw=1, label=f'Quantum meas. order {orders[i]} ($n=${n[i]})')
        if sim_type != "meas":
            axes[1].plot(x_common, W_interp.real, '--', color = colors[i], lw=1, label=f'Quantum sv. order {orders[i]} ($n=${n[i]})')

    axes[1].set_title(rf'$T =$ {T}', fontsize=14)
    axes[1].legend()

    # Determine min/max y-values across both plots
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

    # Apply the same y-limits to both axes
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

    # Adding the table to the plot
    headers=['', f'Error ({error_type})', 'Success rate', '1Q gates', 'CNOT gates']
    axes[2].axis('off')
    tbl = axes[2].table(cellText = table, colLabels = headers, loc = 'center')
    tbl.scale(1,2)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    axes[2].set_title('Data Table')

    plt.suptitle(rf'Advection-Diffusion Simulation at Time $T =$ {T} with Parameters $c =$ {c} and $\nu =$ {nu}',fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_simulations_2D(init_f, n_order=[(6,2),(6,4)], T=0.5, c1=1, c2=1, nu=0.1, shots=10**6, sim_type="sv", diff_plot=True):
    """
    n_order contains 2 tuples. Each tuple in n_order is of the form (n, order). For each tuple, the simulation runs with n spatial qubits and the given order.
    Quantum measurement and/or statevector results are plotted according to the `sim_type` argument ("meas", "sv", or "both").
    if diff_plot is True, the difference between the two orders is also plotted.
    A summary table of max error, succes rate and circuit complexity for each order is printed and displayed in the plot.
    """

    if len(n_order) != 2:
        raise ValueError("This function only supports two simulations (two entries in n_order).")

    if sim_type not in ["sv", "meas", "both"]:
        raise ValueError("sim_type must be 'sv', 'meas', or 'both'.")

    results = []
    orders = [order for (_, order) in n_order]
    n_list = [n for (n, _) in n_order]

    # Run simulations
    for i in range(2):
        x, y, f0, z, w, W, max_err, succ_rate, comp = Simulation_QC_2D.Sim(
            n_list[i], T, c1, c2, nu, init_f=init_f,
            order=orders[i], sim_type="both" if sim_type=="both" else sim_type,
            plot=False, shots=shots, Complexity=True
        )
        results.append((x, y, f0, z, w, W, max_err, succ_rate, comp))

    # Define common grid
    n_max = max(n_list)
    xmin = max(r[0].min() for r in results)
    xmax = min(r[0].max() for r in results)
    ymin = max(r[1].min() for r in results)
    ymax = min(r[1].max() for r in results)
    x_common = np.linspace(xmin, xmax, 2**n_max)
    y_common = np.linspace(ymin, ymax, 2**n_max)
    Xc, Yc = np.meshgrid(x_common, y_common, indexing="ij")
    pts = np.stack([Xc.ravel(), Yc.ravel()], axis=-1)

    # Interpolate both results to common grid
    interp_results = []
    for (x, y, f0, z, w, W, max_err, succ_rate, comp) in results:
        f_interp_func = RegularGridInterpolator((x, y), f0)
        w_interp_func = RegularGridInterpolator((x, y), w)
        z_interp_func = RegularGridInterpolator((x, y), z) if z is not None else None
        W_interp_func = RegularGridInterpolator((x, y), W) if W is not None else None

        f_interp = f_interp_func(pts).reshape(Xc.shape)
        w_interp = w_interp_func(pts).reshape(Xc.shape)
        z_interp = z_interp_func(pts).reshape(Xc.shape) if z_interp_func else None
        W_interp = W_interp_func(pts).reshape(Xc.shape) if W_interp_func else None

        interp_results.append((f_interp, z_interp, w_interp, W_interp, max_err, succ_rate, comp))

    # Compute difference (order2 - order1)
    if diff_plot:
        if sim_type != "sv":
            meas_diff = np.abs(interp_results[1][1] - interp_results[0][1])
        if sim_type != "meas":
            sv_diff = np.abs(interp_results[1][3].real - interp_results[0][3].real)

    # Create summary table
    table = [
        [f"Order {orders[i]} (n={n_list[i]})", f"{max_err[0]:.3e}", f"{succ_rate:.4f}", comp[0], comp[1]]
        for i, (_, _, _, _, _, _, max_err, succ_rate, comp) in enumerate(results)
    ]
    print(f"\n-- SUMMARY --")
    error_type = "m" if sim_type == "meas" else "sv"
    print(tabulate(table, headers=['', f'Error ({error_type})', 'Success rate', '1Q gates', 'CNOT gates'], tablefmt="simple_grid"))

    # Plotting
    if diff_plot:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 6, height_ratios=[4, 4, 1], width_ratios=[0.5, 1, 1, 1, 1, 0.5])
        ax_pad = fig.add_subplot(gs[0, 0:1])  # empty spacing for centering
        ax_pad.axis('off')
        ax_init = fig.add_subplot(gs[0, 1:3], projection='3d')  # initial
        ax_exact = fig.add_subplot(gs[0, 3:5], projection='3d')  # exact
        ax_empty = fig.add_subplot(gs[0, 5:])   # empty spacing for centering
        ax_empty.axis('off')
        ax_q1 = fig.add_subplot(gs[1, 0:2], projection='3d')  # quantum plot first order
        ax_q2 = fig.add_subplot(gs[1, 2:4], projection='3d')  # quantum plot second order
        ax_diff = fig.add_subplot(gs[1, 4:], projection='3d')  # difference plot
        ax_table = fig.add_subplot(gs[2, 1:5])  # table
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.01, top=0.92, hspace=0.2, wspace=0.0)  

    else:
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[4, 1])
        ax_init = fig.add_subplot(gs[0, 0], projection='3d')  # initial
        ax_exact = fig.add_subplot(gs[0, 1], projection='3d')  # exact
        ax_q1 = fig.add_subplot(gs[0, 2], projection='3d')  # quantum plot first order
        ax_q2 = fig.add_subplot(gs[0, 3], projection='3d')  # quantum plot second order
        ax_table = fig.add_subplot(gs[1, 1:3])  # table
        plt.subplots_adjust(left=0, right=0.98, top=1, bottom=0, hspace=0, wspace=0.1)

    # Determine global z-limits for initial, exact and quantum plots (not difference)
    z_values = []
    z_values.append(interp_results[0][0])      # Initial
    z_values.append(interp_results[0][2])      # Exact
    for i in range(2):
        if sim_type in ["sv", "meas"]:
            z_values.append(interp_results[i][3].real if sim_type=="sv" else interp_results[i][1])
        else:
            z_values.append(interp_results[i][3].real)
            z_values.append(interp_results[i][1])
    z_min = np.min([np.min(z) for z in z_values if z is not None])
    z_max = np.max([np.max(z) for z in z_values if z is not None])

    # Plot initial condition
    ax_init.plot_surface(Xc, Yc, interp_results[0][0], cmap='viridis')
    ax_init.set_title('Initial Condition')
    ax_init.set_zlim(z_min, z_max)

    # Plot exact solution
    ax_exact.plot_surface(Xc, Yc, interp_results[0][2], cmap='viridis')
    ax_exact.set_title('Exact Solution')
    ax_init.set_zlim(z_min, z_max)

    # Plot quantum results for each order
    for i, ax in zip(range(2), [ax_q1, ax_q2]):
        if sim_type in ["sv", "meas"]:
            Z = interp_results[i][3].real if sim_type=="sv" else interp_results[i][1]
            ax.plot_surface(Xc, Yc, Z, cmap='plasma')
            ax.set_title(f'Quantum {sim_type}. Order {orders[i]} (n= {n_list[i]})')
        else:  # sim_type == "both"
            # Plot both surfaces with labels
            ax.plot_surface(Xc, Yc, interp_results[i][3].real, cmap='plasma', alpha=0.7, label='sv.')
            ax.plot_surface(Xc, Yc, interp_results[i][1], cmap='cividis', alpha=0.7, label='meas.')
            legend_elements = [Patch(facecolor=plt.cm.plasma(0.5), label='sv.'), Patch(facecolor=plt.cm.cividis(0.5), label='meas.')]
            ax.legend(handles=legend_elements, loc='right', fontsize=10)
            ax.set_title(f'Quantum Order {orders[i]} (n= {n_list[i]})')
        ax_init.set_zlim(z_min, z_max)
        
    # Difference plot
    if diff_plot:
        if sim_type == "sv":
            ax_diff.plot_surface(Xc, Yc, sv_diff, cmap='coolwarm')
        elif sim_type == "meas":
            ax_diff.plot_surface(Xc, Yc, meas_diff, cmap='coolwarm')
        else:  # sim_type == "both"
            ax_diff.plot_surface(Xc, Yc, sv_diff, cmap='plasma', label='sv.')
            ax_diff.plot_surface(Xc, Yc, meas_diff, cmap='cividis', label='meas.')
            legend_elements = [Patch(facecolor=plt.cm.plasma(0.5), label='sv.'), Patch(facecolor=plt.cm.cividis(0.5), label='meas.')]
            ax_diff.legend(handles=legend_elements, loc='right', fontsize=10)
        ax_diff.set_title(f'Difference between order {orders[1]} and {orders[0]}')

    # Table
    ax_table.axis('off')
    tbl = ax_table.table(cellText=table,
                         colLabels=['', f'Error ({sim_type})', 'Success rate', '1Q gates', 'CNOT gates'],
                         loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.6)
    ax_table.set_title('Data Table', fontsize=14)

    fig.suptitle(f'2D Advection–Diffusion Simulation at Time T={T} with Parameters c₁={c1}, c₂={c2} and ν={nu}', fontsize=16)
    plt.show()


def run_examples(examples = [Gaussian, Sine_sum, Wave_pack, Bump, rec, Gaussian_2D, Sine_squared_2D], sim_type="sv", shots=10**6, diff_plot = True):
    """
    Runs and plots simulations for a list of example initial conditions.
    """
    for example in examples:
        
        if example not in [Gaussian, Sine_sum, Wave_pack, Bump, rec, Gaussian_2D, Sine_squared_2D]:
            raise ValueError(f"Example {example} not recognized.")
        
        if example == Gaussian:
            # plot_simulations(n_order=[(7,4),(6,6)], T=2, c=0, nu=0.02, init_f=Gaussian, shots = shots, sim_type=sim_type)
            plot_simulations(n_order=[(8,2),(6,6),(9,2),(7,6)], T=4, c=1, nu=0, init_f=Gaussian, shots = shots, sim_type=sim_type)
        
        elif example == Sine_sum:
            plot_simulations(n_order=[(8,6),(6,14)], T=0.5, c=0, nu=0.1, init_f=Sine_sum, shots=shots, sim_type = sim_type)
        
        elif example == Wave_pack:
            plot_simulations(n_order=[(10,6),(8,14)], T=1, c=1, nu=0, init_f=Wave_pack, shots=shots, sim_type = sim_type)
        
        elif example == Bump:
            plot_simulations(n_order=[(9,2),(7,6)], T=0.6, c=2, nu=0, init_f=Bump, shots=shots, sim_type = sim_type)
        
        elif example == rec:
            plot_simulations(n_order=[(8,2),(7,6)], T=1, c=1, nu=0.02, init_f=rec, shots=shots, sim_type=sim_type)
        
        elif example == Gaussian_2D:
            plot_simulations_2D(Gaussian_2D, n_order=[(7,2),(6,4)], T=0.5, c1=0, c2=0, nu=0.05, shots=shots, sim_type=sim_type, diff_plot=diff_plot)
        
        elif example == Sine_squared_2D:
            plot_simulations_2D(Sine_squared_2D, n_order=[(6,2),(5,4)], T=0.5, c1=1, c2=1, nu=0.1, shots=shots, sim_type=sim_type, diff_plot=diff_plot)
            

run_examples([Sine_squared_2D])
