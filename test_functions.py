import numpy as np
from Adv_Diff import Simulation_QC
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.interpolate import interp1d

def Gaussian(x,c = 5/3,scale = 10):
    return np.exp(-scale*(x-c)**2)

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

def plot_simulations(init_f, n_order=[(6,2),(6,4),(6,6)], T=0.5, c=1, nu=0.1, shots=10**6, d=4, sim_type="both"):
    """
    each tuple in n_order is of the form (n, order). For each tuple in this list, the simulation runs with n spatial qubits and the given order.
    Quantum measurement and/or statevector results are plotted according to the `sim_type` argument ("meas", "sv", or "both"). 
    A summary table of max error, succes rate and circuit complexity for each order is printed and displayed in the plot.
    """

    results = []
    orders = [order for (_, order) in n_order]
    n = [n for (n, _) in n_order]

    # Run simulations
    for i in range(len(orders)):
        x, f0, z_list, w_list, W_list, max_err, succ_rate, comp = Simulation_QC.Sim(n[i], T, c, nu, init_f=init_f,
            order=orders[i], sim_type=sim_type, plot=False, shots=shots,Complexity=True)

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
        [f"Order {orders[i]} (n={n[i]})", f"{max_err[0][0]:.3e}", f"{succ_rate[0]:.3e}", comp[0][0], comp[0][1]]
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
    colors = ["r", "g", "m"]
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

    plt.suptitle(rf'Adv-Diff Simulation at Time $T =$ {T} with Parameters $c =$ {c} and $\nu =$ {nu}',fontsize=15)
    plt.tight_layout()
    plt.show()

plot_simulations(Gaussian, n_order=[(8,2),(7,4),(6,6)], T=1, c=1, nu=0.1, sim_type="both")
#plot_simulations(Sine_sum, n=[7,7,7], T=3, c=1, nu=0, shots=10**7)
#plot_simulations(Wave_pack, n=[8,8,8], T=0.5, c=1, nu=0, shots=10**7)
#plot_simulations(Bump, n=[7,7,7], T=0.5, c=1, nu=0, shots=10**6)
