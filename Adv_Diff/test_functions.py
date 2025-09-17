import numpy as np
import Simulation_QC
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

def Bump(x, I = [1.4,1.5,2.5,2.6]):
    """
    I = (a,b,c,d), a<b<c<d 
    Bump(x) = 1 for x in [b,c]
    Bump(x) = 0 for x not in (a,d)
    Bump(x) in (0,1) for x in (a,b) union (c,d)
    """
    
    return b((x-I[0])/(I[1]-I[0]))*b((I[3]-x)/(I[3]-I[2]))

def plot_simulations(init_f, n=[6,6,6], orders = [2, 4, 6], T=0.5, c=1, nu=0.1, shots=10**6, d=4, plot="both"):
    """
    For each order in `orders`, the simulation runs with the corresponding number of spatial qubits in `n`.
    Quantum measurement and/or statevector results are plotted according to the `plot` argument 
    ("meas", "sv", or "both"). A summary table of max errors and circuit complexity for each order is printed.
    """

    results = []

    # Run simulations
    for i in range(len(orders)):
        x, f0, z_list, w_list, W_list, max_err, comp = Simulation_QC.Sim(n[i], T, c, nu, init_f=init_f,
            order=orders[i], plot=False, shots=shots,Complexity=True)
        results.append((x, f0, z_list[0], w_list[0], W_list[0], max_err, comp))

    # Define a common grid (based on the finest resolution
    n_max = max(n)
    xmin = max([x.min() for (x, _, _, _, _, _, _) in results])
    xmax = min([x.max() for (x, _, _, _, _, _, _) in results])
    x_common = np.linspace(xmin, xmax, 2**n_max)

    # Interpolate results to common grid
    interp_results = []
    for (x, f0, z, w, W, max_err, comp) in results:
        f_interp = interp1d(x, f0, kind="linear")
        z_interp = interp1d(x, z, kind="linear")
        w_interp = interp1d(x, w, kind="linear")
        W_interp = interp1d(x, W, kind="linear")
        interp_results.append((f_interp(x_common), z_interp(x_common), w_interp(x_common), W_interp(x_common), max_err, comp))

    # Create summary table
    table = [
        [f"Order {orders[i]} (n={n[i]})", max_err[0][0], max_err[0][1], comp[0][0], comp[0][1]]
        for i, (_, _, _, _, _, max_err, comp) in enumerate(results)]
    print("-- SUMMARY --")
    print(tabulate(table, headers=['', 'Measurement max error', 'Statevector max error', '1-qubit gates', '2-qubit gates'], tablefmt="simple_grid"))

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # Plot initial condition
    axes[0].plot(x_common, interp_results[0][0], lw=1, color="b")
    axes[0].set_title(rf'Initial Condition', fontsize=14)

    # Plot all orders together
    colors = ["r", "g", "m"]
    for i in range(len(orders)):
        f0_interp, z_interp, w_interp, W_interp, _, _ = interp_results[i]
        if i == 0:
            axes[1].plot(x_common, w_interp, color="b", lw=1, label=f'Fourier')
        if plot != "sv":
            axes[1].plot(x_common, z_interp, color = colors[i], lw=1, label=f'Quantum meas. order {orders[i]} ($n=${n[i]})')
        if plot != "meas":
            axes[1].plot(x_common, W_interp.real, '--', color = colors[i], lw=1, label=f'Quantum sv. order {orders[i]} ($n=${n[i]})')

    axes[1].set_title(rf'Adv-Diff Simulation at Time $T =$ {T} with Parameters $c =$ {c} and $\nu =$ {nu}', fontsize=14)
    axes[1].legend()
    plt.show()

plot_simulations(Gaussian, n=[8,7,6], orders=[2, 4, 6], T=1, c=1, nu=0, plot="meas")
#plot_simulations(Sine_sum, n=[7,7,7], T=3, c=1, nu=0, shots=10**7)
#plot_simulations(Wave_pack, n=[8,8,8], T=0.5, c=1, nu=0, shots=10**7)
#plot_simulations(Bump, n=[7,7,7], T=0.5, c=1, nu=0, shots=10**6)