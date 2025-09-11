import numpy as np
import Simulation_QC
import matplotlib.pyplot as plt
from tabulate import tabulate

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

def plot_simulations(init_f, n=[6,6,6], T=0.5, c=1, nu=0.1, shots = 10 **6):

    orders = [2, 4, 6]
    results = []

    # Run simulations
    for i in range(3):
        x, f0, z_list, w_list, max_err = Simulation_QC.Sim(
            n[i], T, c, nu, init_f=init_f, order=orders[i], plot=False, shots = shots, Complexity=False, max_error=False)
        results.append((x, z_list[0], w_list[0], max_err))

    # create table of max errors
    table = [
        [f"Order {orders[i]} (n={n[i]})", max_err[0][0], max_err[0][1] if len(max_err[0]) > 1 else 'N/A']
        for i, (_, _, _, max_err) in enumerate(results)]
    print("-- MAX ERRORS --")
    print(f"Number of shots for measurement: 10^{int(np.log10(shots))}")
    print(tabulate(table, headers=['', 'Measurement max error', 'Statevector max error'], tablefmt="simple_grid"))

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # Plot initial condition
    x, _, _, _ = results[0]
    axes[0].plot(x, f0, lw=1)
    axes[0].set_title(rf'Initial Condition', fontsize=14)

    # Plot all orders together
    for i in range(3):
        x, z, w, max_err = results[i]
        if i ==0:
            axes[1].plot(x, w, lw=1, label=f'Fourier')
        axes[1].plot(x, z, lw=1,
                     label=f'Quantum order {orders[i]} ($n=${n[i]})')

    axes[1].set_title(rf'Adv-Diff Simulation at Time $T =$ {T} with Parameters $c =$ {c} and $\nu =$ {nu}', fontsize=14)
    axes[1].legend()
    plt.show()


plot_simulations(Gaussian, n=[5,5,5], T=0.5, c=1, nu=0.2)
plot_simulations(Sine_sum, n=[6,6,6], T=0.5, c=1, nu=0.1, shots=10**7)
plot_simulations(Wave_pack, n=[7,7,7], T=0.5, c=1, nu=0.01, shots=10**7)
plot_simulations(Bump, n=[7,7,7], T=0.5, c=1, nu=0.5, shots=10**6)