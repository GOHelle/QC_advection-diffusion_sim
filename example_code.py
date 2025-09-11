# Import relevant modules and methods.
import numpy as np
from Adv_Diff import Simulation_QC
from Adv_Diff import Simulation_QC_2D

# specify parameters for 1D simulation
n = 6                                       # Number of spatial qubits
T = [0.25, 0.5, 0.75]                       # List of final times (could also be a single value)
c = 1                                       # Advection speed parameter
nu = 0.05                                   # Diffusion coefficient
d = 4                                       # Length of spatial domain
init_f = lambda x: np.exp(-10*(x-4/3)**2)   # Initial function
shots = 10**6                               # Number of measurement shots
Complexity = True                           # Whether or not to display complexity data
order = 6                                   # Method order
eps = 10**(-6)                              # error tolerance when deriving angle sequences

# Run 1D simulation with specified parameters. 
Simulation_QC.Sim(n, T, c, nu, d, init_f, shots, Complexity, order, eps)

# specify parameters for 2D simulation
n = 5                                                       # Number of qubits per spatial dimension
T = 1                                                       # Final time for evolution
c1 = 1                                                      # Advection speed parameter in x direction
c2 = 2                                                      # Advection speed parameter in y direction
nu = 0.1                                                    # Diffusion coefficient
d = 4                                                       # Length of each spatial dimension
init_f = lambda X, Y: np.sin(np.pi * (0.5 * X + Y)) + 1     # Initial function
shots = 10**7                                               # Number of measurement shots
Complexity = True                                           # Whether or not to display complexity data
order = 2                                                   # Method order
eps = 10**(-6)                                              # error tolerance when deriving angle sequences

# Run 2D simulation with specified parameters. 
Simulation_QC_2D.Sim(n, T, c1, c2, nu, d, init_f, shots, Complexity, order, eps)