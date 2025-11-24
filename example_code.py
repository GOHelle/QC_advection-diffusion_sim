# Import relevant modules and methods.
import numpy as np
from Adv_Diff import Simulation_QC
from Adv_Diff import Simulation_QC_2D

# specify parameters for 1D simulation
num_qubits = 6                                      # Number of spatial qubits
times = [0.25, 0.5, 0.75]                           # List of final times (could also be a single value)
adv_speed = 2                                       # Advection speed parameter
diff_coeff = 0.1                                    # Diffusion coefficient
domain_length = 4                                   # Length of spatial domain
init_f = lambda x: np.exp(-10*(x-4/3)**2)           # Initial function
shots = 10**6                                       # Number of measurement shots
report_complexity = True                            # Whether or not to display complexity data
order = 4                                           # Method order
tolerance = 10**(-6)                                # error tolerance when deriving angle sequences
sim_type  = "both"                                  # Type of simulation: "meas" for measurement-based, "sv" for statevector-based, or "both"
compute_exact = True                                # Whether or not to compute and plot the exact solution
plot = True                                         # Whether or not to plot results

# Run 1D simulation with specified parameters. 
Simulation_QC.simulate_adv_diff(num_qubits, times, adv_speed, diff_coeff, domain_length, init_f, shots, report_complexity, order, tolerance, sim_type, compute_exact, plot)

# specify parameters for 2D simulation
num_qubits = 6                                              # Number of qubits per spatial dimension
time = 0.5                                                  # Final time for evolution
adv_speed_x = 1                                             # Advection speed parameter in x direction
adv_speed_y = 1                                             # Advection speed parameter in y direction
diff_coeff = 0.05                                           # Diffusion coefficient
domain_length = 4                                           # Length of each spatial dimension
init_f = lambda X, Y: np.sin(np.pi * (0.5 * X + Y)) ** 2    # Initial function
shots = 10**8                                               # Number of measurement shots
report_complexity = True                                    # Whether or not to display complexity data
order = 2                                                   # Method order
tolerance = 10**(-6)                                        # error tolerance when deriving angle sequences
sim_type  = "both"                                          # Type of simulation: "meas" for measurement-based, "sv" for statevector-based, or "both"
compute_exact = True                                        # Whether or not to compute and plot the exact solution
plot = True                                                 # Whether or not to plot results

# Run 2D simulation with specified parameters. 
Simulation_QC_2D.simulate_adv_diff_2d(num_qubits, time, adv_speed_x, adv_speed_y, diff_coeff, domain_length, init_f, shots, report_complexity, order, tolerance, sim_type, compute_exact, plot)
