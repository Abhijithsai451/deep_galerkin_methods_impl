from matplotlib import pyplot as plt

from black_scholes_trainer import BS_Solver_Generalized
from black_scholes_pde_residuals import *
import numpy as np
from utility_functions import get_device
from visualize import plot_2d_solution_surface

EPOCHS = 15000
LR = 1e-4
PDE_POINTS = 10000
BC_POINTS = 2000
IC_POINTS = 2000
LAMBDA_BC = 100
LAMBDA_IC = 1000
device = get_device()


# 1 Dimensional Black Scholes Equation
print("1 Dimensional Black Scholes Equation")

s_min, s_max = 0.1, 200.0
t_min, t_max = 0, T_exp_BS # Time Range (time elapsed)
domain_bound = [[s_min, s_max],[t_max, t_min]]
layer_sizes = [64,64,64,64,64,1]

solver = BS_Solver_Generalized(spatial_dimension = 1,  layer_sizes = layer_sizes)

pde_params_bs = {'r_bs': R_BS, 'sigma_bs': Sigma_BS, 'K_bs': K_BS, 'T_exp_bs': T_exp_BS}

solver.train(pde_residual_func= black_scholes_pde_residual_1d, pde_parameters= pde_params_bs,
              domain_bound = domain_bound,
              boundary_condition_func= black_scholes_boundary_condition_1d,
              num_pde_points = PDE_POINTS,num_bc_points = BC_POINTS,
              epochs= EPOCHS, learning_rate=  LR,lambda_bc = LAMBDA_BC,lambda_ic=LAMBDA_IC,
              initial_condition_func = black_scholes_initial_condition_1d, num_ic_points = IC_POINTS)

plot_times = [0.0, T_exp_BS * 0.25, T_exp_BS * 0.5, T_exp_BS * 0.75, T_exp_BS]
s_test = torch.linspace(s_min, s_max, 200).reshape(-1, 1).to(device)

fig, axes = plt.subplots(1, len(plot_times), figsize=(20, 5))
fig.suptitle("1D Black-Scholes Option Price: DGM vs Analytical")

for i, t_val in enumerate(plot_times):
    t_test = torch.full_like(s_test, t_val).to(device)
    s_t_test = torch.cat([s_test, t_test], dim=1)

    dgm_solution = solver.predict(s_t_test).cpu().detach().numpy()
    analytical_solution = black_scholes_analytical_1d(s_test, t_test).cpu().detach().numpy()

    ax = axes[i]
    ax.plot(s_test.cpu().numpy(), dgm_solution, label='DGM Solution', color='blue')
    ax.plot(s_test.cpu().numpy(), analytical_solution, label='Analytical', linestyle='--', color='red')
    ax.set_title(f"Time (t): {t_val:.2f}")
    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Option Price (V)")
    ax.grid(True)
    if i == 0:
        ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 2 Dimensional Black Scholes Equation
print("\n--- Running 2D Black-Scholes (DGM) ---")
s1_min, s1_max = 0.1, 200.0
s2_min, s2_max = 0.1, 200.0
t_min,t_max = 0, T_exp_BS
domain_bound = [[s1_min, s1_max],[s2_min, s2_max],[t_min, t_max]]
layer_sizes = [128, 128, 128, 1]

solver = BS_Solver_Generalized(spatial_dimension = 2,  layer_sizes = layer_sizes)

pde_params_bs = {'r_bs': R_BS, 'sigma1_bs': Sigma_BS,'sigma2_bs':Sigma_BS,'rho_bs':0.5, 'K_bs': K_BS, 'T_exp_bs': T_exp_BS}


solver.train(pde_residual_func= black_scholes_pde_residual_2d, pde_parameters= pde_params_bs,
              domain_bound = domain_bound,
              boundary_condition_func= lambda x_coords,t_coords:black_scholes_boundary_condition_2d(x_coords,t_coords),
              num_pde_points = PDE_POINTS * 4,num_bc_points = BC_POINTS * 4,
              epochs= EPOCHS * 3, learning_rate=  LR * 0.5,lambda_bc = LAMBDA_BC * 10, lambda_ic = LAMBDA_IC * 20,
              initial_condition_func =lambda x_coords: black_scholes_initial_condition_2d(x_coords), num_ic_points = IC_POINTS)

# Plotting for 2D Black-Scholes (surface plot at final time)
num_plot_points = 30  # Resolution for meshgrid
plot_t = domain_bound[2][1]  # Max time (expiration)

s1_grid, s2_grid = np.meshgrid(np.linspace(s1_min, s1_max, num_plot_points),
                               np.linspace(s2_min, s2_max, num_plot_points))
s1_test_flat = torch.tensor(s1_grid.ravel(), dtype=torch.float32).reshape(-1, 1).to(device)
s2_test_flat = torch.tensor(s2_grid.ravel(), dtype=torch.float32).reshape(-1, 1).to(device)
t_test_flat = torch.full_like(s1_test_flat, plot_t).to(device)

X_test_2d_plot = torch.cat([s1_test_flat, s2_test_flat, t_test_flat], dim=1)

dgm_solution_2d = solver.predict(X_test_2d_plot)
analytical_solution_2d = black_scholes_analytical_2d(s1_test_flat, s2_test_flat, t_test_flat)  # Placeholder

# Note: analytical_solution_2d is a placeholder. For proper comparison,
# you'd need a true analytical solution for the specific 2D option.
plot_2d_solution_surface(torch.cat([s1_test_flat, s2_test_flat], dim=1), dgm_solution_2d, analytical_solution_2d,
                         f"2D Black-Scholes (t={plot_t:.2f}) - DGM vs Placeholder Analytical", "S1", "S2")
