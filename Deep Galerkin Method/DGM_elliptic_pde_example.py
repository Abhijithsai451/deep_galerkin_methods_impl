from  DGM_Solver_Generalized import DGM_Solver_Generalized
from pde_residuals import *
from utility_functions import *
from visualize_bs import *


# Setting Random Seed
torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    # --- Example for 1D poisson Equation ---
    print("--- Example for 1D poisson Equation ---")
    spatial_dim_1d = 1
    lx_1d = 1.0
    domain_bound_1d = [[0.0, lx_1d]]
    poisson_pde_params_1d = {'f_func': f_func_1d}
    num_pde_points = 5000
    num_bc_points = 1000
    epochs = 5000
    learning_rate = 1e-2
    layer_sizes = [32,32,32]

    poisson_solver = DGM_Solver_Generalized(spatial_dim_1d,layer_sizes)
    print("Created the Deep Galerkin Neural Model for 1D Poisson Equation ")

    poisson_solver.train(pde_residual_func = poisson_equation_residual,
                         pde_parameters = poisson_pde_params_1d,
                         domain_bound = domain_bound_1d,
                         boundary_condition_func = bc_1d_poisson,
                         num_pde_points = num_pde_points,
                         num_bc_points = num_bc_points,
                         epochs = epochs, learning_rate = learning_rate
                         )
    print("Visualizing the Poisson Equation Solution in 1D")
    plot_1d_solution(poisson_solver, domain_bound_1d,
                             analytical_solution_func=analytical_solution_1d_poisson,
                             plot_params={'title': "1D Poisson: DGM vs Analytical"})

# --- Example 2: 2D Poisson Equation ---
    print("\n" * 3 + "-" * 50 + "\n" * 3)  # Separator
    print("--- Example for  2D Poisson Equation ---")
    spatial_dim_2d = 2
    Lx_2d, Ly_2d = 1.0, 1.0
    domain_bounds_2d = [[0.0, Lx_2d], [0.0, Ly_2d]]
    num_pde_points = 10000
    num_bc_points = 4000
    epochs = 7500
    learning_rate = 1e-4
    layer_sizes = [64, 64, 64,63,32]
    poisson_pde_params_2d = {'f_func': f_func_2d}



    poisson_solver_2d = DGM_Solver_Generalized(spatial_dim_2d, layer_sizes=layer_sizes)
    print("Created the Deep Galerkin Neural Model for 2D Poisson Equation ")

    poisson_solver_2d.train(pde_residual_func=poisson_equation_residual,
                                pde_parameters=poisson_pde_params_2d,
                                domain_bound=domain_bounds_2d,
                                boundary_condition_func=bc_2d_poisson,
                                num_pde_points=num_pde_points, num_bc_points=num_bc_points,
                                epochs=epochs, learning_rate=learning_rate)
    print("Visualizing the Poisson Equation Solution in 2D")
    plot_2d_solution_surface(poisson_solver_2d, domain_bounds=domain_bounds_2d,analytical_solution_func = analytical_solution_2d_poisson,
                              plot_params= {'title':"DGM Solution (2D Poisson Equation)"})
    time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
    plot_2d_solution_over_time_subplots(poisson_solver_2d, domain_bounds=domain_bounds_2d,time_slices=time_slices)

# --- Example 3: 3D Poisson Equation ---
    print("\n" * 3 + "-" * 50 + "\n" * 3)  # Separator
    print("--- Example for  3D Poisson Equation ---")
    spatial_dim_3d = 3
    Lx_3d, Ly_3d, Lz_3d = 1.0, 1.0, 1.0
    domain_bounds_3d = [[0.0, Lx_3d], [0.0, Ly_3d], [0.0, Lz_3d]]
    num_pde_points = 15000
    num_bc_points = 6000
    epochs = 3000
    learning_rate = 1e-3
    layer_sizes = [64, 64, 64,64]
    poisson_pde_params_3d = {'f_func': f_func_3d}

    poisson_solver_3d = DGM_Solver_Generalized(spatial_dim_3d, layer_sizes=layer_sizes)
    print("Created the Deep Galerkin Neural Model for 3D Poisson Equation ")

    poisson_solver_3d.train(pde_residual_func=poisson_equation_residual,
                                pde_parameters=poisson_pde_params_3d,
                                domain_bound=domain_bounds_3d,
                                boundary_condition_func=bc_3d_poisson,
                                num_pde_points=num_pde_points, num_bc_points=num_bc_points,
                                epochs=epochs, learning_rate=learning_rate)

    print("Visualizing the Poisson Equation Solution in 3D")
    plot_3d_solution_slice(poisson_solver_3d, fixed_coord_dim=2, fixed_coord_val=Lz_3d / 2,
                           domain_bounds=domain_bounds_3d, title="DGM 3D Poisson Slice")
    spatial_slices_3d_plot = [0.1, 0.25, 0.5, 0.75, 0.9]
    plot_3d_solution_spatial_slices_subplots(poisson_solver_3d, fixed_coord_dim=2,
                                             fixed_coord_vals=spatial_slices_3d_plot,
                                             domain_bounds=domain_bounds_3d,
                                             plot_params={'main_title': "3D Poisson DGM Solution Slices along Z"})


