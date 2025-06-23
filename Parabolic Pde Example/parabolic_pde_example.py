from pde_residuals import heat_equation_residual
from utility_functions import *
from  visualize_bs import *
from DGM_Solver_Generalized import DGM_Solver_Generalized

# Setting Random Seed
torch.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':

# --- Example for 1D poisson Equation ---
    print("--- Example for 1D Heat Equation ---")
    spatial_dim_1d = 1
    alpha_1d = 0.01 # Thermal Diffusivity
    len_1d = 1.0 # Length of Domain
    T_end_1d = 1.0 # End Time

    domain_bound_1d = [[0.0,len_1d]]
    time_bound_1d = [0.0,T_end_1d]

    # Neural Network Architecture for 1D. (List of Layers)
    layer_sizes_1d = [32,32,32,1]
    num_pde_points = 5000
    num_ic_points = 1000
    num_bc_points = 1000
    epochs = 500
    learning_rate = 1e-3

    heat_solver_1d = DGM_Solver_Generalized(spatial_dimension=spatial_dim_1d,layer_sizes=layer_sizes_1d)

    # PDE parameters for heat equation
    heat_pde_params_1d = {'alpha': alpha_1d}

    heat_solver_1d.train(pde_residual_func=heat_equation_residual,
                                pde_parameters=heat_pde_params_1d,
                                domain_bound=domain_bound_1d,
                                boundary_condition_func=bc_1d_heat,
                                num_pde_points=num_pde_points, num_bc_points=num_bc_points,
                                epochs=epochs, learning_rate=learning_rate,num_ic_points=num_ic_points)
    print("Visualizing the Heat Equation Solution in 1D")
    plot_1d_solution(heat_solver_1d, domain_bound_1d,
                             analytical_solution_func=analytical_solution_1d_heat,
                             plot_params={'title': "1D Heat: DGM vs Analytical",'analytical_params': heat_pde_params_1d})



    print("\n" * 3 + "-" * 50 + "\n" * 3)
    print("2D Heat Equation Example")
    spatial_dim_2d = 2
    alpha_2d = 0.01
    lx_2d, ly_2d = 1.0,1.0
    T_end_2d = 0.5

    domain_bound_2d = [[0.0,lx_2d],[0.0,ly_2d]]
    time_bound_2d = [0.0,T_end_2d]

    layer_sizes = [64,64,64]

    num_pde_points = 20000
    num_bc_points = 4000
    num_ic_points=2000
    epochs = 15000
    learning_rate = 1e-3

    heat_pde_params_2d = {'alpha': alpha_2d}
    heat_solver_2d = DGM_Solver_Generalized(spatial_dim_2d,layer_sizes)

    heat_solver_2d.train(pde_residual_func=heat_equation_residual,
                     pde_parameters=heat_pde_params_2d,
                     domain_bound=domain_bound_2d,
                     boundary_condition_func=bc_2d_heat,
                     num_pde_points=num_pde_points, num_bc_points=num_bc_points,
                     epochs=epochs, learning_rate=learning_rate, num_ic_points=num_ic_points)

    print("Visualizing the Heat Equation Solution in 2D")
    plot_2d_solution_surface(heat_solver_2d, domain_bound_2d,analytical_solution_func=analytical_solution_2d_heat,
                             time_slice=0.01,plot_params={'title': "2D Heat Eq: DGM vs Analytical at t=0.01",
                                                          'analytical_params': heat_pde_params_2d})

    print("\n" * 3 + "-" * 50 + "\n" * 3)
    print("3D Heat Equation Example")

    spatial_dim_3d = 3
    alpha_3d = 0.01
    lx_3d, ly_3d, lz_3d = 1.0,1.0, 1.0
    t_end_3d = 0.5

    domain_bound_3d = [[0.0,lx_3d],[0.0,ly_3d],[0.0,lz_3d]]
    time_bound_3D = [0.0,t_end_3d]

    layer_sizes = [64,64,64,64]
    heat_pde_params_3d = {'alpha': alpha_3d}
    solver_3D = DGM_Solver_Generalized(spatial_dim_3d,layer_sizes)

    num_pde_points_3d = 20000
    num_ic_points_3d = 2000
    num_bc_points_3d = 4000
    epochs = 15000
    learning_rate = 1e-3

    solver_3D.train(pde_residual_func=heat_equation_residual,
                     pde_parameters=heat_pde_params_3d,
                     domain_bound=domain_bound_3d,
                     boundary_condition_func=bc_3d_heat,
                     num_pde_points=num_pde_points, num_bc_points=num_bc_points,
                     epochs=epochs, learning_rate=learning_rate, num_ic_points=num_ic_points)

    print("Visualizing the Heat Equation Solution in 3D")
    time_slices_3d_plot = [0.0, 0.02, 0.05, t_end_3d]
    plot_3d_solution_slice_over_time_subplots(solver_3D, fixed_coord_dim=2, fixed_coord_val=lz_3d/2,
                                           domain_bounds=domain_bound_3d, time_slices=time_slices_3d_plot,
                                           plot_params={'main_title': "3D Heat Eq Slice (Z=0.5) over Time"})

print("End of Training and Heat Equation Example")






























