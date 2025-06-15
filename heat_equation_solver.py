import numpy as np
import torch

import visualize
from heat_equation import HeatEquation


if __name__ == '__main__':
    """
    print("1D Heat Equation Example")
    spatial_dim_1d = 1
    alpha_1D = 0.01 # Thermal Diffusivity
    len_1D = 1.0 # Length of Domain
    T_end_1D = 1.0 # End Time

    domain_bound_1D = [[0.0,len_1D]] # Format: [[xmin, xmax]]
    time_bound_1D = [0.0,T_end_1D]

    # Neural Network Architecture for 1D. (List of Layers)
    Layer_sizes_1d = [32,32,32,1]

    solver_1d = HeatEquation(spatial_dimension=spatial_dim_1d,layer_sizes=Layer_sizes_1d)

    num_pde_points_1D = 5000
    num_ic_points_1D = 1000
    num_bc_points_1D = 1000
    epochs = 5000
    learning_rate = 1e-3

    solver_1d.train(alpha_1D,domain_bound_1D,time_bound_1D,num_pde_points_1D,num_ic_points_1D,
                 num_bc_points_1D,epochs=epochs, learning_rate=learning_rate)
    print("Visualizing the Heat Equation Solution in 1D")
    visualize.plot_1d_(solver_1d, alpha_1D,domain_bound_1D,time_bound_1D)


    print("2D Heat Equation Example")

    spatial_dim_2D = 2
    alpha_2D = 0.01
    lx_2d, ly_2d = 1.0,1.0
    T_end_2d = 0.5

    domain_bound_2D = [[0.0,lx_2d],[0.0,ly_2d]]
    time_bound_2D = [0.0,T_end_2d]

    layer_sizes_2D = [64,64,64,1]

    solver_2D = HeatEquation(spatial_dim_2D,layer_sizes_2D)

    num_pde_points_2D = 10000
    num_ic_points_2D = 2000
    num_bc_points_2D = 4000
    epochs = 7500
    learning_rate = 1e-3

    solver_2D.train(alpha_2D,domain_bound_2D,time_bound_2D,num_pde_points_2D,num_ic_points_2D,num_bc_points_2D,
                    epochs=epochs, learning_rate=learning_rate)
    print("Visualizing the Heat Equation Solution in 2D")

    visualize.plot_2d_(solver_2D, domain_bound_2D, time_point=0.01)
    visualize.plot_2d_(solver_2D, domain_bound_2D, time_point=0.1)
    visualize.plot_2d_(solver_2D, domain_bound_2D, time_point=T_end_2d)
    """

    print("3D Heat Equation Example")

    spatial_dim_3D = 3
    alpha_3D = 0.01
    lx_3D, ly_3D, lz_3D = 1.0,1.0, 1.0
    T_end_3D = 0.5

    domain_bound_3D = [[0.0,lx_3D],[0.0,ly_3D],[0.0,lz_3D]]
    time_bound_3D = [0.0,T_end_3D]

    layer_sizes_3D = [64,64,64,64, 1]

    solver_3D = HeatEquation(spatial_dim_3D,layer_sizes_3D)

    num_pde_points_3D = 15000
    num_ic_points_3D = 3000
    num_bc_points_3D = 6000
    epochs = 5000
    learning_rate = 1e-3

    solver_3D.train(alpha_3D,domain_bound_3D,time_bound_3D,num_pde_points_3D,num_ic_points_3D,num_bc_points_3D,
                    epochs=epochs, learning_rate=learning_rate)

    print("Visualizing the Heat Equation Solution in 2D")

    visualize.plot_3d_(solver_3D,domain_bound=domain_bound_3D,time_point=T_end_3D / 2,fixed_coord_dim =2, fixed_coord_val=lz_3D/2)
    visualize.plot_3d_(solver_3D,domain_bound=domain_bound_3D,time_point=T_end_3D,fixed_coord_dim =2, fixed_coord_val=lx_3D/2)





























