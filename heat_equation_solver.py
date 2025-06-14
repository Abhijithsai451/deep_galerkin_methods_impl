import numpy as np
import torch

import visualize
from heat_equation import HeatEquation


if __name__ == '__main__':
    print("1D Heat Equation Example")
    spatial_dim_1d = 1
    alpha_1D = 0.01 # Thermal Diffusivity
    len_1D = 1.0 # Length of Domain
    T_end_1D = 1.0 # End Time

    domain_bound_1D = [[0.0,len_1D]] # Format: [[xmin, xmax]]
    time_bound_1D = [0.0,T_end_1D]

    # Neural Network Architecture for 1D. (List of Layers)
    Layer_sizes_1d = [32,32,32,1]

    solver = HeatEquation(spatial_dimension=spatial_dim_1d,layer_sizes=Layer_sizes_1d)

    num_pde_points_1D = 5000
    num_ic_points_1D = 1000
    num_bc_points_1D = 1000
    epochs = 5000
    learning_rate = 1e-3

    solver.train(alpha_1D,domain_bound_1D,time_bound_1D,num_pde_points_1D,num_ic_points_1D,
                 num_bc_points_1D,epochs=epochs, learning_rate=learning_rate)

    visualize.plot_1d_(solver, alpha_1D,domain_bound_1D,time_bound_1D)


































