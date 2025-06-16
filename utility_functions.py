import numpy as np
import torch

# <<<<<--------- All Utility Functions for 1D poissons equations
def f_func_1d(x_coords):
    return -2.0 * torch.ones_like(x_coords[:, 0:1])

def bc_1d_poisson(spatial_coord,lx_1d):
    x = spatial_coord[:, 0:1]
    tolerance = 1e-6
    result = torch.zeros_like(x)
    # Apply BC at x=1
    result[torch.abs(x - lx_1d) < tolerance] = 1.0
    # u(0)=0 is implicitly handled because result is initialized to zeros
    return result

def analytical_solution_1d_poisson(x: np.ndarray, params: dict) -> np.ndarray:
    return -x ** 2 + 2 * x

# ---------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#<<<<<<--------- All Utility Functions for 2D poissons equations
# Source term f(x,y) = -2*pi^2 * sin(pi*x) * sin(pi*y)
def f_func_2d(coords: torch.Tensor) -> torch.Tensor:
    x, y = coords[:, 0:1], coords[:, 1:2]
    return -2.0 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

 # Boundary conditions: u(x,y) = 0 on all boundaries
def bc_2d_poisson(spatial_coord,lx_1d):
    return torch.zeros_like(spatial_coord[:, 0:1])
# ---------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



#<<<<<<--------- All Utility Functions for 3D poissons equations
# Source term f(x,y,z) = -3*pi^2 * sin(pi*x) * sin(pi*y) * sin(pi*z)
def f_func_3d(coords: torch.Tensor) -> torch.Tensor:
    x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    return -3.0 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z)

# Boundary conditions: u(x,y,z) = 0 on all boundaries
def bc_3d_poisson(spatial_coord,lx_1d) -> torch.Tensor:
    return torch.zeros_like(spatial_coord[:, 0:1])
# ---------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
