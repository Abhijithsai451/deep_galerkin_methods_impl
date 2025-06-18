import numpy as np
import torch

from sampling import device

def get_device():
    if torch.backends.mps.is_available(): # For Apple Silicon GPUs
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

def analytical_solution_1d_poisson(x, t_torch, pde_params):
    return -x ** 2 + 2 * x

#<<<<<<--------- All Utility Functions for 2D poissons equations
# Source term f(x,y) = -2*pi^2 * sin(pi*x) * sin(pi*y)
def f_func_2d(coords):
    x, y = coords[:, 0:1], coords[:, 1:2]
    return -2.0 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

 # Boundary conditions: u(x,y) = 0 on all boundaries
def bc_2d_poisson(spatial_coord,lx_1d):
    return torch.zeros_like(spatial_coord[:, 0:1])

def analytical_solution_2d_poisson(x: torch.Tensor, t: torch.Tensor, params: dict):
    """
    Analytical solution for the 2D Poisson equation:
    - (d^2u/dx^2 + d^2u/dy^2) = 4
    The solution is u(x,y) = -x^2 + 2x - y^2 + 2y.
    """
    """
    x = x.float()
    t = t.float()
    """
    # Calculate the analytical solution
    u_analytical = -x**2 + 2 * x - t**2 + 2 * t
    return u_analytical

#<<<<<<--------- All Utility Functions for 3D poissons equations
def f_func_3d(coords):
    x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    return -3.0 * torch.pi ** 2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z)

def bc_3d_poisson(spatial_coord,lx_1d):
    return torch.zeros_like(spatial_coord[:, 0:1])

def analytical_solution_3d_poisson(x, y, z,pde_params):
    """
    Calculates the Analytical Solution for the 3D Poisson Equations
    Equation: u(x,y,z) = -x^2 + 2x - y^2 + 2y - z^2 + 2z
    """
    u_analytical = -x**2 + 2 * x - y**2 + 2 * y + z**2 + z

    return u_analytical

#<<<<<<--------- All Utility Functions for 1D Heat equations
def bc_1d_heat(spatial_coord, random_variable) :
    return torch.zeros_like(spatial_coord[:, 0:1])

def ic_1d_heat(spatial_coord: torch.Tensor):
    x = spatial_coord[:, 0:1]
    return torch.sin(torch.pi * x)

def analytical_solution_1d_heat(x: torch.Tensor, t: torch.Tensor, params: dict):
    alpha = params.get('alpha', 0.01)
    x = x.to(device)
    t = t.to(device)
    soln = torch.sin(torch.pi * x) * torch.exp(-alpha * torch.pi**2 * t)
    return soln.cpu().numpy()

#<<<<<<--------- All Utility Functions for 2D Heat equations

def bc_2d_heat(spatial_coord, random_vaiable):
    # u(x,y,t) = 0 on all boundaries
    return torch.zeros_like(spatial_coord[:, 0:1])

def ic_2d_heat(spatial_coord):
    # u(x,y,0) = sin(pi*x) * sin(pi*y)
    x, y = spatial_coord[:, 0:1], spatial_coord[:, 1:2]
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def analytical_solution_2d_heat(coords_np: np.ndarray, t_np: np.ndarray, params: dict) -> np.ndarray:
    x, y = coords_np[:, 0:1], coords_np[:, 1:2]
    alpha = params.get('alpha', 0.01)  # Get alpha from params
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * alpha * np.pi ** 2 * t_np)


#<<<<<<--------- All Utility Functions for 3D Heat equations

def bc_3d_heat(spatial_coord,random_vaiable):
    return torch.zeros_like(spatial_coord[:, 0:1])