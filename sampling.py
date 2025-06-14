import torch
from torch import device

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def initial_condition(param):
    pass

def generate_random_points(num_points, bounds):
    # Generates random points
    points = []
    for b in bounds:
        point = torch.randn((num_points,1),dtype=torch.float32) * (b[1] - b[0]) + b[0]
        points.append(point)
    return torch.cat(points, dim =1).to(device)

def generate_pde_points(num_points, domain_bound, time_bound):
    # Generate random spatial and time points for PDE residual calculation
    spatial_coords = generate_random_points(num_points, domain_bound)
    t = (torch.rand((num_points,1),dtype=torch.float32) * (time_bound[1] - time_bound[0])+ time_bound[0]).to(device)

    return spatial_coords, t

def generate_ic_points(num_points, domain_bound, time_bound):
    # Generates the random spatial points for initial conditions at t = time_bound[0]
    spatial_coords = generate_random_points(num_points, domain_bound)
    u0_ic = initial_condition(spatial_coords.cpu()).to(device)
