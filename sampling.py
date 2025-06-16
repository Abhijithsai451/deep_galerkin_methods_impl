import numpy as np
import torch
from torch import device

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

'''
def initial_condition(spatial_coord, spatial_dimension, domain_bound):
    lx = domain_bound[0][1] - domain_bound[0][0]
    if spatial_dimension > 1:
        ly = domain_bound[1][1] - domain_bound[1][0]
    if spatial_dimension > 2:
        lz = domain_bound[2][1] - domain_bound[2][0]

    if spatial_dimension == 1:
        x = spatial_coord[:,0:1]
        return torch.sin(torch.pi*x)
    elif spatial_dimension == 2:
        x = spatial_coord[:,0:1]
        y = spatial_coord[:,1:2]
        return torch.exp(-((x - lx / 2) ** 2 + (y - ly / 2) ** 2) / 0.05)
    elif spatial_dimension == 3:
        x = spatial_coord[:,0:1]
        y = spatial_coord[:,1:2]
        z = spatial_coord[:,2:3]

        return torch.exp(-((x - lx / 2) ** 2 + (y - ly / 2) ** 2 + (z - lz / 2) ** 2) / 0.05)
    else:
        raise ValueError(f"Unsupported spatial dimension: {spatial_dimension}."
                         "This function is only implemented for 2D or 3D.")

def boundary_condition(t):
    return torch.zeros_like(t)
'''
def generate_random_points(num_points, bounds):
    # Generates random points
    points = []
    for b in bounds:
        point = torch.randn((num_points,1),dtype=torch.float32) * (b[1] - b[0]) + b[0]
        points.append(point)
    return torch.cat(points, dim =1).to(device)

def generate_pde_points(num_points, domain_bound):
    # Generate random spatial and time points for PDE residual calculation
    spatial_coords = generate_random_points(num_points, domain_bound).to(device)
    t = torch.zeros_like(spatial_coords[:,0:1]).to(device)
    return spatial_coords, t

def generate_ic_points(num_points, domain_bound,initial_condition_func:callable):
    # Generates the random spatial points for initial conditions at t = time_bound[0]
    spatial_coords = generate_random_points(num_points, domain_bound).to(device)
    u_ic = initial_condition_func(spatial_coords).to(device)
    return spatial_coords, u_ic


def generate_bc_points(num_points, domain_bound, spatial_dimension, boundary_condition_func:callable):
    """
    1. Calculate the number of faces (num_faces) and points per face (points_per_face) by integer division.
    2. For each spatial dimension d:
        a. For the min face (e.g., x_d = xmin):
            - For each coordinate i:
            - If i == d, set the coordinate to the min value (domain_bounds[i][0]) for all points on this face.
            - Otherwise, sample randomly in the range [domain_bounds[i][0], domain_bounds[i][1]].
            - Also, for the time coordinate, sample randomly in [time_bounds[0], time_bounds[1]] for each point on this face.
        b. Similarly for the max face (e.g., x_d = xmax).
    3. After processing all faces, we might have a total of (points_per_face * num_faces) points, which may be less than `num_points`.
        So, if there are remaining points, we generate them randomly in the entire spatial domain (and time randomly as well) and add them.
    4. If we have generated more points (due to integer division), we truncate to `num_points`.
    5. Then, we evaluate the boundary condition function (boundary_condition_func) at these points to get the boundary values (U_bc_vals).
    Note: The function `generate_random_points_in_box` is used to generate the extra points in the spatial domain.
    """

    # Generates random points on the boundaries.
    all_x_bc = []

    num_faces = 2 * spatial_dimension
    points_per_face = num_points // num_faces

    if num_points % num_faces != 0:
        points_per_face += 1

    for d in range(spatial_dimension):
        # Minimum face for dimension
        coord_min_face = []
        for i in range(spatial_dimension):
            if i == d:
                coord_min_face.append(
                    torch.full((points_per_face, 1), domain_bound[i][0], dtype=torch.float32))
            else:
                coord_min_face.append(
                    torch.rand((points_per_face, 1), dtype=torch.float32) * (
                                domain_bound[i][1] - domain_bound[i][0]) + domain_bound[i][0])
        all_x_bc.append(torch.cat(coord_min_face, dim=1))

        # Maximum face for dimension
        coord_max_face = []
        for i in range(spatial_dimension):
            if i == d:
                coord_max_face.append(
                    torch.full((points_per_face, 1), domain_bound[i][1], dtype=torch.float32))
            else:
                coord_max_face.append(
                    torch.rand((points_per_face, 1), dtype=torch.float32) * (
                                domain_bound[i][1] - domain_bound[i][0]) + domain_bound[i][0])
        all_x_bc.append(torch.cat(coord_max_face, dim=1))

        X_bc_coords = torch.cat(all_x_bc, dim=0).to(device)
        if X_bc_coords.shape[0] > num_points:
            X_bc_coords = X_bc_coords[:num_points]

        U_bc_vals = boundary_condition_func(X_bc_coords,domain_bound[0][1]).to(device)
        return X_bc_coords, U_bc_vals

def analytic_func(x,t, alpha):
    soln = np.exp(-alpha * np.pi**2 * t)*np.sin(np.pi*x)
    return soln