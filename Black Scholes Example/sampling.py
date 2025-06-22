import torch


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def generate_pde_points(num_points, spatial_domain, time_domain):
    coords = []
    for d_min, d_max in spatial_domain:
        coords.append(torch.rand(num_points,1)* (d_max - d_min) + d_min)
    if time_domain:
        t_min, t_max = time_domain
        coords.append(torch.rand(num_points,1)* (t_max - t_min) + t_min)
    return torch.cat(coords, dim=1).to(device)

def generate_bc_points(num_points, spatial_domain, time_domain, spatial_dimension, time_dependent):
    """
    Generates the random points on the spatial boundaries for all the time points.
    """
    all_bc_points = []
    points_per_face = num_points // (2* spatial_dimension)
    if points_per_face == 0:
        points_per_face = 1
    for dim_idx in range(spatial_dimension):
        d_min, d_max = spatial_domain[dim_idx]

        # lower boundary for this dimension
        coords_lower = []
        for i, ( min_d, max_d) in enumerate(spatial_domain):
            if i == dim_idx:
                coords_lower.append(torch.full((points_per_face, 1), d_min))
            else:
                coords_lower.append(torch.rand(points_per_face,1)*  (max_d - min_d) + min_d)
        if time_domain:
            t_min, t_max = time_domain
            coords_lower.append(torch.rand(points_per_face,1)* (t_max - t_min) + t_min)
        all_bc_points.append(torch.cat(coords_lower, dim=1))
    if len(all_bc_points) > 0:
        return torch.cat(all_bc_points, dim= 0)[: num_points].to(device)
    return torch.empty(0, spatial_dimension + (1 if time_dependent else 0)).to(device)



def generate_ic_points(num_points,domain_bound, spatial_domain):
    coords = []
    for d_min, d_max in spatial_domain:
        coords.append(torch.rand(num_points,1) * (d_max - d_min) + d_min)
    coords.append(torch.full((num_points,1), domain_bound[-1][1]))
    return torch.cat(coords, dim =1).to(device)

