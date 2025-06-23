import torch

# Black Scholes parameters
K_BS = 100.0 # Strike price
R_BS = 0.05 # Risk-Free rate
Sigma_BS = 0.2 # Volatility
T_exp_BS = 1.0  # Total Time to expiration

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def generate_pde_points_bs(num_points, spatial_domain, time_domain):
    coords = []
    for d_min, d_max in spatial_domain:
        coords.append(torch.rand(num_points,1)* (d_max - d_min) + d_min)
    if time_domain:
        t_min, t_max = time_domain
        coords.append(torch.rand(num_points,1)* (t_max - t_min) + t_min)
    return torch.cat(coords, dim=1).to(device)

def generate_bc_points_bs(num_points, spatial_domain, time_domain, spatial_dimension, time_dependent):
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



def generate_ic_points_bs(num_points,domain_bound, spatial_domain):
    coords = []
    for d_min, d_max in spatial_domain:
        coords.append(torch.rand(num_points,1) * (d_max - d_min) + d_min)
    coords.append(torch.full((num_points,1), domain_bound[-1][1]))
    return torch.cat(coords, dim =1).to(device)
"""
def generate_ic_points(num_points, domain_bound, spatial_domain):
    coords = []
    s_min, s_max = spatial_domain[0] # Assuming S is the first spatial dimension

    num_kink_focused = int(num_points * 0.8)
    num_uniform = num_points - num_kink_focused

    # 1. Uniform sampling across the whole S range
    s_uniform = torch.rand(num_uniform, 1) * (s_max - s_min) + s_min

    kink_width = (s_max - s_min) * 0.1
    s_kink_min = max(s_min, K_BS - kink_width / 2)
    s_kink_max = min(s_max, K_BS + kink_width / 2)

    s_kink_focused = torch.rand(num_kink_focused, 1) * (s_kink_max - s_kink_min) + s_kink_min

    S_coords_for_ic = torch.cat([s_uniform, s_kink_focused], dim=0)
    coords.append(S_coords_for_ic)

    for d_idx in range(1, len(spatial_domain)):
        d_min, d_max = spatial_domain[d_idx]
        coords.append(torch.rand(num_points, 1) * (d_max - d_min) + d_min)

    # Time coordinate at expiration (ensure this is T_EXP_BS, using domain_bound[-1][1])
    coords.append(torch.full((num_points, 1), domain_bound[-1][1]))

    return torch.cat(coords, dim=1).to(device)
"""