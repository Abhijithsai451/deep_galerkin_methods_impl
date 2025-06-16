import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_1d_solution_elliptic(solver_instance, domain_bounds, analytical_solution_func=None, plot_params=None):
    """Plots the 1D solution for elliptic PDEs."""
    if plot_params is None:
        plot_params = {}

    num_points = 200
    x_np = np.linspace(domain_bounds[0][0], domain_bounds[0][1], num_points).reshape(-1, 1)
    x_torch = torch.from_numpy(x_np).float().to(solver_instance.device)

    # For plotting elliptic (steady-state), time is 0
    t_dummy = torch.zeros_like(x_torch[:, 0:1]).to(solver_instance.device)

    solver_instance.model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        u_pred_torch = solver_instance.predict(x_torch, t_dummy)
    u_pred_np = u_pred_torch.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(x_np, u_pred_np, label='DGM Solution', color='blue')

    if analytical_solution_func:
        u_analytical_np = analytical_solution_func(x_np, {})
        plt.plot(x_np, u_analytical_np, label='Analytical Solution', linestyle='--', color='red')

    plt.title(plot_params.get('title', "1D Elliptic PDE Solution"))
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d_solution_elliptic(solver_instance, domain_bounds, title="2D Elliptic PDE Solution"):
    """Plots the 2D solution for elliptic PDEs."""
    num_points = 50
    x_coords = np.linspace(domain_bounds[0][0], domain_bounds[0][1], num_points)
    y_coords = np.linspace(domain_bounds[1][0], domain_bounds[1][1], num_points)
    X, Y = np.meshgrid(x_coords, y_coords)

    coords_flat_np = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    coords_flat_torch = torch.from_numpy(coords_flat_np).float().to(solver_instance.device)

    # For plotting elliptic (steady-state), time is 0
    t_dummy = torch.zeros_like(coords_flat_torch[:, 0:1]).to(solver_instance.device)

    solver_instance.model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        u_pred_flat = solver_instance.predict(coords_flat_torch, t_dummy).cpu().numpy()

    U_pred = u_pred_flat.reshape(num_points, num_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, U_pred, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u(x,y)')
    plt.show()


def plot_3d_slice_elliptic(solver_instance, fixed_coord_dim, fixed_coord_val, domain_bounds,
                           title="3D Elliptic PDE Slice"):
    """Plots a 2D slice of a 3D elliptic PDE solution."""
    if solver_instance.spatial_dimension != 3:
        print("This function is for 3D solutions only.")
        return

    varying_dims = [i for i in range(3) if i != fixed_coord_dim]

    num_points = 50
    coord1_vals = np.linspace(domain_bounds[varying_dims[0]][0], domain_bounds[varying_dims[0]][1], num_points)
    coord2_vals = np.linspace(domain_bounds[varying_dims[1]][0], domain_bounds[varying_dims[1]][1], num_points)

    C1, C2 = np.meshgrid(coord1_vals, coord2_vals)

    coords_flat_np = np.zeros((num_points * num_points, 3))
    coords_flat_np[:, varying_dims[0]] = C1.flatten()
    coords_flat_np[:, varying_dims[1]] = C2.flatten()
    coords_flat_np[:, fixed_coord_dim] = fixed_coord_val

    coords_flat_torch = torch.from_numpy(coords_flat_np).float().to(solver_instance.device)

    # For plotting elliptic (steady-state), time is 0
    t_dummy = torch.zeros_like(coords_flat_torch[:, 0:1]).to(solver_instance.device)

    solver_instance.model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        u_pred_flat = solver_instance.predict(coords_flat_torch, t_dummy).cpu().numpy()

    U_pred_slice = u_pred_flat.reshape(num_points, num_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(C1, C2, U_pred_slice, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_title(f"{title}\n(Fixed {['X', 'Y', 'Z'][fixed_coord_dim]}={fixed_coord_val:.2f})")
    ax.set_xlabel(['X', 'Y', 'Z'][varying_dims[0]])
    ax.set_ylabel(['X', 'Y', 'Z'][varying_dims[1]])
    ax.set_zlabel('u')
    plt.show()