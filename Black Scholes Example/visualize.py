import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


def plot_1d_solution(x_test: torch.Tensor, dgm_solution: torch.Tensor, analytical_solution: torch.Tensor,
                     title: str, xlabel: str, ylabel: str, save_path: str = None):
    """Plots 1D DGM vs Analytical solution."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.cpu().numpy(), dgm_solution.cpu().numpy(), label='DGM Solution', color='blue')
    plt.plot(x_test.cpu().numpy(), analytical_solution.cpu().numpy(), label='Analytical Solution', linestyle='--',
             color='red')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_2d_solution_surface(X_test: torch.Tensor, dgm_solution: torch.Tensor, analytical_solution: torch.Tensor,
                             title: str, xlabel: str, ylabel: str, save_path: str = None):
    """Plots 2D DGM vs Analytical solution as a surface."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Assuming X_test is (N, 2) for (x, y)
    x_coords = X_test[:, 0].cpu().numpy()
    y_coords = X_test[:, 1].cpu().numpy()

    # Reshape for surface plot: assumes x, y are gridded
    # This part requires X_test to be generated from meshgrid for correct reshaping
    num_points_per_dim = int(np.sqrt(len(x_coords)))
    X = x_coords.reshape(num_points_per_dim, num_points_per_dim)
    Y = y_coords.reshape(num_points_per_dim, num_points_per_dim)
    Z_dgm = dgm_solution.cpu().numpy().reshape(num_points_per_dim, num_points_per_dim)
    Z_analytical = analytical_solution.cpu().numpy().reshape(num_points_per_dim, num_points_per_dim)

    ax.plot_surface(X, Y, Z_dgm, cmap=cm.viridis, alpha=0.8, label='DGM Solution', rstride=1, cstride=1)
    ax.plot_surface(X, Y, Z_analytical, cmap=cm.plasma, alpha=0.5, label='Analytical Solution', rstride=1, cstride=1)

    # Custom legend due to plot_surface not supporting label directly
    dgm_proxy = plt.Line2D([0], [0], linestyle="none", c=cm.viridis(0.5), marker='o')
    analytical_proxy = plt.Line2D([0], [0], linestyle="none", c=cm.plasma(0.5), marker='o')
    ax.legend([dgm_proxy, analytical_proxy], ['DGM Solution', 'Analytical Solution'], numpoints=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("u")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_2d_solution_contour(X_test: torch.Tensor, dgm_solution: torch.Tensor, analytical_solution: torch.Tensor,
                             title: str, xlabel: str, ylabel: str, save_path: str = None):
    """Plots 2D DGM vs Analytical solution as contour plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_coords = X_test[:, 0].cpu().numpy()
    y_coords = X_test[:, 1].cpu().numpy()

    num_points_per_dim = int(np.sqrt(len(x_coords)))
    X = x_coords.reshape(num_points_per_dim, num_points_per_dim)
    Y = y_coords.reshape(num_points_per_dim, num_points_per_dim)
    Z_dgm = dgm_solution.cpu().numpy().reshape(num_points_per_dim, num_points_per_dim)
    Z_analytical = analytical_solution.cpu().numpy().reshape(num_points_per_dim, num_points_per_dim)

    levels = np.linspace(min(Z_analytical.min(), Z_dgm.min()), max(Z_analytical.max(), Z_dgm.max()), 20)

    contour1 = ax1.contourf(X, Y, Z_dgm, levels=levels, cmap=cm.viridis)
    fig.colorbar(contour1, ax=ax1, label='DGM Solution u')
    ax1.set_title(f'{title} (DGM)')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_aspect('equal', adjustable='box')

    contour2 = ax2.contourf(X, Y, Z_analytical, levels=levels, cmap=cm.viridis)
    fig.colorbar(contour2, ax=ax2, label='Analytical Solution u')
    ax2.set_title(f'{title} (Analytical)')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_3d_solution_slice(X_test_slice: torch.Tensor, dgm_solution_slice: torch.Tensor,
                           analytical_solution_slice: torch.Tensor,
                           fixed_coord_name: str, fixed_coord_value: float, title_prefix: str = "",
                           save_path: str = None):
    """
    Plots a 2D slice of a 3D solution (DGM vs Analytical).
    X_test_slice is expected to be (N, 2) for the two varying coordinates.
    """
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Assuming X_test_slice is (N, 2) for (x,y) if z is fixed, etc.
    # Need to reshape for surface plot, so assume points are from meshgrid
    num_points_per_dim = int(np.sqrt(len(X_test_slice)))
    coord1 = X_test_slice[:, 0].cpu().numpy().reshape(num_points_per_dim, num_points_per_dim)
    coord2 = X_test_slice[:, 1].cpu().numpy().reshape(num_points_per_dim, num_points_per_dim)

    Z_dgm = dgm_solution_slice.cpu().numpy().reshape(num_points_per_dim, num_points_per_dim)
    Z_analytical = analytical_solution_slice.cpu().numpy().reshape(num_points_per_dim, num_points_per_dim)

    # Plot DGM Solution
    surf1 = ax1.plot_surface(coord1, coord2, Z_dgm, cmap=cm.viridis, alpha=0.8, rstride=1, cstride=1)
    ax1.set_title(f'{title_prefix} DGM Solution\n({fixed_coord_name}={fixed_coord_value:.2f})')
    ax1.set_xlabel('Coord 1')
    ax1.set_ylabel('Coord 2')
    ax1.set_zlabel('u')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Plot Analytical Solution
    surf2 = ax2.plot_surface(coord1, coord2, Z_analytical, cmap=cm.plasma, alpha=0.8, rstride=1, cstride=1)
    ax2.set_title(f'{title_prefix} Analytical Solution\n({fixed_coord_name}={fixed_coord_value:.2f})')
    ax2.set_xlabel('Coord 1')
    ax2.set_ylabel('Coord 2')
    ax2.set_zlabel('u')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_meshgrid(domain_bounds: list, num_points_per_dim: int, device: torch.device):
    """
    Creates a meshgrid for 1D, 2D, or 3D domains.
    Returns X_test (N, D) and lists of meshgrid arrays for plotting.
    """
    ranges = [np.linspace(d_min, d_max, num_points_per_dim) for d_min, d_max in domain_bounds]

    if len(domain_bounds) == 1:
        x_mesh = ranges[0]
        X_test = torch.tensor(x_mesh, dtype=torch.float32).reshape(-1, 1).to(device)
        return X_test, [x_mesh]
    elif len(domain_bounds) == 2:
        x_mesh, y_mesh = np.meshgrid(ranges[0], ranges[1])
        X_test = torch.tensor(np.stack([x_mesh.ravel(), y_mesh.ravel()], axis=1), dtype=torch.float32).to(device)
        return X_test, [x_mesh, y_mesh]
    elif len(domain_bounds) == 3:
        x_mesh, y_mesh, z_mesh = np.meshgrid(ranges[0], ranges[1], ranges[2])
        X_test = torch.tensor(np.stack([x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()], axis=1),
                              dtype=torch.float32).to(device)
        return X_test, [x_mesh, y_mesh, z_mesh]
    else:
        raise ValueError("Unsupported dimension for meshgrid generation.")