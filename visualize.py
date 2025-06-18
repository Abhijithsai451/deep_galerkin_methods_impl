import math

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

'''
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
'''

def plot_1d_solution(solver_instance, domain_bounds, analytical_solution_func=None, time_slice=0.0, plot_params=None):
    """
    Plots the 1D solution u(x, t) at a specific time_slice.
    Compares DGM prediction with analytical solution if provided.
    """
    if plot_params is None:
        plot_params = {}
    pde_params = plot_params.get('analytical_params')
    num_points = 200
    x_tensor = torch.linspace(domain_bounds[0][0], domain_bounds[0][1], num_points).reshape(-1, 1)
    x_torch = x_tensor.float().to(solver_instance.device)

    # Create time tensor for the specific time_slice
    t_torch = torch.full_like(x_torch[:, 0:1], time_slice).to(solver_instance.device)

    solver_instance.model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        u_pred_torch = solver_instance.predict(x_torch, t_torch)
    u_pred_np = u_pred_torch.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(x_tensor, u_pred_np, label='DGM Solution', color='blue')

    if analytical_solution_func:
        # Pass time_coord_np to the analytical solution function
        t_np = torch.full_like(x_tensor, time_slice)
        u_analytical_np = analytical_solution_func(x_tensor, t_torch,pde_params)
        plt.plot(x_tensor, u_analytical_np, label='Analytical Solution', linestyle='--', color='red')

    title = plot_params.get('title', f"1D PDE Solution at t={time_slice:.2f}")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d_solution_surface(solver_instance, domain_bounds, analytical_solution_func=None, time_slice=0.0,
                             plot_params=None):
    """
    Plots the 2D solution u(x,y,t) as a 3D surface at a specific time_slice.
    Compares DGM prediction with analytical solution if provided (as a wireframe).
    """
    if plot_params is None:
        plot_params = {}
    title = plot_params.get('title', f"2D PDE Solution Surface at t={time_slice:.2f}")

    num_points = 50
    x_coords = np.linspace(domain_bounds[0][0], domain_bounds[0][1], num_points)
    y_coords = np.linspace(domain_bounds[1][0], domain_bounds[1][1], num_points)
    X, Y = np.meshgrid(x_coords, y_coords)

    coords_flat_np = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    coords_flat_torch = torch.from_numpy(coords_flat_np).float().to(solver_instance.device)

    t_torch = torch.full_like(coords_flat_torch[:, 0:1], time_slice).to(solver_instance.device)

    solver_instance.model.eval()
    with torch.no_grad():
        u_pred_flat = solver_instance.predict(coords_flat_torch, t_torch).cpu().numpy()
    U_pred = u_pred_flat.reshape(num_points, num_points)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot DGM Solution
    surf_dgm = ax.plot_surface(X, Y, U_pred, cmap='Accent_r', edgecolor='none', alpha=0.8)

    # Plot Analytical Solution if provided
    if analytical_solution_func:
        x_flat_np = coords_flat_np[:, 0]  # (2500,)
        t_flat_np = coords_flat_np[:, 1]  # (2500,)

        u_analytical_flat = analytical_solution_func(x_flat_np, t_flat_np, plot_params.get('analytical_params', {}))
        U_analytical = u_analytical_flat.reshape(num_points, num_points)

        # Plot analytical solution as a wireframe for visibility
        ax.plot_wireframe(X, Y, U_analytical, color='blue', linewidth=0.5)
        # Create a legend with custom patches to represent the surfaces/wireframes
        ax.legend(
            handles=[Patch(color='red', label='DGM Solution'), Patch(color='blue', label='Analytical Solution')],
            loc='best')
    else:

        # If no analytical solution, just label the DGM solution
        ax.legend(handles=[Patch(color='red', label='DGM Solution')], loc='best')

    fig.colorbar(surf_dgm, ax=ax, shrink=0.5, aspect=5)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u(x,y,t)')
    plt.show()


def plot_3d_solution_slice(solver_instance, fixed_coord_dim, fixed_coord_val, domain_bounds, time_slice=0.0,
                           title="3D PDE Solution Slice"):
    """
    Plots a 2D slice of a 3D solution u(x,y,z,t) at a specific fixed spatial coordinate and time_slice.
    (Analytical comparison for 2D/3D slices is more complex and not included here for generality).
    """
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

    # Create time tensor for the specific time_slice
    t_torch = torch.full_like(coords_flat_torch[:, 0:1], time_slice).to(solver_instance.device)

    solver_instance.model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        u_pred_flat = solver_instance.predict(coords_flat_torch, t_torch).cpu().numpy()

    U_pred_slice = u_pred_flat.reshape(num_points, num_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(C1, C2, U_pred_slice, cmap='Accent_r', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    fixed_dim_label = ['X', 'Y', 'Z'][fixed_coord_dim]
    varying_dim1_label = ['X', 'Y', 'Z'][varying_dims[0]]
    varying_dim2_label = ['X', 'Y', 'Z'][varying_dims[1]]

    ax.set_title(f"{title}\n(Fixed {fixed_dim_label}={fixed_coord_val:.2f}, at t={time_slice:.2f})")
    ax.set_xlabel(varying_dim1_label)
    ax.set_ylabel(varying_dim2_label)
    ax.set_zlabel('u')
    plt.show()
#-------------------


def plot_2d_solution_over_time_subplots(solver_instance, domain_bounds, time_slices: list, plot_params=None):
    """
    Plots the 2D solution u(x,y,t) as 3D surfaces in subplots for various time_slices.
    """
    if plot_params is None:
        plot_params = {}

    num_time_slices = len(time_slices)
    if num_time_slices == 0:
        print("No time slices provided for plotting.")
        return

    cols = min(3, num_time_slices)
    rows = math.ceil(num_time_slices / cols)

    fig = plt.figure(figsize=(cols * 6, rows * 5))  # Adjust figure size based on subplot count
    fig.suptitle(plot_params.get('main_title', "2D PDE Solution over Time"), fontsize=16)

    num_points = 50
    x_coords = np.linspace(domain_bounds[0][0], domain_bounds[0][1], num_points)
    y_coords = np.linspace(domain_bounds[1][0], domain_bounds[1][1], num_points)
    X, Y = np.meshgrid(x_coords, y_coords)

    coords_flat_np = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    coords_flat_torch = torch.from_numpy(coords_flat_np).float().to(solver_instance.device)

    solver_instance.model.eval()

    for i, time_slice in enumerate(time_slices):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        t_torch = torch.full_like(coords_flat_torch[:, 0:1], time_slice).to(solver_instance.device)

        with torch.no_grad():
            u_pred_flat = solver_instance.predict(coords_flat_torch, t_torch).cpu().numpy()
        U_pred = u_pred_flat.reshape(num_points, num_points)

        surf = ax.plot_surface(X, Y, U_pred, cmap='viridis', edgecolor='none')
        ax.set_title(f"t={time_slice:.2f}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('u')
        ax.tick_params(axis='both', which='major', labelsize=8)  # Smaller tick labels
        ax.set_zticklabels([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_3d_solution_slice_over_time_subplots(solver_instance, fixed_coord_dim, fixed_coord_val, domain_bounds,
                                              time_slices: list, plot_params=None):
    """
    Plots a 2D slice of a 3D solution u(x,y,z,t) in subplots for various time_slices.
    """
    if plot_params is None:
        plot_params = {}

    if solver_instance.spatial_dimension != 3:
        print("This function is for 3D solutions only.")
        return

    num_time_slices = len(time_slices)
    if num_time_slices == 0:
        print("No time slices provided for plotting.")
        return

    cols = min(3, num_time_slices)
    rows = math.ceil(num_time_slices / cols)

    fig = plt.figure(figsize=(cols * 6, rows * 5))

    fixed_dim_label = ['X', 'Y', 'Z'][fixed_coord_dim]
    fig.suptitle(
        plot_params.get('main_title', f"3D PDE Slice (Fixed {fixed_dim_label}={fixed_coord_val:.2f}) over Time"),
        fontsize=16)

    varying_dims = [i for i in range(3) if i != fixed_coord_dim]
    num_points = 50
    coord1_vals = np.linspace(domain_bounds[varying_dims[0]][0], domain_bounds[varying_dims[0]][1], num_points)
    coord2_vals = np.linspace(domain_bounds[varying_dims[1]][0], domain_bounds[varying_dims[1]][1], num_points)
    C1, C2 = np.meshgrid(coord1_vals, coord2_vals)

    coords_flat_np_base = np.zeros((num_points * num_points, 3))
    coords_flat_np_base[:, varying_dims[0]] = C1.flatten()
    coords_flat_np_base[:, varying_dims[1]] = C2.flatten()
    coords_flat_np_base[:, fixed_coord_dim] = fixed_coord_val  # This dimension is fixed

    coords_flat_torch_base = torch.from_numpy(coords_flat_np_base).float().to(solver_instance.device)

    solver_instance.model.eval()  # Set model to evaluation mode

    for i, time_slice in enumerate(time_slices):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        t_torch = torch.full_like(coords_flat_torch_base[:, 0:1], time_slice).to(solver_instance.device)

        with torch.no_grad():
            u_pred_flat = solver_instance.predict(coords_flat_torch_base, t_torch).cpu().numpy()
        U_pred_slice = u_pred_flat.reshape(num_points, num_points)

        surf = ax.plot_surface(C1, C2, U_pred_slice, cmap='viridis', edgecolor='none')
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # Optional colorbar

        ax.set_title(f"t={time_slice:.2f}")
        ax.set_xlabel(['X', 'Y', 'Z'][varying_dims[0]])
        ax.set_ylabel(['X', 'Y', 'Z'][varying_dims[1]])
        ax.set_zlabel('u')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_zticklabels([])  # Hide z-tick labels for cleaner look

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_3d_solution_spatial_slices_subplots(solver_instance, fixed_coord_dim, fixed_coord_vals: list, domain_bounds,
                                             plot_params=None):
    """
    Plots 2D slices of a 3D solution u(x,y,z) in subplots for various fixed spatial coordinate values.
    This is suitable for elliptic (steady-state) 3D problems.
    """
    if plot_params is None:
        plot_params = {}

    if solver_instance.spatial_dimension != 3:
        print("This function is for 3D solutions only.")
        return

    num_slices = len(fixed_coord_vals)
    if num_slices == 0:
        print("No fixed coordinate values provided for plotting slices.")
        return

    cols = min(3, num_slices)
    rows = math.ceil(num_slices / cols)

    fig = plt.figure(figsize=(cols * 6, rows * 5))

    fixed_dim_label = ['X', 'Y', 'Z'][fixed_coord_dim]
    fig.suptitle(plot_params.get('main_title', f"3D PDE Solution Slices (Fixed {fixed_dim_label})"), fontsize=16)

    varying_dims = [i for i in range(3) if i != fixed_coord_dim]
    num_points = 50
    coord1_vals = np.linspace(domain_bounds[varying_dims[0]][0], domain_bounds[varying_dims[0]][1], num_points)
    coord2_vals = np.linspace(domain_bounds[varying_dims[1]][0], domain_bounds[varying_dims[1]][1], num_points)
    C1, C2 = np.meshgrid(coord1_vals, coord2_vals)

    # Time coordinate is fixed at 0 for elliptic problems
    t_fixed_torch = torch.tensor([[0.0]]).float().to(solver_instance.device)

    solver_instance.model.eval()  # Set model to evaluation mode

    for i, fixed_val in enumerate(fixed_coord_vals):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        coords_flat_np = np.zeros((num_points * num_points, 3))
        coords_flat_np[:, varying_dims[0]] = C1.flatten()
        coords_flat_np[:, varying_dims[1]] = C2.flatten()
        coords_flat_np[:, fixed_coord_dim] = fixed_val  # This dimension is fixed for the slice

        coords_flat_torch = torch.from_numpy(coords_flat_np).float().to(solver_instance.device)

        # Predict the solution
        # Since time is constant for elliptic, expand t_fixed_torch to match coords_flat_torch batch size
        t_batch = t_fixed_torch.expand(coords_flat_torch.shape[0], -1)

        with torch.no_grad():
            u_pred_flat = solver_instance.predict(coords_flat_torch, t_batch).cpu().numpy()
        U_pred_slice = u_pred_flat.reshape(num_points, num_points)

        surf = ax.plot_surface(C1, C2, U_pred_slice, cmap='viridis', edgecolor='none')
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5) # Optional colorbar

        ax.set_title(f"{fixed_dim_label}={fixed_val:.2f}")
        ax.set_xlabel(['X', 'Y', 'Z'][varying_dims[0]])
        ax.set_ylabel(['X', 'Y', 'Z'][varying_dims[1]])
        ax.set_zlabel('u')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_zticklabels([])  # Hide z-tick labels for cleaner look

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
