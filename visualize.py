import numpy as np
import matplotlib.pyplot as plt
import torch

import sampling


def plot_1d_(solver, alpha, domain_bound,time_bound):
    """
    plots 1D Solution against the analystic solution at various times
    """

    x_test = np.linspace(domain_bound[0][0],domain_bound[0][1],100)[:, None]
    t_test = np.linspace(time_bound[0],time_bound[1],5)[:, None]

    plt.figure(figsize=(10,6))
    plt.title("Deep Galerkin Solution vs Analytic Solution")
    plt.xlabel("Spatial Coordinates [x]")
    # u--> Temperature (Heat)
    plt.ylabel("Solution [u]")

    for i, t_val in enumerate(t_test):
        x_tensor = torch.tensor(x_test,dtype=torch.float32)
        t_tensor = torch.full_like(x_tensor,float(t_val),dtype=torch.float32)


        solver.model.eval()
        with torch.no_grad():
            predict = solver.predict(x_tensor,t_tensor).cpu().numpy()
        solver.model.train()

        u_analytic = sampling.analytic_func(x_tensor,t_tensor, alpha)
        plt.plot(x_test, predict, label=f'DGM (t={t_val[0]:.2f})',linestyle='-')
        plt.plot(x_test,u_analytic, label=f'Analytic (t={t_val[0]:.2f})',linestyle='--')

    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_2d_(solver, domain_bound,time_point):
    """
    plots 2D Solution against the analystic solution at a specific time
    """
    nx, ny = 100,100
    x_np = np.linspace(domain_bound[0][0],domain_bound[0][1],nx)
    y_np = np.linspace(domain_bound[1][0],domain_bound[1][1],ny)
    X_mesh, Y_mesh = np.meshgrid(x_np, y_np)

    #Flatten grid for network input (N,2)
    Spatial_coords_flat_np = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T
    t_flat_np = np.full((nx * ny,1), time_point)

    Spatial_coords_flat_tensor = torch.tensor(Spatial_coords_flat_np,dtype=torch.float32).to(solver.device)
    t_flat_tensor = torch.tensor(t_flat_np,dtype=torch.float32).to(solver.device)

    solver.model.eval()
    with torch.no_grad():
        predict = solver.predict(Spatial_coords_flat_tensor,t_flat_tensor)
    solver.model.train()
    predict_np = predict.cpu().detach().numpy()
    U_pred = predict_np.reshape(nx , ny)

    plt.figure(figsize=(10,6))
    contour = plt.contour(X_mesh, Y_mesh, U_pred,levels = 50)
    plt.colorbar(contour,label = 'u')
    plt.xlabel('Spatial Coordinates [x]')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def plot_3d_(solver,domain_bound,time_point,fixed_coord_dim, fixed_coord_val):
    """
    Plots 3D Solution against the analystic solution at various times
    """

    if solver.spatial_dim != 3:
        raise ValueError('Only 3D Spatial Dimension supported')

    plot_dims = [i for i in range(3) if i != fixed_coord_dim]

    n_points_slice = 40
    var1_val = np.linspace(domain_bound[0][0],domain_bound[0][1],n_points_slice)
    var2_val = np.linspace(domain_bound[1][0],domain_bound[1][1],n_points_slice)
    var1_mesh, var2_mesh = np.meshgrid(var1_val, var2_val)

    spatial_coord_flat_np = np.zeros((n_points_slice * n_points_slice,3))

    spatial_coord_flat_np[:,plot_dims[0]] = var1_mesh.ravel()
    spatial_coord_flat_np[:,plot_dims[1]] = var2_mesh.ravel()

    spatial_coord_flat_np[:fixed_coord_dim] = fixed_coord_val

    t_flat_np = np.full((n_points_slice * n_points_slice,1),time_point)

    spatial_coord_flat_tensor = torch.tensor(spatial_coord_flat_np,dtype=torch.float32).to(solver.device)
    t_flat_tensor = torch.tensor(t_flat_np,dtype=torch.float32).to(solver.device)

    solver.model.eval()
    with torch.no_grad():
        predict = solver.predict(spatial_coord_flat_tensor,t_flat_tensor)
    solver.model.train()

    predict_np = predict.cpu().detach().numpy()
    U_pred = predict_np.reshape(n_points_slice, n_points_slice)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(var1_mesh, var2_mesh, U_pred, cmap='viridis',edgecolor='k')

    ax.set_xlabel(['Spatial Coordinates [x] [y] [z]'][plot_dims[0]])
    ax.set_ylabel(['Spatial Coordinates [x] [y] [z]'][plot_dims[1]])
    ax.set_zlabel('u')
    ax.set_title(f'3d Solution at t={time_point:.2f}\n Fixed{['x','y','z'][fixed_coord_dim]}={fixed_coord_val:.2f}')
    ax.view_init(elev=20, azim=60)
    plt.tight_layout()
    plt.show()
