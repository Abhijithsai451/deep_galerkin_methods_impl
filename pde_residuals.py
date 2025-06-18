import torch
from sympy.vector import Laplacian


def calculate_laplacian(predict, spatial_dims, spatial_coord, time_coord):
    """
    Calculates the time drivetive du/dt and the laplacian (sum of partials of u)
    """
    spatial_coord.requires_grad_(True)
    time_coord.requires_grad_(True)

    u = predict(spatial_coord,time_coord)

    # 1. Computing the First Derivatives wrt time (du/dt)
    u_t = torch.autograd.grad(u,time_coord,grad_outputs=torch.ones_like(u),create_graph=True)[0]

    # 2. Computing the Second Derivatives or Laplacian
    du_dx_all = torch.autograd.grad(u,spatial_coord,grad_outputs=torch.ones_like(u),create_graph=True)[0]

    u_xx_sum = torch.zeros_like(u)
    for i in range(spatial_dims):
        # Here we extract the gradient wrt the i-th spatial dimension
        u_x_i = du_dx_all[:, i:i+1]

        # compute the 2nd derivative wrt x_i (d2u/dx_i^2)
        u_ii = torch.autograd.grad(u_x_i,spatial_coord,grad_outputs=torch.ones_like(u_x_i),create_graph=True)[0][:, i:i+1]

        u_xx_sum = u_xx_sum + u_ii

    return u_t, u_xx_sum

def heat_equation_residual(solver, spatial_coord, time_coord, **pde_params):
    """
    Defines the residual for the heat equation: du/dt - alpha * Laplacian(u) =  0
    : spatial_coord:  Spatial Coordinates
    : time_coords: Time Coordinates
    : pde_params: contains Alpha and other parameters
    """

    alpha = pde_params['alpha']
    u_t , laplacian_u = calculate_laplacian(solver.predict, solver.spatial_dimension, spatial_coord, time_coord)

    # Assuming the Heat Equation: du/dt - alpha * Laplacian(u) = 0
    heat_equation_residual = u_t - alpha * laplacian_u

    # If Souce term in available then the Heat equation will be equal to the source  q(x,t)

    return heat_equation_residual

def poisson_equation_residual(solver, spatial_coord, time_coord, **pde_params):
    """
    Defines the residual for Poisson's Equation: Laplacian(u) =  f
    : spatial_coord:  Spatial Coordinates
    : time_coord:  Time Coordinates
    : pde_params:  Contains Alpha and other parameters
    """

    # For Poisson we only need to calculate the Laplacian. du/dt term will be zero
    u_t, laplacian_u = calculate_laplacian(solver.predict, solver.spatial_dimension, spatial_coord, time_coord)

    f_val = pde_params['f_func'](spatial_coord)

    poisson_equation_residual = laplacian_u - f_val
    return poisson_equation_residual

#
def poisson_energy_functional(u_pred: torch.Tensor, grad_u_spatial: torch.Tensor,
                                                x: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Defines the integrand of the energy
    u_pred: Predicted Solution value u(x) . (N-1)
    grad_u_spatial: grad of u wrt spatial coordinates. (N, spatial_dimension)
    x: Spatial Coordinates (N, spatial_dimension)
    params: contains callable parameters in the form of Dictionary
    """

    f_func = params.get('f_func')
    if f_func is None:
        raise ValueError("Poisson energy functional parameters must include 'f_func' for the source term.")

    grad_u_squared = torch.sum(grad_u_spatial ** 2, dim=1, keepdim=True)
    source_f = f_func(x)

    integrand = 0.5 * grad_u_squared + source_f * u_pred

    return integrand