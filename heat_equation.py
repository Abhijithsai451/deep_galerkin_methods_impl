import torch
from torch import nn

from NeuralNetwork import NeuralNetwork


class HeatEquation:
    """
        A generalized Deep Galerkin Method (DGM) solver for the heat equation
        in 1D, 2D, or 3D using PyTorch.
        The heat equation is: du/dt = alpha * (sum of second spatial derivatives)
    """
    def __init__(self, spatial_dimension, layer_sizes, activations = nn.Tanh):
        """
         Initializes the DGM solver
         Args:
             spatial_dimension (int): The number of spatial dimensions (1D, 2D or 3D)
             layer_sizes (list): List of layers specifing the number of neurons in each layer
             activations inherits (nn.Module): Activation function used for the layers
        """
        if spatial_dimension not in [1,2,3]:
            raise ValueError("spatial_dimension must be 1 or 2 or 3")

        self.spatial_dimension = spatial_dimension
        input_features = spatial_dimension +1
        self.model = NeuralNetwork(input_features, layer_sizes, activations)

        self.domain_bounds = None
        self.time_bounds = None

        # Device Configuration
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Using Device: {self.device}")

    def predict(self, spatial_coord, time_coord):
        """

        :param spatial_coord: Tensor which holds the spatial coordinates
        :param time_coord: Tensor which holds the time coordinates
        :return: predicted solution "u(x,1)"

        """
        spatial_coord = spatial_coord.to(self.device)
        time_coord = time_coord.to(self.device)
        input_tensor = torch.cat((spatial_coord, time_coord), dim = 1)

        return self.model(input_tensor)

    def pde_residual(self, spatial_coord, time_coord, alpha):
        """
        Calculates the residual of the heat equation (du/dt - alpha[Thermal Diffusibility] * Laplacial[∆^2] *(u))

        :param spatial_coord: Tensor which holds the spatial coordinates
        :param time_coord: Tensor which holds the time coordinates
        :return: The PDE residual

        """
        spatial_coord.requires_grad = True
        time_coord.requires_grad = True

        u = self.predict(spatial_coord, time_coord)
        # Computing the First derivative with respect to time
        u_t = torch.autograd.grad(u,time_coord, grad_outputs=torch.ones_like(u),create_graph=True)[0]

        # Computing the second derivative with respect to spacial coordinates (x_i) as per the dimensions
        u_xx_sum = 0.0
        for i in range(self.spatial_dimension):
            coord_i = spatial_coord[:, i:i+1]

            # Compute 1 st derivative wrt x_i (du/dx_i)
            u_i = torch.autograd.grad(u_xx_sum, coord_i, grad_outputs=torch.ones_like(u), create_graph=True,
                                      retain_graph=True)[0]
            u_ii = torch.autograd.grad(u_i, coord_i, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            u_xx_sum += u_ii

        # Heat Equation: du/dt = alpha * (sum of second spatial derivatives)

        pde_eqn = u_t - alpha * u_xx_sum

        return pde_eqn

    def compute_loss(self, alpha, X_pde, T_pde, X_ic, U0_ic, X_bc, T_bc, U_bc):
        """
        :param alpha: Themal Diffisivity
        :param X_pde: Spatial Coordinates for PDE
        :param T_pde: Time Coordinates for PDE
        :param X_ic: Spatial Coordinates for Initial Conditions
        :param U0_ic: True initial value at X_ic
        :param X_bc: Spatial Coordinates for Boundary Conditions
        :param T_bc: Time Coordinates for Boundary Conditions

        :return:
               Tuple (Total loss, PDE loss, Initial conditions loss, Boundary conditions loss)
        """
        T_bc = T_bc.to(self.device)
        U_bc = U_bc.to(self.device)
        U0_ic = U0_ic.to(self.device)

        # PDE Residual Loss
        pde_res = self.pde_residual(X_pde, T_pde, alpha)
        loss_pde = torch.mean(pde_res ** 2)

        # Initial Condition Loss
        t_ic = (torch.ones_like(X_ic[:, 0:1]) * self.time_bounds[0]).to(self.device)
        u_pred_ic = self.predict(X_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - U0_ic)**2)

        # Boundary Condition Loss
        u_pred_bc = self.predict(X_bc, T_bc)
        loss_bc = torch.mean((u_pred_bc - U_bc)**2)

        total_loss = loss_pde + loss_ic + loss_bc
        return total_loss, pde_res, loss_pde, loss_ic, loss_bc

    def train(self, alpha, domain_bound, time_bound, ini_cond, bound_cond, num_pde_points,
              num_ic_points, num_bc_points, epochs, learning_rate):
        """
        :param alpha: Thermal Diffisivity
        :param domain_bound: List of [min, max] for each spatial dimension,
                                  e.g., [[0.0, 1.0]] for 1D, [[0.0, 1.0], [0.0, 1.0]] for 2D.
        :param time_bound (list): [start_time, end_time].
        :param ini_cond (callable): Function u0(coords_spatial) returning initial values.
        :param bound_cond: Function u_b(coords_spatial, t) returning boundary values.
        :param num_pde_points: Number of sampling points for PDE residual
        :param num_ic_points: Number of sampling points for Initial Conditions
        :param num_bc_points: Number of sampling points for Boundary Conditions
        :param epochs: Number of training epochs
        :param learning_rate: Hyper Parameter
        """

        self.domain_bounds = domain_bound
        self.time_bounds = time_bound

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)























