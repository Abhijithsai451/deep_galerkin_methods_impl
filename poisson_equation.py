import torch
from torch import nn

from NeuralNetwork import NeuralNetwork


class EllipticPDE:
    def __init__(self, spatial_dimension, layer_sizes, activation=nn.Tanh):
        """
        Initializes the Solver
        :spatial_dimension: Number of dimensions
        :layer_sizes: List containing the size (neurons) of each layer
        :activation: Activation function (Tanh)
        """

        if spatial_dimension not in [1,2,3]:
            raise ValueError("Spatial dimension must be 1,2 or 3")

        self.spatial_dimension = spatial_dimension
        input_features = spatial_dimension
        self.model = NeuralNetwork(input_features, layer_sizes, activation)
        self.domain_bound = None

        print("[Debug] Initializing the Device ")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[Debug] Device Initialized, Using the device: {self.device}")

    def predict(self, spatial_coords):
        spatial_coords = spatial_coords.to(self.device)
        return self.model(spatial_coords)

    def pde_residual(self, spatial_coord, time_coord, alpha):
        """
        :param spatial_coord: Tensor which holds the spatial coordinates
        :param time_coord: Tensor which holds the time coordinates
        :return: The PDE residual

        """
        spatial_coord.requires_grad_(True)
        time_coord.requires_grad_(True)

        u = self.predict(spatial_coord, time_coord)
        # 1. Computing the First derivative with respect to time (du/dt)
        u_t = torch.autograd.grad(u,time_coord, grad_outputs=torch.ones_like(u),create_graph=True)[0]

        # 2. Compute the Laplasian
        du_dx_all = torch.autograd.grad(u, spatial_coord,
                                       grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]

        u_xx_sum = 0.0
        for i in range(self.spatial_dimension):
            u_x_i = du_dx_all[:, i:i+1]

            # Computing 2nd derivative wrt x_i (du/dx_i)
            du_dxx_all = torch.autograd.grad(u_x_i, spatial_coord, grad_outputs=torch.ones_like(u_x_i), create_graph=True,
                                      retain_graph=True)[0]
            # Compute 2nd derivative with respect to x_i (d2u/dx_i^2)
            u_ii = du_dxx_all[:, i:i+1]

            u_xx_sum += u_ii

        # Heat Equation: du/dt = alpha * (sum of second spatial derivatives)

        pde_eqn = u_t - alpha * u_xx_sum

        return pde_eqn

    def compute_spatial_derivatives(self, spatial_coords):
        """
        Computes the first and second order spatial derivatives of given coordinates. Here it is 'u'
        """

        spatial_coords.requires_grad_(True)
        u = self.predict(spatial_coords)

        derivatives = {
            'u': u,
            'u_x': None, 'u_y': None, 'u_z': None,
            'u_xx_sum': torch.zeros_like(u)
        }
        spatial_coords_list = [spatial_coords[:, i:i +1] for i in range(self.spatial_dimension)]
        coord_names = ['u_x', 'u_y', 'u_z']

        for i, coord_i in enumerate(spatial_coords_list):
            """
            1. Compute the first derivative wrt x_i --> (du/dx_i)
            2. Create_graph=True as we will compute the second derivatives from this
            3. Retain_graph = True as we can recall 'u' multiple times 
            4. Compute the Second Derivative
            """
            # Computing the First Derivative
            u_i = torch.autograd.grad(u,coord_i,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]
            derivatives[coord_names[i]] = u_i

            # Computing teh second derivative
            u_ii = torch.autograd.grad(u_i, coord_i, grad_outputs=torch.ones_like(u_i),create_graph=True)[0]
            derivatives['u_xx_sum'] += u_ii

        return derivatives
    def compute_loss(self,alpha, X_pde,X_bc,U_bc):
        """
        Calculates the losses of the pde and loss at the boundary conditions
        loss we calculate using Mean Square Error
        """

        derivateives = self.compute_spatial_derivatives(X_pde)
        pde_res = self.pde_residual(X_pde, T_pde, alpha)

























