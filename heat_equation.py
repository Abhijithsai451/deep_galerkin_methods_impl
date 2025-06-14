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




