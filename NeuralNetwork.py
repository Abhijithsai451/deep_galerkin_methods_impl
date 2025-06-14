import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class NeuralNetwork(nn.Module):
    def __init__(self, input, layer_sizes, activation):
        super(NeuralNetwork,self).__init__()

        layers = []

        # Input Layer
        layers.append(nn.Linear(input, layer_sizes[0]))
        layers.append(activation())

        # Hidden Layers
        for i in range(len(layer_sizes) -1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(activation())

        # Output Layer
        layers.append(nn.Linear(layer_sizes[-1], 1))

        self.network = nn.Sequential(*layers)

    def forward(self,input_tensor: torch.Tensor) -> torch.Tensor:
        # Input_tensor = Concatenated Spatial and Time Coordinates (N, Spatial_dim +1)

        output_tensor = self.network(input_tensor)
        return output_tensor

