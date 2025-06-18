import torch
from torch import nn

from sampling import *
from NeuralNetwork import NeuralNetwork

class DGM_Solver_Generalized():
    def __init__(self, spatial_dimension, layer_sizes, activation = nn.Tanh ):

        if spatial_dimension not in [1,2,3]:
            raise ValueError("Spatial dimension must be 1 or 2 or 3")
        self.spatial_dimension = spatial_dimension
        input_features = spatial_dimension + 1
        self.model = NeuralNetwork(input_features, layer_sizes, activation)
        self.domain_bounds = None

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)

        print(f'Using device: {self.device}')

    def predict(self,spatial_coords, time_coord):
        spatial_coords = spatial_coords.to(device)
        time_coord = time_coord.to(device)

        # Concatenating the spatial and time coordinates for neural input
        x_t_coord = torch.cat([spatial_coords, time_coord], dim=1)
        return self.model(x_t_coord)

    def compute_loss(self, pde_residual_func: callable, pde_parameters: dict,
                     X_pde, T_pde,
                     X_bc, U_bc,
                     X_ic, U_ic):
        """
        Calculates the losses of the pde and loss at the boundary conditions
        loss we calculate using Mean Square Error
        """

        pde_res = pde_residual_func(self,X_pde, T_pde, **pde_parameters)
        loss_pde = torch.mean(pde_res ** 2)

        u_pred_bc = self.predict(X_bc, torch.zeros_like(X_bc[:, 0:1]))
        loss_bc = torch.mean((u_pred_bc - U_bc) ** 2)

        loss_ic = torch.tensor(0.0).to(device)
        if X_ic is not None and U_ic is not None:
            u_pred_ic = self.predict(X_ic, torch.zeros_like(X_ic[:, 0:1]))
            loss_ic = torch.mean((u_pred_ic - U_ic) ** 2)

        total_loss = loss_pde + loss_bc + loss_ic
        return total_loss, loss_pde, loss_bc, loss_ic

    """def train(self, alpha, domain_bound, time_bound, num_pde_points, num_ic_points, num_bc_points, epochs, learning_rate,
              pde_residual_func: callable, pde_parameters: dict):
        """
    def train(self, pde_residual_func: callable, pde_parameters: dict,
              domain_bound: list,
              boundary_condition_func: callable,
              num_pde_points: int, num_bc_points: int,
              epochs: int, learning_rate: float,
              initial_condition_func: callable = None, num_ic_points: int = 0):
        self.domain_bound = domain_bound

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # --- Training Loop ---
        print(f'\n --- Training Loop for DGM for {self.spatial_dimension} spatial dimensional pde on {self.device}')

        for epoch in range(epochs):
            self.model.train()

            x_pde, t_pde = generate_pde_points(num_pde_points, domain_bound)
            x_bc, u_bc = generate_bc_points(num_bc_points, domain_bound,self.spatial_dimension, boundary_condition_func)
            x_ic, u_ic = None, None
            if initial_condition_func is not None and num_ic_points > 0:
                x_ic, u_ic = generate_ic_points(num_ic_points)

            optimizer.zero_grad()

            total_loss , loss_pde, loss_bc, loss_ic = self.compute_loss(
                pde_residual_func, pde_parameters,
                x_pde, t_pde,
                x_bc, u_bc,
                x_ic, u_ic
            )
            total_loss.backward()
            optimizer.step()

            if epoch % (epochs // 10 if epochs > 10 else 1) == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch}/{epochs}: Total Loss={total_loss.item():.4e}, PDE Loss={loss_pde.item():.4e}, "
                    f"BC Loss={loss_bc.item():.4e}, IC Loss={loss_ic.item():.4e}")
        print("Training complete.")


