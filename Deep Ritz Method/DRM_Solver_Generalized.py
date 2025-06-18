import torch
from torch import nn, Tensor

from NeuralNetwork import NeuralNetwork
from sampling import generate_pde_points, generate_bc_points
from utility_functions import get_device


class DRM_Solver_Generalized():
    def __init__(self, spatial_dimension, layer_sizes, activation = nn.Tanh):
        if spatial_dimension not in [1, 2, 3]:
            raise ValueError("Spatial dimension must be 1 or 2 or 3")
        self.spatial_dimension = spatial_dimension
        input_features = spatial_dimension + 1
        self.model = NeuralNetwork(input_features, layer_sizes, activation)
        self.domain_bounds = None

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        self.losses = {}
        self.lambda_bc = 1000.0
        print(f'Using device: {self.device}')

    def predict(self,spatial_coords, time_coord):
        spatial_coords = spatial_coords.to(self.device)
        time_coord = time_coord.to(self.device)

        # Concatenating the spatial and time coordinates for neural input
        x_t_coord = torch.cat([spatial_coords, time_coord], dim=1)
        return self.model(x_t_coord)

    def compute_u(self,  x_pde: torch.Tensor) :

        t_zero = torch.zeros_like(x_pde[:, 0:1])
        X_all = torch.cat([x_pde, t_zero], dim=1)
        X_all.requires_grad_(True)
        u_pred = self.model(X_all)

        grads = torch.autograd.grad(outputs=u_pred,inputs=X_all,grad_outputs=torch.ones_like(u_pred),create_graph=True,
                                        retain_graph=True)[0]

        grad_u_spatial = grads[:, :self.spatial_dimension]

        return u_pred, grad_u_spatial

    def _compute_bc_loss(self, boundary_condition_func: callable, x_bc: torch.Tensor,domain_bound) -> torch.Tensor:
        """
        Computes the boundary condition loss (soft constraint) for DRM.
        """
        # Time is always 0 for DRM BC evaluation
        t_zero = torch.zeros_like(x_bc[:, 0:1])

        X_all_bc = torch.cat([x_bc, t_zero], dim=1)
        u_pred_bc = self.model(X_all_bc)

        u_bc_target = boundary_condition_func(x_bc,domain_bound[0][1])

        loss_bc = torch.mean((u_pred_bc - u_bc_target) ** 2)
        return loss_bc

    def compute_loss(self,domain_bound, pde_residual_func: callable, pde_parameters: dict,
                     X_pde, T_pde,
                     X_bc, U_bc,boundary_condition_func
                     ) :
        """
        Computes the total loss for DRM by combining functional and BC losses.
        Note: IC is not applicable for steady-state DRM and X_ic will be None.
        """

        u_pred, grad_u_spatial = self.compute_u(X_pde)

        integral_values = pde_residual_func(u_pred=u_pred,grad_u_spatial=grad_u_spatial,x=X_pde,
                            params=pde_parameters )
        loss_functional = torch.mean(integral_values)

        loss_bc = self._compute_bc_loss(boundary_condition_func,X_bc,domain_bound)

        total_loss = loss_functional + self.lambda_bc * loss_bc

        loss_details = {'total': total_loss.item(), 'functional': loss_functional.item(),
                        'bc': loss_bc.item()}
        return total_loss, loss_details

    def train(self, pde_residual_func, pde_parameters, domain_bound, boundary_condition_func, num_pde_points,
              num_bc_points, epochs, learning_rate):
        self.domain_bound = domain_bound

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # --- Training Loop ---
        print(f'\n --- Training Loop for DGM for {self.spatial_dimension} spatial dimensional pde on {self.device}')

        for epoch in range(epochs):
            self.model.train()

            x_pde, t_pde = generate_pde_points(num_pde_points, domain_bound)
            x_bc, u_bc = generate_bc_points(num_bc_points, domain_bound, self.spatial_dimension,
                                            boundary_condition_func)

            optimizer.zero_grad()

            total_loss, loss_details = self.compute_loss(domain_bound,
                pde_residual_func, pde_parameters,
                x_pde, t_pde,
                x_bc, u_bc,boundary_condition_func
            )
            total_loss.backward()
            optimizer.step()

            for key, value in loss_details.items():
                if key not in self.losses:
                    self.losses[key] = []
                self.losses[key].append(value)

            if (epoch + 1) % 100 == 0:
                loss_str = f"Epoch {epoch + 1}/{epochs}: Total Loss={total_loss.item():.4e}"
                for key, value in loss_details.items():
                    loss_str += f", {key.replace('loss_', '').upper()} Loss={value:.4e}"
                print(loss_str)
        print("Training complete.")



