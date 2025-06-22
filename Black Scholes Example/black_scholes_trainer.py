import torch
from torch import nn
from tqdm import tqdm
from NeuralNetwork import NeuralNetwork
from utility_functions import get_device
from sampling import generate_pde_points, generate_bc_points, generate_ic_points

device = get_device()
class BS_Solver_Generalized():
    def __init__(self, spatial_dimension, layer_sizes, activation = nn.Tanh ):

        if spatial_dimension not in [1,2,3]:
            raise ValueError("Spatial dimension must be 1 or 2 or 3")
        self.spatial_dimension = spatial_dimension
        input_features = spatial_dimension + 1
        self.model = NeuralNetwork(input_features, layer_sizes, activation)
        self.domain_bounds = None
        self.time_dependent = True
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)

        print(f'Using device: {self.device}')

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))

    def compute_derivatives(self,X) -> tuple:
        """
        Computes the neural network's prediction and its first/second derivatives wrt input spatial coordinates.
        """
        X.requires_grad_(True)
        u_pred = self.model(X)
        # 1st Derivative
        first_deriv = torch.autograd.grad(
            outputs=u_pred, inputs=X,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True)[0]

        # 2nd Derivative -> Diagonal element from the hessian - Laplacian
        second_deriv = []
        for i in range(first_deriv.shape[1]):
            d_u_dxi = first_deriv[:, i:i + 1]
            d2_u_dxi2 = torch.autograd.grad(
                outputs=d_u_dxi, inputs=X,
                grad_outputs=torch.ones_like(d_u_dxi),
                create_graph=True, retain_graph=True
            )[0][:, i:i + 1]
            second_deriv.append(d2_u_dxi2)
        second_deriv = torch.cat(second_deriv, dim=1)

        mixed_deriv = {}
        if self.spatial_dimension > 1:
            spatial_dims_indices = list(range(self.spatial_dimension))
            for i in spatial_dims_indices:
                for j in spatial_dims_indices:
                    if i < j:
                        # d^2 u/ (dx_i dx_j) = d/dx_j (du/dx_i)
                        d2_u_dxi_dxj = torch.autograd.grad(
                            outputs=first_deriv[:, i:i + 1], inputs=X,
                            grad_outputs=torch.ones_like(first_deriv[:, i:i + 1]),
                            create_graph=True, retain_graph=True
                        )[0][:, j:j + 1]
                        mixed_deriv[(i, j)] = d2_u_dxi_dxj
        return u_pred, first_deriv, second_deriv, mixed_deriv

        # For Higher Derivatives

    def compute_pde_loss(self,pde_residual_func: callable, x_pde,
                         pde_params: dict):
        u_pred_pde = self.model(x_pde)
        u_pred, first_deriv, second_deriv, mixed_deriv = self.compute_derivatives(x_pde)

        d_u_dt = first_deriv[:, -1:] if self.time_dependent else None
        spatial_first_deriv = first_deriv[:, :self.spatial_dimension]
        spatial_second_deriv = second_deriv[:, :self.spatial_dimension]

        residual = pde_residual_func(u_pred=u_pred, d_u_dt=d_u_dt, spatial_first_deriv=spatial_first_deriv,
                                     spatial_second_deriv=spatial_second_deriv,spatial_mixed_deriv=mixed_deriv,
                                     x_coords=x_pde[:, :self.spatial_dimension],
                                     t_coords=x_pde[:, -1:] if self.time_dependent else None,
                                     params=pde_params)

        return torch.mean(residual ** 2)

    def compute_bc_loss(self,boundary_condition_func: callable, x_bc):
        """
        Computes the Boundary Condition Loss.
        """
        u_pred_bc = self.model(x_bc)
        u_bc_target = boundary_condition_func(x_bc[:, : self.spatial_dimension],
                                              x_bc[:, -1:] if self.time_dependent else None)
        return torch.mean((u_pred_bc - u_bc_target) ** 2)

    def compute_ic_loss(self,initial_condition_func: callable, x_ic):
        """
        Computes the Initial Condition Loss
        """
        u_pred_ic = self.model(x_ic)
        u_ic_target = initial_condition_func(x_ic[:, :self.spatial_dimension])
        return torch.mean((u_pred_ic - u_ic_target) ** 2)

    def train(self, pde_residual_func: callable, pde_parameters: dict,
              domain_bound: list,
              boundary_condition_func: callable,
              num_pde_points: int, num_bc_points: int,
              epochs: int, learning_rate: float,lambda_bc: float, lambda_ic: float = 0.0,
              initial_condition_func: callable = None, num_ic_points: int = 0):
        self.domain_bound = domain_bound

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        if self.time_dependent and initial_condition_func is None:
            raise ValueError("Initial condition function is required for time dependent problems")
        if pde_parameters is None:
            pde_parameters = {}

        # Extracting the Domain boundaries for points generation
        spatial_domain = self.domain_bound[:self.spatial_dimension]
        time_domain = self.domain_bound[-1] if self.time_dependent else None


        # --- Training Loop ---
        print(f'\n --- Training Loop for DGM for {self.spatial_dimension} spatial dimensional pde on {self.device}')

        for epoch in tqdm(range(epochs), desc = "Training Black Scholes Example:"):
            optimizer.zero_grad()

            x_pde = generate_pde_points(num_pde_points, spatial_domain, time_domain)
            x_bc = generate_bc_points(num_bc_points, spatial_domain, time_domain, self.spatial_dimension, self.time_dependent)
            x_ic = None

            if self.time_dependent:
               x_ic = generate_ic_points(num_ic_points,domain_bound, spatial_domain)


            loss_pde = self.compute_pde_loss(pde_residual_func, x_pde, pde_params={'params': pde_parameters})
            loss_bc = self.compute_bc_loss(boundary_condition_func , x_bc)

            total_loss = loss_pde + (lambda_bc * loss_bc)

            if self.time_dependent:
                loss_ic = self.compute_ic_loss(initial_condition_func, x_ic)
                total_loss += (lambda_ic * loss_ic)
            else:
                loss_ic = torch.tensor(0.0).to(device)

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                tqdm.write(
                    f"Epoch {epoch + 1}/{epochs}: Total Loss={total_loss.item():.4e},"
                    f" PDE Loss={loss_pde.item():.4e}, BC Loss={loss_bc.item():.4e},"
                    f" IC Loss={loss_ic.item():.4e}")

        print("Training complete.")