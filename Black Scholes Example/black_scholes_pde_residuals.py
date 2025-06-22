import math
import torch
from scipy.stats import norm

# Black Scholes parameters
K_BS = 100.0 # Strike price
R_BS = 0.05 # Risk-Free rate
Sigma_BS = 0.2 # Volatility
T_exp_BS = 1.0  # Total Time to expiration

def black_scholes_analytical_1d(s_coords:torch.Tensor, t_coords: torch.Tensor)-> torch.Tensor:
    """
    Calculate the analytical solution to the Black-Scholes equation for a one-dimensional option pricing problem.

    The Black-Scholes analytical solution calculates the price of a financial derivative
    based on a stochastic differential equation. The function computes the solution
    for given asset price coordinates, time coordinates, and pre-defined constants
    such as the strike price, volatility, and risk-free interest rate. The formula
    accounts for the log-normal behavior of asset prices over time.
    Equation: dV/dt + rS dV/dS + 0.5*sigma^2*S^2 d2V/dS2 - rV = 0
    """
    # Time remaining to expiration (T-t)
    tau = T_exp_BS - t_coords
    
    # t = T_exp_BS (tau = 0)
    # At expiration, V = max(S-K, 0)
    non_zero_tau_mask = tau> 1e-10
    
    d1 = torch.zeros_like(s_coords)
    d2 = torch.zeros_like(s_coords)
    
    # Compute d1 and d2 only where tau is not zero
    d1[non_zero_tau_mask]= (
    (torch.log(s_coords[non_zero_tau_mask]/K_BS) + (R_BS + 0.5 * Sigma_BS ** 2) * tau[non_zero_tau_mask])
    / (Sigma_BS * torch.sqrt(tau[non_zero_tau_mask]))
    )

    d2[non_zero_tau_mask] = d1[non_zero_tau_mask] - Sigma_BS * torch.sqrt(tau[non_zero_tau_mask])

    # convert tensors to numpy for scipy.stats.norm.cdf
    d1_np = d1.cpu().numpy()
    d2_np = d2.cpu().numpy()

    N_d1 = torch.tensor(norm.cdf(d1_np), dtype=torch.float32,device= s_coords.device)
    N_d2 = torch.tensor(norm.cdf(d2_np), dtype=torch.float32,device= s_coords.device)

    # Calculating the Option Price
    V_BS = s_coords * N_d1 - K_BS * torch.exp(-R_BS * tau) * N_d2

    # For points at expiration (tau is zeor or very small), use the payoff function
    V_BS[~non_zero_tau_mask] = torch.max(s_coords[~non_zero_tau_mask] - K_BS, torch.tensor(0.0, device=s_coords.device))

    return V_BS


def black_scholes_pde_residual_1d(u_pred: torch.Tensor, d_u_dt: torch.Tensor,
                                  spatial_first_deriv: torch.Tensor, spatial_second_deriv: torch.Tensor,
                                    x_coords: torch.Tensor, t_coords: torch.Tensor,
                                    spatial_mixed_deriv: dict,
                                  params: dict) -> torch.Tensor:
    """
    Computes the residual of the 1D Black-Scholes PDE:
    dV/dt + rS dV/dS + 0.5*sigma^2*S^2 d2V/dS2 - rV = 0
    """
    r = params.get('r_bs', R_BS)
    sigma = params.get('sigma_bs', Sigma_BS)

    d_u_ds = spatial_first_deriv[:, 0:1]  # S is the first (and only) spatial dim
    d2_u_ds2 = spatial_second_deriv[:, 0:1]  # S is the first (and only) spatial dim

    # Left-hand side of the PDE
    bs_lhs = d_u_dt + r * x_coords * d_u_ds + 0.5 * sigma ** 2 * x_coords ** 2 * d2_u_ds2 - r * u_pred

    return bs_lhs  # This is R(u), so L_PDE = mean(R(u)^2)


def black_scholes_boundary_condition_1d(S_coords: torch.Tensor, t_coords: torch.Tensor) -> torch.Tensor:
    """
    Boundary condition for 1D Black-Scholes:
    - At S_min, V = 0 (for a call option)
    - At S_max, V approx S_max - K * exp(-r*(T_exp-t))
    """
    r = R_BS
    K = K_BS
    T_exp = T_exp_BS

    # Assuming S_min and S_max are obtained from the domain_bound passed to solver
    # This function should be general and check for boundary points.

    # A robust way is to use the full analytical solution at the boundaries:
    # This also handles the S=0 boundary condition automatically by Black-Scholes formula
    # S_coords, t_coords here are the boundary points for the BC loss.
    return black_scholes_analytical_1d(S_coords, t_coords)


def black_scholes_initial_condition_1d(S_coords: torch.Tensor) -> torch.Tensor:
    """
    Initial condition for Black-Scholes (at t=T_exp, using t from 0 to T_exp) - this is the payoff function.
    V(S, T_exp) = max(S - K, 0) for a European Call option.
    """
    return torch.max(S_coords - K_BS, torch.tensor(0.0, device=S_coords.device))


# --- 2D Black-Scholes (Simplified for demonstration) ---
# For a full 2D correlated Black-Scholes, the PDE is significantly more complex
# and analytical solutions are rare. This is a placeholder for basic structure.
# You might consider an "exchange option" or "basket option" for which some analytical
# approximations exist or simplify the problem to be independent.

def black_scholes_analytical_2d(S1_coords: torch.Tensor, S2_coords: torch.Tensor,
                                t_coords: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for 2D Black-Scholes analytical solution.
    For simplicity, return a dummy value or a simplified problem's analytical solution.
    E.g., just the payoff at t=T_EXP_BS for comparison.
    """
    # For a basket option max(S1+S2-K, 0) at expiration
    # This would typically be non-trivial to compute for t < T_EXP_BS
    # For a basic demo, you might just use max(S1+S2-K, 0) for final time, and None otherwise.

    # As a simple illustration, let's assume a simplified case:
    # An option whose value is based on S1, but varies with S2 for some reason.
    # OR, more common, simply return the payoff.
    # To enable plotting for t < T_EXP_BS, this would need a proper analytical solution.

    # If no simple analytical for general t, just return payoff.
    # This part needs a specific analytical solution to function correctly for plotting.
    # For demo, you might just return S1+S2.
    return S1_coords + S2_coords  # Placeholder: Not a true BS solution


def black_scholes_pde_residual_2d(u_pred: torch.Tensor, d_u_dt: torch.Tensor,
                                  spatial_first_deriv: torch.Tensor, spatial_second_deriv: torch.Tensor,
                                  spatial_mixed_deriv: dict, x_coords: torch.Tensor, t_coords: torch.Tensor,
                                  # x_coords will be S1, S2
                                  params: dict) -> torch.Tensor:
    """
    PDE residual for 2D Black-Scholes (simplified):
    dV/dt + rS1 dV/dS1 + rS2 dV/dS2 + 0.5*sigma1^2*S1^2 d2V/dS1^2 + 0.5*sigma2^2*S2^2 d2V/dS2^2
    + rho*sigma1*sigma2*S1*S2*d2V/dS1dS2 - rV = 0
    """
    r = params.get('r_bs', R_BS)
    sigma1 = params.get('sigma1_bs', Sigma_BS)
    sigma2 = params.get('sigma2_bs', Sigma_BS)
    rho = params.get('rho_bs', 0.5)  # Example correlation

    S1 = x_coords[:, 0:1]
    S2 = x_coords[:, 1:2]

    d_u_ds1 = spatial_first_deriv[:, 0:1]
    d_u_ds2 = spatial_first_deriv[:, 1:2]

    d2_u_ds1_ds1 = spatial_second_deriv[:, 0:1]
    d2_u_ds2_ds2 = spatial_second_deriv[:, 1:2]

    # Mixed derivative
    d2_u_ds1_ds2 = spatial_mixed_deriv.get((0, 1), torch.zeros_like(S1))  # Get mixed derivative (S1, S2)

    bs_lhs = (d_u_dt + r * S1 * d_u_ds1 + r * S2 * d_u_ds2 +
              0.5 * sigma1 ** 2 * S1 ** 2 * d2_u_ds1_ds1 +
              0.5 * sigma2 ** 2 * S2 ** 2 * d2_u_ds2_ds2 +
              rho * sigma1 * sigma2 * S1 * S2 * d2_u_ds1_ds2 -
              r * u_pred)
    return bs_lhs


def black_scholes_boundary_condition_2d(S_coords: torch.Tensor, t_coords: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for 2D Black-Scholes boundary conditions.
    Using a simpler approximation for S_max boundaries, and 0 for S_min boundaries.
    S_coords here will be (S1, S2) for boundary points.
    """
    # For demo, use 0 at low S values, and S1+S2-K*exp(-r*tau) at high S values
    # This needs careful implementation for each of the 4 edges of the S1-S2 plane.
    # A robust approach would be to use the analytical solution if it exists.

    # For now, return max(S_coords.sum(dim=1, keepdim=True) - K_BS, torch.tensor(0.0, device=S_coords.device))
    # Or, as in 1D, use the analytical formula. Here, we'll assume a simplified
    # scenario, e.g., European call on max(S1, S2) or a basket.
    # Using a simple boundary condition:
    return S_coords[:, 0:1] + S_coords[:, 1:2]  # Placeholder. Should be problem specific.


def black_scholes_initial_condition_2d(S_coords: torch.Tensor) -> torch.Tensor:
    """
    Initial condition for 2D Black-Scholes (at t=T_exp) - payoff function.
    E.g., max(S1 + S2 - K, 0) for a basket option.
    S_coords here will be (S1, S2).
    """
    S1 = S_coords[:, 0:1]
    S2 = S_coords[:, 1:2]
    return torch.max(S1 + S2 - K_BS, torch.tensor(0.0, device=S_coords.device))


# --- 3D Black-Scholes (Simplified for demonstration) ---
# Even more complex with 3 assets and correlations.
# Placeholder for basic structure.

def black_scholes_analytical_3d(S1_coords: torch.Tensor, S2_coords: torch.Tensor, S3_coords: torch.Tensor,
                                t_coords: torch.Tensor) -> torch.Tensor:
    """Placeholder for 3D Black-Scholes analytical solution."""
    # Likely no simple analytical solution. Return a dummy value or payoff for comparison.
    return S1_coords + S2_coords + S3_coords  # Placeholder. Not a true BS solution


def black_scholes_pde_residual_3d(u_pred: torch.Tensor, d_u_dt: torch.Tensor,
                                  spatial_first_derivs: torch.Tensor, spatial_second_derivs_diag: torch.Tensor,
                                  spatial_mixed_derivs: dict, x_coords: torch.Tensor, t_coords: torch.Tensor,
                                  # x_coords will be S1, S2, S3
                                  params: dict) -> torch.Tensor:
    """
    PDE residual for 3D Black-Scholes (simplified):
    Sum of 1st, 2nd, and mixed derivatives.
    """
    r = params.get('r_bs', R_BS)
    sigma1 = params.get('sigma1_bs', Sigma_BS)
    sigma2 = params.get('sigma2_bs', Sigma_BS)
    sigma3 = params.get('sigma3_bs', Sigma_BS)
    rho12 = params.get('rho12_bs', 0.3)
    rho13 = params.get('rho13_bs', 0.3)
    rho23 = params.get('rho23_bs', 0.3)

    S1 = x_coords[:, 0:1]
    S2 = x_coords[:, 1:2]
    S3 = x_coords[:, 2:3]

    d_u_ds1 = spatial_first_derivs[:, 0:1]
    d_u_ds2 = spatial_first_derivs[:, 1:2]
    d_u_ds3 = spatial_first_derivs[:, 2:3]

    d2_u_ds1_ds1 = spatial_second_derivs_diag[:, 0:1]
    d2_u_ds2_ds2 = spatial_second_derivs_diag[:, 1:2]
    d2_u_ds3_ds3 = spatial_second_derivs_diag[:, 2:3]

    d2_u_ds1_ds2 = spatial_mixed_derivs.get((0, 1), torch.zeros_like(S1))
    d2_u_ds1_ds3 = spatial_mixed_derivs.get((0, 2), torch.zeros_like(S1))
    d2_u_ds2_ds3 = spatial_mixed_derivs.get((1, 2), torch.zeros_like(S1))

    bs_lhs = (d_u_dt + r * S1 * d_u_ds1 + r * S2 * d_u_ds2 + r * S3 * d_u_ds3 +
              0.5 * sigma1 ** 2 * S1 ** 2 * d2_u_ds1_ds1 +
              0.5 * sigma2 ** 2 * S2 ** 2 * d2_u_ds2_ds2 +
              0.5 * sigma3 ** 2 * S3 ** 2 * d2_u_ds3_ds3 +
              rho12 * sigma1 * sigma2 * S1 * S2 * d2_u_ds1_ds2 +
              rho13 * sigma1 * sigma3 * S1 * S3 * d2_u_ds1_ds3 +
              rho23 * sigma2 * sigma3 * S2 * S3 * d2_u_ds2_ds3 -
              r * u_pred)
    return bs_lhs