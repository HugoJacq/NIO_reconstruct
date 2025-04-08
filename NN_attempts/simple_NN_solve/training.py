import jax.numpy as jnp
import jax
import optax

from ode_solver import solve_ode
from dissipation_NN import DissipationModel, Model


def loss_fn(model, U0, t_span, f, tau, U_data):
    dissipation_model = model.dissipation_model
    K0 = model.K0  # Use K0 from the model
    
    # Solve the ODE to get U predictions
    U_pred = solve_ode(U0, t_span, f, K0, tau, dissipation_model)
    
    # Calculate loss as the difference between predicted and observed U values
    return jnp.mean((U_pred - U_data) ** 2)  # Mean squared error





