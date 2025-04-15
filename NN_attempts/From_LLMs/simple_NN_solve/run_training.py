import numpy as np
import xarray as xr
import jax
import optax

from training import loss_fn
from dissipation_NN import DissipationModel
from dissipation_NN import Model


ds = xr.open_mfdataset("../data_regrid/croco_1h_inst_surf_2005-05-01-2005-05-31_0.1deg_conservative.nc")
dt_forcing = 3600 

oneday = 86400
dt = 60.
f = 1e-4

t_span = np.arange(0.,len(ds.time)*dt_forcing, dt)
tau = ds.oceTAUX.values + 1j*ds.oceTAUY.values
U0 = 1j*0.
U_data = ds.U.values + 1j*ds.V.values


SEED = 5678
key = jax.random.PRNGKey(SEED)



# Create an optimizer
optimizer = optax.adam(learning_rate=1e-3)

# Initialize the neural network
dissipation_model = DissipationModel(hidden_size=64, key=key)

# Initialize optimizer state
opt_state = optimizer.init(dissipation_model.params)

# Define the update step
@jax.jit
def update(model, opt_state, U0, t_span, f, tau, U_data):
    grads = jax.grad(loss_fn)(model, U0, t_span, f, tau, U_data)
    updates, opt_state = optimizer.update(grads, opt_state)
    
    # Apply the updates
    model = optax.apply_updates(model, updates)
    return model, opt_state


def train(U0, t_span, f, tau, U_data, num_epochs=1000):
    # Initialize model with initial guess for K0
    key, subkey = jax.random.split(key, 2)
    dissipation_model = DissipationModel(hidden_size=64, key=subkey)
    model = Model(dissipation_model=dissipation_model, K0=1.0)  # Start with K0 = 1.0
    
    opt_state = optimizer.init(model)
    
    for epoch in range(num_epochs):
        model, opt_state = update(model, opt_state, U0, t_span, f, tau, U_data)
        
        if epoch % 100 == 0:
            loss_value = loss_fn(model, U0, t_span, f, tau, U_data)
            print(f"Epoch {epoch}, Loss: {loss_value}, K0: {model.K0}")

    return model


trained_model = train(U0, t_span, f, tau, U_data)


