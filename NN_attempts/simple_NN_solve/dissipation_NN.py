import jax
import jax.numpy as jnp
import equinox as eqx
import optax

class DissipationModel(eqx.Module):
    # Define a simple neural network model
    hidden_size: int = 64
    layers: list
    
    def __init__(self, hidden_size: int, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = [
                    eqx.nn.Linear(1, hidden_size, key=key1),  # input dimension: 1 (U)
                    eqx.nn.Linear(hidden_size, 1, key=key2),  # output dimension: 1 (dissipation)
                        ]
        
    def __call__(self, U: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([U, tau], axis=-1)  # concatenate U and tau as input to the network
        x = jax.nn.relu(self.layer1(x))  # First hidden layer with ReLU activation
        return self.layer2(x)  # Output layer for dissipation

class Model(eqx.Module):
    dissipation_model: DissipationModel
    K0: float  # K0 as a learnable parameter
    
    def __init__(self, dissipation_model: DissipationModel, K0: float):
        self.dissipation_model = dissipation_model
        self.K0 = K0



