import equinox as eqx
import jax
from jaxtyping import Array, Float
import jax.numpy as jnp

class blackbox(eqx.Module):
    layers : list

    def __init__(self, key, Ninput_features):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)
        self.layers = [
            eqx.nn.Conv2d(Ninput_features, 64, padding='SAME', kernel_size=3, key=key1),
            #jax.nn.hard_tanh,
            eqx.nn.Conv2d(64, 32, padding='SAME', kernel_size=3, key=key2),
            #jax.nn.hard_tanh,
            eqx.nn.Conv2d(32, 1, padding='SAME', kernel_size=3, key=key3),
            #eqx.nn.Conv2d(16, 1, padding='SAME', kernel_size=3, key=key4),
        ]

    def __call__(self, x: Float[Array, "Ninput_features Nx Ny"]) -> Float[Array, "1 Nx Ny"]:
        for layer in self.layers:
            x = layer(x)
        return x