import equinox as eqx
import jax
from jaxtyping import Array, Float


class blackbox(eqx.Module):
    # layers: list
    layer1 : eqx.Module
    layer2 : eqx.Module
    layer3 : eqx.Module

    def __init__(self, key, Ninput_features):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layer1 = eqx.nn.Conv2d(Ninput_features, 16, padding='SAME', kernel_size=3, key=key1)
        self.layer2 = eqx.nn.Conv2d(16, 32, padding='SAME', kernel_size=3, key=key2)
        self.layer3 = eqx.nn.Conv2d(32, 2, padding='SAME', kernel_size=3, key=key3)

    def __call__(self, x: Float[Array, "Ninput_features 32 32"]) -> Float[Array, "2 32 32"]:
        x = jax.nn.relu( self.layer1(x) )
        x = jax.nn.relu( self.layer2(x) )
        x = self.layer3(x)
        
        return x