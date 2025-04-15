import jax.numpy as jnp
import jax
from jax import lax
import equinox as eqx
from jaxtyping import Array, Float
from functools import partial
import diffrax

import sys

from experiments.optimal_dTK.run_opti_dTK import TAy
sys.path.insert(0, '../../src')

jax.config.update("jax_enable_x64", True)

from constants import *

class DissipationNN(eqx.Module):
    # layers: list
    layer1 : eqx.Module
    layer2 : eqx.Module
    #layer3 : eqx.Module

    # DO I NEED FLOAT64 for the NN ???

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        # key1, key2 = jax.random.split(key, 2)

        self.layer1 = eqx.nn.Conv2d(2, 16, padding='SAME', kernel_size=3, key=key1)
        self.layer2 = eqx.nn.Conv2d(16, 2, padding='SAME', kernel_size=3, key=key2)
        # self.layer2 = eqx.nn.Conv2d(16, 32, padding='SAME', kernel_size=3, key=key2)
        # self.layer3 = eqx.nn.Conv2d(32, 2, padding='SAME', kernel_size=3, key=key3)

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        # for layer in self.layers:
        #     x = layer(x)
        x = jax.nn.relu( self.layer1(x) )
        x = jax.nn.relu( self.layer2(x) )
        #x = jax.nn.relu( self.layer3(x) )
        
        return x
    
class RHS(eqx.Module):
    
    U : jnp.ndarray
    V : jnp.ndarray
    TAx : jnp.ndarray
    TAy : jnp.ndarray
    fc : jnp.ndarray
    
    K : jnp.ndarray
    dissipation : eqx.Module
    
    def __init__(self, C, fc, Tau, K, dissipation_model):
        self.U,self.V = C
        self.TAx, self.TAy = Tau
        self.K = K
        self.dissipation = dissipation_model
        self.fc = fc
  
    def __call__(self):
        d_U = self.fc*self.V + self.K*self.TAx - self.dissipation_model[0]
        d_V = -self.fc*self.U + self.K*self.TAy - self.dissipation_model[1]
        return d_U, d_V