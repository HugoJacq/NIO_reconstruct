import jax
import equinox as eqx
from jaxtyping import Array, Float
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, '../../src')
from constants import *



class DissipationCNN(eqx.Module):
    """
    Dissipation is parametrized as a CNN.
    """
    
    
    layers: list
    
    # renormalization values (mean,std)
    RENORM : tuple 
    
    def __init__(self, key, Nfeatures):
        key1, key2, key3 = jax.random.split(key, 3)
                
        self.layers = [eqx.nn.Conv2d(Nfeatures, 16, padding='SAME', kernel_size=3, key=key1),
                        eqx.nn.Conv2d(16, 32, padding='SAME', kernel_size=3, key=key2),
                        eqx.nn.Conv2d(32, 2, padding='SAME', kernel_size=3, key=key3) ]
        self.RENORM = (0.,0.)
        # eqx.tree_at( lambda t:t.layers[-1].weight, self, self.layers[-1].weight*0.)
        # eqx.tree_at( lambda t:t.layers[-1].bias, self, self.layers[-1].bias*0.)
        

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = jax.nn.relu(layer(x))        
        return x

class DissipationMLP(eqx.Module):
    """
    Dissipation is parametrized as a MLP.
    """
    
    
    layers: list
    def __init__(self, key, Nfeatures):
        key1, key2, key3 = jax.random.split(key, 3)
                
        self.layers = []       

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
             
        return x  
    
class DissipationRayleigh(eqx.Module):
    """
    """
    
class DissipationRayleigh_MLP(eqx.Module):
    """
    """
 
class DissipationRayleigh_NNlinear(eqx.Module):
    """
    """   
    
    
class RHS_dynamic():
    fc : np.ndarray
    K : np.ndarray
        
    def __init__(self, fc, K):
        self.K = np.asarray(K)
        self.fc = np.asarray(fc)
  
    def __call__(self, U, V, TAx, TAy):
        d_U =  self.fc*V + self.K*TAx
        d_V = -self.fc*U + self.K*TAy
        return d_U, d_V