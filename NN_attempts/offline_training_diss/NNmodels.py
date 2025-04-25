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
    RENORMmean  : np.array 
    RENORMstd    : np.array
    
    def __init__(self, key, Nfeatures):
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
                
        self.layers = [eqx.nn.Conv2d(Nfeatures, 8, padding='SAME', kernel_size=3, key=key1),
                        jax.nn.relu,
                        eqx.nn.Conv2d(8, 16, padding='SAME', kernel_size=3, key=key2),
                        jax.nn.relu,
                        eqx.nn.Conv2d(16, 2, padding='SAME', kernel_size=3, key=key3),
                        # eqx.nn.MaxPool2d(kernel_size=2),
                        # jnp.ravel,
                        # eqx.nn.Linear(2*127*127, 512, key=key4),
                        # jax.nn.sigmoid,
                        # eqx.nn.Linear(512, 2*128*128, key=key5),
                        # jax.nn.sigmoid,
                        # lambda x:jnp.reshape(x, (2,128,128)),
                        ]
        
        self.RENORMmean, self.RENORMstd = np.zeros(2, dtype='float32'), np.zeros(2, dtype='float32')   
        # eqx.tree_at( lambda t:t.layers[-1].weight, self, self.layers[-1].weight*0.)
        # eqx.tree_at( lambda t:t.layers[-1].bias, self, self.layers[-1].bias*0.)
        

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = layer(x)      
        return x

class DissipationMLP(eqx.Module):
    """
    Dissipation is parametrized as a MLP.
    """
    layers: list
    # renormalization values (mean,std)
    RENORMmean  : np.array 
    RENORMstd    : np.array
    
    def __init__(self, key, Nfeatures):
        key1, key2, key3 = jax.random.split(key, 3)
                
        self.layers = [jnp.ravel,
                        eqx.nn.Linear(Nfeatures*128*128, 1024, key=key1),
                        jax.nn.tanh,
                        eqx.nn.Linear(1024, 1024, key=key2),
                        jax.nn.tanh,
                        eqx.nn.Linear(1024, 2*128*128, key=key2),
                        jax.nn.sigmoid,
                        ]   
        self.RENORMmean, self.RENORMstd = np.zeros(2, dtype='float32'), np.zeros(2, dtype='float32')   

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = layer(x)    
        x = jnp.reshape(x, (2,128,128)) 
        return x  
    
    
    
class DissipationRayleigh(eqx.Module):
    """
    D = -rU with r real
    
    first and second features needs to be U and V
    """
    layers: list
    # renormalization values (mean,std)
    RENORMmean  : np.array 
    RENORMstd    : np.array
    
    def __init__(self, R = jnp.asarray(0.1)):
        self.layers = [ Rayleigh_damping(R) ]
        self.RENORMmean, self.RENORMstd = np.zeros(2, dtype='float32'), np.zeros(2, dtype='float32')           

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = layer(x)      
        return x
    
  
    
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
    
    
class Rayleigh_damping(eqx.Module):
    R : jnp.ndarray
    def __init__(self, R):
        self.R = R
    def __call__(self, x):
        return jnp.stack([- self.R*x[0], -self.R*x[1]])