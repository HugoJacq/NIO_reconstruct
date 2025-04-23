import jax.numpy as jnp
import jax
import numpy as np
from jax import lax
import equinox as eqx
from jaxtyping import Array, Float
from functools import partial
import diffrax

import sys

sys.path.insert(0, '../../src')



from constants import *

class DissipationNN(eqx.Module):
    layers: list
    # layer1 : eqx.Module
    # layer2 : eqx.Module
    #layer3 : eqx.Module

    # DO I NEED FLOAT64 for the NN ???

    def __init__(self, key, Nfeatures):
        key1, key2, key3 = jax.random.split(key, 3)
                
        self.layers = [eqx.nn.Conv2d(Nfeatures, 16, padding='SAME', kernel_size=3, key=key1),
                        eqx.nn.Conv2d(16, 32, padding='SAME', kernel_size=3, key=key2),
                        eqx.nn.Conv2d(32, 2, padding='SAME', kernel_size=3, key=key3) ]
        
        eqx.tree_at( lambda t:t.layers[-1].weight, self, self.layers[-1].weight*0.)
        eqx.tree_at( lambda t:t.layers[-1].bias, self, self.layers[-1].bias*0.)
        

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = jax.nn.relu(layer(x))        
        return x
    
class RHS(eqx.Module):
    
    fc : jnp.ndarray
    K : jnp.ndarray
    dissipation : eqx.Module
    
    def __init__(self, fc, K, dissipation_model):
        self.K = K
        self.dissipation = dissipation_model
        self.fc = fc
  
    def __call__(self, U, V, TAx, TAy, features):

        # mean = inputs.mean()
        # std = inputs.std()
        
        # n_inputs = (inputs-mean)/std
        # n_diss = self.dissipation(n_inputs)
        
        # d_diss = n_diss*std + mean
        
        diss = self.dissipation(features)
        
        #jax.debug.print('U: Coriolis, stress, diss, {}, {}, {}', self.fc[5]*V[5,5], self.K*TAx[5,5], -diss[0][5,5])
        
        
        d_U =  self.fc*V + self.K*TAx - diss[0]
        d_V = -self.fc*U + self.K*TAy - diss[1]
        return d_U, d_V
    

def Forward_Euler(X0, RHS, forcing, dt, Nsteps):
    TAx, TAy, features = forcing

    # initialisation 
    ny,nx = TAx[0].shape
    U,V = np.zeros((Nsteps,ny, nx)), np.zeros((Nsteps,ny, nx))
    U[0], V[0] = X0[0], X0[1]
    
    for it in range(1, Nsteps):
        d_U, d_V = RHS(U[it-1], V[it-1], TAx[it], TAy[it], features[it])   
        U[it],V[it] = U[it-1]+dt*d_U, V[it-1]+dt*d_V
    return U,V
    
    