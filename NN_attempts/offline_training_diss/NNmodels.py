import jax
import equinox as eqx
from jaxtyping import Array, Float
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, '../../src')
from constants import *

# a tester : Unet
# https://github.com/Ceyron/UNet-in-JAX/blob/main/simple_unet_poisson_solver_in_jax.ipynb

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
                        eqx.nn.Conv2d(8, 4, padding='SAME', kernel_size=3, key=key2),
                        eqx.nn.MaxPool2d(kernel_size=3),
                        # jax.nn.relu,
                        eqx.nn.Conv2d(4, 2, padding='SAME', kernel_size=3, key=key3),
                        eqx.nn.MaxPool2d(kernel_size=3),
                        jnp.ravel,
                        eqx.nn.Linear(2*124*124, 256, key=key4),
                        # jax.nn.tanh,
                        eqx.nn.Linear(256, 2*128*128, key=key5),
                        # jax.nn.sigmoid,
                        lambda x:jnp.reshape(x, (2,128,128)),
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
                        jax.nn.tanh,
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
    RENORMmean  :jnp.array 
    RENORMstd    : jnp.array
    
    def __init__(self, R = jnp.asarray(0.1)):
        self.layers = [ Rayleigh_damping(R) ]
        self.RENORMmean, self.RENORMstd = jnp.zeros(2, dtype='float32'), jnp.zeros(2, dtype='float32')           

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        for layer in self.layers:
            x = layer(x)      
        return x
    
  
    
class DissipationRayleigh_MLP(eqx.Module):
    """
    D = -rU with r real, NN with non linear activation functions
    
    first and second features needs to be U and V (normalized)
    """
    layers: list
    # renormalization values (mean,std)
    RENORMmean  : np.array 
    RENORMstd    : np.array
    
    def __init__(self, key, Nfeatures, width):
        key1, key2, key3 = jax.random.split(key, 3)
        
        self.layers = [
                        jnp.ravel,
                       eqx.nn.Linear(Nfeatures*128*128, width, key=key1), 
                       jax.nn.relu,
                       eqx.nn.Linear(width, width, key=key2), 
                       jax.nn.relu,
                       eqx.nn.Linear(width,2*128*128, key=key3),
                       lambda x: jnp.reshape(x, (2,128,128)),
                       ]
        
        self.RENORMmean, self.RENORMstd = np.zeros(2, dtype='float32'), np.zeros(2, dtype='float32')           

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        uv = x[:2]
        for layer in self.layers:
            x = layer(x)      
        return x*uv
 
class DissipationRayleigh_NNlinear(eqx.Module):
    """
    D = -rU with r real, linear NN
    
    first and second features needs to be U and V (normalized)
    """
    layers: list
    # renormalization values (mean,std)
    RENORMmean  : np.array 
    RENORMstd    : np.array
    
    def __init__(self, key, Nfeatures, width):
        key1, key2, key3 = jax.random.split(key, 3)
        
        self.layers = [
                        jnp.ravel,
                       eqx.nn.Linear(Nfeatures*128*128, width, key=key1), 
                       eqx.nn.Linear(width,2*128*128, key=key2),
                       lambda x: jnp.reshape(x, (2,128,128)),
                       ]
        
        self.RENORMmean, self.RENORMstd = np.zeros(2, dtype='float32'), np.zeros(2, dtype='float32')           

    def __call__(self, x: Float[Array, "Nfeatures Ny Nx"]) -> Float[Array, "Currents Ny Nx"]:
        uv = x[:2]
        for layer in self.layers:
            x = layer(x)      
        return x*uv
    
    
class RHS_dynamic(eqx.Module):   
    fc : jnp.ndarray
    K : jnp.ndarray
        
    def __init__(self, fc, K):
        self.fc = jnp.asarray(fc)
        self.K = jnp.asarray(K)
  
    def __call__(self, U, V, TAx, TAy):
        d_U =  self.fc*V + jnp.exp(self.K)*TAx
        d_V = -self.fc*U + jnp.exp(self.K)*TAy
        return jnp.stack([d_U, d_V], axis=0)
    
class Coriolis(eqx.Module):
    fc : jnp.ndarray
        
    def __init__(self, fc):
        self.fc = jnp.asarray(fc)
    def __call__(self, U, V):
        d_U =  self.fc*V
        d_V = -self.fc*U
        return jnp.stack([d_U, d_V], axis=0)
    
    
class Stress(eqx.Module):
    K : jnp.ndarray
        
    def __init__(self, K):
        self.K = jnp.asarray(K)
  
    def __call__(self, TAx, TAy):
        # d_U = jnp.exp(self.K)*TAx
        # d_V = jnp.exp(self.K)*TAy
        d_U = self.K*TAx
        d_V = self.K*TAy
        return jnp.stack([d_U, d_V], axis=0)

class RHS_turb(eqx.Module):
    
    stress : eqx.Module
    dissipationNN : eqx.Module
    # renormalization values
    RENORMmean  : jnp.ndarray 
    RENORMstd   : jnp.ndarray
    
    def __init__(self, stress, dissipationNN):
        self.stress = stress
        self.dissipationNN = dissipationNN
        self.RENORMmean, self.RENORMstd = jnp.zeros(2), jnp.zeros(2)       
        
    def __call__(self, features: Float[Array, "Nfeatures Ny Nx"],
                        forcing: Float[Array, '2 Ny Nx'],                   
                    ) -> Float[Array, "Currents Ny Nx"]:
        
        TAx,TAy = forcing[0], forcing[1]
        stress = self.stress(TAx, TAy)
        dissipation_part = self.dissipationNN(features)
        return stress + dissipation_part
        
        
        
    
    
    
    
class Rayleigh_damping(eqx.Module):
    R : jnp.ndarray
    def __init__(self, R):
        self.R = R
    def __call__(self, x):
        return jnp.stack([- self.R*x[0], -self.R*x[1]])
    



def Forward_Euler(X0, RHS_dyn, diss_model, forcing, dt, dt_forcing, Nsteps):
    TAx, TAy, features = forcing
    nsubsteps = dt_forcing // dt
    
    # initialisation 
    ny,nx = TAx[0].shape
    U,V = np.zeros((Nsteps, ny, nx)), np.zeros((Nsteps, ny, nx))
    U[0], V[0] = X0[0], X0[1]
    
    for it in range(1, Nsteps):
        
        # on-the-fly interpolation
        itf = int(it//nsubsteps)
        aa = np.mod(it,nsubsteps)/nsubsteps
        itsup = np.where(itf+1>=len(TAx), -1, itf+1) 
        TAxnow = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAynow = (1-aa)*TAy[itf] + aa*TAy[itsup]
        features_now = (1-aa)*features[itf] + aa*features[itsup]
        
        dyn = RHS_dyn(U[it-1], V[it-1], TAxnow, TAynow)
        diss = diss_model(features_now)
        d_U = dyn[0] + diss[0]
        d_V = dyn[1] + diss[1]
        U[it],V[it] = U[it-1]+dt*d_U, V[it-1]+dt*d_V
    return U,V