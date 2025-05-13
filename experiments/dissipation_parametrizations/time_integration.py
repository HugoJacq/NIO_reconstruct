"""
    dC/dt = -i.fc.C + K0.Tau - dissipation              (1)
"""
from jaxtyping import Int, Float, Array, PyTree
from jax import lax
import numpy as np
import jax.numpy as jnp
import jax 


def one_step_Euler(X0       : Float[Array, "2 Ny Nx"], 
                   forcing  : Float[Array, "2 Ny Nx"], 
                   features : Float[Array, "Nfeatures Ny Nx"], 
                   RHS      : PyTree, 
                   dt       : Float
                   ) -> Float[Array, "2 Ny Nx"]:
    """
    One step of Euler integration
    
    INPUTS:
        - X0        : initial current
        - forcing   : wind stress
        - features  : features (at least U,V)
        - RHS       : equinox module that computes the RHS of (1)
        - dt        : time step (sec)
    OUTPUT:
        Current at next time step
    """ 
    return X0 + dt*RHS(X0, forcing, features)





def Integration_Euler(X0        : Float[Array, "2 Ny Nx"], 
                      forcing   : Float[Array, "Nt 2 Ny Nx"], 
                      features  : Float[Array, "Nt Nfeatures Ny Nx"], 
                      RHS       : PyTree, 
                      dt        : Float, 
                      Nsteps    : Int,
                      norms     : dict,
                      ) -> Float[Array, "Nt 2 Ny Nx"]:  
    """
    Nsteps steps of Euler integration from currents at t=t0.
    
    INPUTS:
        - X0        : initial current
        - forcing   : wind stress (time evolution)
        - features  : features (at least U,V they will be replaced) (time evolution)
        - RHS       : equinox module that computes the RHS of (1)
        - dt        : time step (sec)
        - Nsteps    : how many time step to integrate for
        - norms     : a dictionnary on how to normalize 'features'
    OUTPUT:
        - trajectory of current 
    """
    def one_step_for_scan(carry, k):
        C, forcing, features = carry
        Cnext = one_step_Euler(C[k], forcing[k], features[k], RHS, dt)
        # replacing the current for next iteration in the features
        features = features.at[k,0].set( Cnext[0]*norms['features']['std'][0]+norms['features']['mean'][0] )
        features = features.at[k,1].set( Cnext[1]*norms['features']['std'][1]+norms['features']['mean'][1] )
        C = C.at[k].set(Cnext)
        carry = C, forcing, features
        
        return carry, None

    # initialization of carry
    C = jnp.zeros_like(forcing)
    C = C.at[0].set(X0)
    carry = C, forcing, features
    # loop on time steps
    C,_ = lax.scan(one_step_for_scan, carry, xs=np.arange(Nsteps))

    return C