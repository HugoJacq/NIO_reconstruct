"""
    dC/dt = -i.fc.C + K0.Tau - dissipation              (1)
"""
from jaxtyping import Int, Float, Array, PyTree
from jax import lax
import numpy as np
import jax.numpy as jnp
import jax 
from functools import partial
import equinox as eqx

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




# @partial(jax.jit, static_argnames=['dt','dt_forcing','Nsteps'])
def Integration_Euler(X0        : Float[Array, "2 Ny Nx"], 
                      forcing   : Float[Array, "Nt 2 Ny Nx"], 
                      features  : Float[Array, "Nt Nfeatures Ny Nx"], 
                      RHS       : PyTree, 
                      dt        : Float, 
                      dt_forcing: Float,
                      Nsteps    : Int,
                      norms     : dict,
                      ) -> Float[Array, "Nt 2 Ny Nx"]:  
    """
    Nsteps steps of Euler integration from currents at t=t0.
    
    The output is on 'dt_forcing' timescale, with a sub loop at 'dt'
    
    INPUTS:
        - X0        : initial current
        - forcing   : wind stress (time evolution)
        - features  : features (at least U,V they will be replaced) (time evolution)
        - RHS       : equinox module that computes the RHS of (1)
        - dt        : time step (sec), should be a divider of 'dt_forcing'
        - dt_forcing: time step of forcing (sec)
        - Nsteps    : how many time step to integrate for (how many 'dt_forcing')
        - norms     : a dictionnary on how to normalize 'features'
    OUTPUT:
        - trajectory of current 
    """
    def inner_loop(carry, iin):
        Cold, forcing, features, iout = carry
        
        # on the fly interpolation
        t = iout*dt_forcing + iin*dt
        forcing_now = linear_interpolator(t, dt, dt_forcing, forcing)
        features_now = linear_interpolator(t, dt, dt_forcing, features)
        
        # replacing the current for next iteration in the features        
        features_now = features_now.at[0,:,:].set( (Cold[0]-norms['features']['mean'][0])*norms['features']['std'][0] )
        features_now = features_now.at[1,:,:].set( (Cold[0]-norms['features']['mean'][1])*norms['features']['std'][1] )
        
        # one step forward at 'dt'
        Cnext = one_step_Euler(Cold, forcing_now, features_now, RHS, dt)
        
        return (Cnext, forcing, features, iout), None
        
    def outer_loop(carry, iout):
        C, forcing, features = carry
        Cold = C[iout]

        init = Cold, forcing, features, iout
        # loop on time steps every 'dt'
        final, _ = lax.scan(inner_loop, init, jnp.arange(nsubstep))
        Cnext, _, _, _ = final
        
        
        C = C.at[iout+1].set(Cnext)
        carry = C, forcing, features
        return carry, None

    nsubstep = int(dt_forcing//dt)

    # initialization of carry
    C = jnp.zeros_like(forcing)
    C = C.at[0,:,:,:].set(X0)
    carry = C, forcing, features
    # loop on time steps every 'dt_forcing'
    final,_ = eqx.internal.scan(outer_loop, carry, xs=np.arange(Nsteps), kind='checkpointed')
    C,_,_ = final
    
    return C

def linear_interpolator(t, dt, dt_forcing, field_forcing ):
    nsubsteps = dt_forcing//dt
    it = jnp.array(t//dt, int)
    itf = jnp.array(it//nsubsteps, int)
    aa = jnp.mod(it,nsubsteps)/nsubsteps
    itsup = jnp.where(itf+1>=len(field_forcing), -1, itf+1) 
    i_field = (1-aa)*field_forcing[itf] + aa*field_forcing[itsup]
    return i_field