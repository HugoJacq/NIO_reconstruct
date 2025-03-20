"""
This modules gather the definition of classical models used in the litterature 
to reconstruct inertial current from wind stress
Each model needs a forcing, from the module 'forcing.py'

The models are written in JAX to allow for automatic differentiation

refs:
Wang et al. 2023: https://www.mdpi.com/2072-4292/15/18/4526

By: Hugo Jacquet march 2025
"""
import numpy as np
import xarray as xr

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import equinox as eqx
from diffrax import ODETerm, diffeqsolve, Euler
import diffrax

class jslab(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.array
    # parameters
    TAx : jnp.array
    TAy : jnp.array
    fc : jnp.array
    dt_forcing : np.int32
    nl : np.int32
    AD_mode : str
    
    @eqx.filter_jit
    def __call__(self, call_args, save_traj_at = None):
        t0, t1, dt = call_args
        nsubsteps = self.dt_forcing // dt
        def vector_field(t, C, args):
            U,V = C
            fc, K, TAx, TAy = args
            
            # on the fly interpolation
            it = jnp.array(t//dt, int)
            itf = jnp.array(it//nsubsteps, int)
            
            aa = jnp.mod(it,nsubsteps)/nsubsteps
            itsup = lax.select(itf+1>=len(TAx), -1, itf+1) 
            TAx = (1-aa)*TAx[itf] + aa*TAx[itsup]
            TAy = (1-aa)*TAy[itf] + aa*TAy[itsup]
            # def cond_print(it):
            #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
            
            # jax.lax.cond(it<=10, cond_print, lambda x:None, it)
            
            # physic
            d_U = fc*V + K[0]*TAx - K[1]*U
            d_V = -fc*U + K[0]*TAy - K[1]*V
            d_y = d_U,d_V
            return d_y
        
        term = ODETerm(vector_field)
        
        solver = Euler()
        # Auto-diff mode
        if self.AD_mode=='F':
            adjoint = diffrax.ForwardMode()
        else:
            adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)

        y0 = 0.0,0.0 # self.U0,self.V0
        # control
        K = jnp.exp( jnp.asarray(self.pk) )
  
        args = self.fc, K, self.TAx, self.TAy
        
        if save_traj_at is None:
            saveat = diffrax.SaveAt(steps=True)
        else:
            saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            #saveat = diffrax.SaveAt(ts=save_traj_at)
        
        maxstep = int((t1-t0)//dt) +1 
        
        return diffeqsolve(term, 
                           solver, 
                           t0=t0, 
                           t1=t1, 
                           y0=y0, 
                           args=args, 
                           dt0=dt, #dt, None
                           saveat=saveat,
                           #stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                           adjoint=adjoint,
                           max_steps=maxstep,
                           made_jump=False) # here this is needed to be able to forward AD