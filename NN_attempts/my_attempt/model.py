import jax.numpy as jnp
import jax
from jax import lax
import equinox as eqx
from jaxtyping import Array, Float
from functools import partial

import sys
sys.path.insert(0, '../../src')

jax.config.update("jax_enable_x64", True)

from constants import *

class DissipationNN(eqx.Module):
    # layers: list
    layer1 : eqx.Module
    layer2 : eqx.Module
    #layer3 : eqx.Module

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)

        self.layer1 = eqx.nn.Conv2d(2, 16, padding='SAME', kernel_size=3, key=key1)
        # self.layer2 = eqx.nn.Conv2d(16, 16, padding='SAME', kernel_size=3, key=key2)
        # self.layer3 = eqx.nn.Conv2d(16, 2, padding='SAME', kernel_size=3, key=key3)
        self.layer2 = eqx.nn.Conv2d(16, 2, padding='SAME', kernel_size=3, key=key2)

    def __call__(self, x: Float[Array, "2 128 128"]) -> Float[Array, "2 128 128"]:
        # for layer in self.layers:
        #     x = layer(x)
        x = jax.nn.relu( self.layer1(x) )
        x = jax.nn.relu( self.layer2(x) )
        #x = jax.nn.relu( self.layer3(x) )
        
        return x

class jslab(eqx.Module):
    # control vector
    K0 : jnp.ndarray
    # parameters
    TAx : jnp.ndarray   #= eqx.static_field()
    TAy : jnp.ndarray   #= eqx.static_field()
    fc : jnp.ndarray    #= eqx.static_field()
    dt_forcing : jnp.ndarray          = eqx.static_field()
    # t0 : jnp.ndarray          
    # t1 : jnp.ndarray          
    dt : jnp.ndarray         = eqx.static_field()
    
    dissipation_model : DissipationNN
    
    def __init__(self, K0, TAx, TAy, fc, dt_forcing, dissipationNN, dt): #, call_args
        self.K0 = K0
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        self.dt_forcing = dt_forcing            
        # t0,t1,dt = call_args
        # self.t0 = t0
        # self.t1 = t1
        self.dt = dt
        
        self.dissipation_model = dissipationNN


    
    
    #@jax.checkpoint
    # @partial(eqx.filter_jit, static_argnums=0)
    @partial(jax.jit, static_argnums=(1,2))
    # def __call__(self, call_args=[0.0,oneday], save_traj_at = None):
    def __call__(self, t0=0.0, t1=oneday, save_traj_at = None):
        #t0, t1, dt = self.t0, self.t1, self.dt # call_args
        # t0, t1 = call_args
        nsubsteps = self.dt_forcing // self.dt
        
        # control
        K = jnp.exp( jnp.asarray(self.K0) )
  
        args = self.fc, K, self.TAx, self.TAy, nsubsteps, self.dissipation_model


        Nforcing = int((t1-t0)//self.dt_forcing)
        
        if save_traj_at is None:
            step_save_out = 1
        else:
            if save_traj_at<self.dt_forcing:
                raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
            else:
                step_save_out = int(save_traj_at//self.dt_forcing)
        
        # initialisation at null current
        nx, ny = self.TAx.shape[-1], self.TAx.shape[-2]
        U, V = jnp.zeros((Nforcing, ny, nx)), jnp.zeros((Nforcing, ny, nx))
        
        # inner loop at dt
        def __inner_loop(carry, iin):
            Uold, Vold, iout = carry
            t = iout*self.dt_forcing + iin*self.dt
            C = Uold, Vold
            d_U,d_V = self.vector_field(t, C, args)
            # newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V 
            # Euler hard coded
            X1 = Uold + self.dt*d_U, Vold + self.dt*d_V, iout
            return X1, X1
        
        # outer loop at dt_forcing
        def __outer_loop(carry, iout):
            U,V = carry
            X1 = U[iout], V[iout], iout
            final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
            newU, newV, _ = final
            X0 = U.at[iout+1].set(newU), V.at[iout+1].set(newV)
            return X0, X0
        
        # old way        
        # final, _ = lax.scan(__outer_loop, init=(U,V), xs=jnp.arange(0,Nforcing))
            
        def __outer_loop_for_while(val):
            carry, iout = val
            X0,_ = __outer_loop(carry, iout)
            return X0, iout+1
        # new way, backward AD
        final, _ = eqx.internal.while_loop(cond_fun = lambda val: val[1] < Nforcing,
                                           body_fun = __outer_loop_for_while,
                                           init_val = ((U,V), 0),
                                           kind='checkpointed',
                                           max_steps = Nforcing)
        # new way, forward AD
        # final, _ = eqx.internal.while_loop(cond_fun = lambda val: cond_fun(val, Nforcing),
        #                                    body_fun = __outer_loop_for_while,
        #                                    init_val = ((U,V), 0),
        #                                    kind='lax',
        #                                    max_steps = Nforcing)
        
        """
        equinox/internal/_loop/checkpointed.py
        
        `checkpoints`: The number of steps at which to checkpoint. The memory consumed
        will be that of `checkpoints`-many copies of `init_val`. (As the state is
        updated throughout the loop.
        """
        U,V = final
        
        if save_traj_at is None:
            solution = U,V
        else:
            solution = U[::step_save_out], V[::step_save_out]
            
            
        return solution

    # vector field is common whether we use diffrax or not
    def vector_field(self, t, C, args):
        U,V = C
        fc, K, TAxt, TAyt, nsubsteps, dissipation_model = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAxt), -1, itf+1) 
        TAx = (1-aa)*TAxt[itf] + aa*TAxt[itsup]
        TAy = (1-aa)*TAyt[itf] + aa*TAyt[itsup]
        
        # evaluate the NN at specific forcing
        
        input = jnp.stack([U,V])
        print(input.shape)
        dissipation = dissipation_model(input)
        #print(dissipation.shape)
        
        # physic
        d_U = fc*V + K*( TAx )  - dissipation[0]
        d_V = -fc*U + K*( TAy ) - dissipation[1]
        # d_U = fc*V + K*( (1-aa)*TAxt[itf] + aa*TAxt[itsup] )  - dissipation[0]
        # d_V = -fc*U + K*( (1-aa)*TAyt[itf] + aa*TAyt[itsup] ) - dissipation[1]
        d_y = d_U,d_V
        
        # def cond_print(it):
        #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
        # jax.lax.cond(it<=10, cond_print, lambda x:None, it)

        return d_y
