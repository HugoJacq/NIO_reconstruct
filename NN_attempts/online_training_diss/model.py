import jax.numpy as jnp
import jax
from jax import lax
import equinox as eqx
from jaxtyping import Array, Float
from functools import partial
import diffrax

import sys
sys.path.insert(0, '../../src')

jax.config.update("jax_enable_x64", True)

from constants import *

# tester avec un MLP (dense layers)

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

    def __call__(self, x: Float[Array, "2 256 256"]) -> Float[Array, "2 256 256"]:
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
    Ug : jnp.ndarray            
    Vg : jnp.ndarray
    
    # t0 : jnp.ndarray          
    # t1 : jnp.ndarray          
    
    
    dissipation_model : DissipationNN
    
    dt_forcing : jnp.ndarray        = eqx.static_field()
    dt : jnp.ndarray                = eqx.static_field()
    use_difx : bool                 = eqx.static_field()
    
    def __init__(self, K0, TAx, TAy, Ug, Vg, fc, dt_forcing, dissipationNN, dt): #, call_args
        self.K0 = K0
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        self.Ug = Ug
        self.Vg = Vg
        self.dt_forcing = dt_forcing            
        # t0,t1,dt = call_args
        # self.t0 = t0
        # self.t1 = t1
        self.dt = dt
        
        self.dissipation_model = dissipationNN

        self.use_difx = False
    
    #@jax.checkpoint
    # @partial(eqx.filter_jit, static_argnums=0)
    #@partial(jax.jit, static_argnums=(1,2))
    # def __call__(self, call_args=[0.0,oneday], save_traj_at = None):
    @eqx.filter_jit
    def __call__(self, t0=0.0, t1=oneday, save_traj_at = None):

        nsubsteps = self.dt_forcing // self.dt
        
        # control
        K = jnp.exp( jnp.asarray(self.K0) )
  
        args = self.fc, K, self.TAx, self.TAy, self.Ug, self.Vg, nsubsteps, self.dissipation_model

        Nforcing = int((t1-t0)//self.dt_forcing)
        maxstep = int((t1-t0)//self.dt) +1 
        
        if self.use_difx:
            solver = diffrax.Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(ts=jnp.arange(0., t1-t0, self.dt_forcing))
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(0., t1-t0, save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            #adjoint = diffrax.ForwardMode()
            adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=40000) # checkpoints=10
            U,V = diffrax.diffeqsolve(terms=diffrax.ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=0., 
                            t1=t1-t0, 
                            y0=(jnp.zeros(self.TAx[0,:,:].shape), jnp.zeros(self.TAx[0,:,:].shape)), 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(0., t1-t0+self.dt, self.dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys
        
        else:
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
                return X1, None #X1, X1
            
            # outer loop at dt_forcing
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                X0 = U.at[iout+1].set(newU), V.at[iout+1].set(newV)
                return X0, None #X0, X0
                 
            final, _ = eqx.internal.scan(__outer_loop, init=(U,V), xs=jnp.arange(0,Nforcing),
                                         kind='lax') #  lax 'checkpointed' # <- to use checkpoints
                
            
            U,V = final

        if save_traj_at is None:
            step_save_out = 1
        else:
            if save_traj_at<self.dt_forcing:
                raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
            else:
                step_save_out = int(save_traj_at//self.dt_forcing)   
        
        solution = U[::step_save_out], V[::step_save_out]   
        
        return solution

    # vector field is common whether we use diffrax or not
    def vector_field(self, t, C, args):
        U,V = C
        fc, K, TAxt, TAyt, Ugt, Vgt, nsubsteps, dissipation_model = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAxt), -1, itf+1) 
        TAxnow = (1-aa)*TAxt[itf] + aa*TAxt[itsup]
        TAynow = (1-aa)*TAyt[itf] + aa*TAyt[itsup]
        Ugnow = (1-aa)*Ugt[itf] + aa*Ugt[itsup]
        Vgnow = (1-aa)*Vgt[itf] + aa*Vgt[itsup]
        # evaluate the NN at specific forcing
        
        input = jnp.stack([Ugnow,Vgnow])
        # mean = input.mean()
        # std = input.std()
        # input = (input-mean)/std
        dissipation_undim = dissipation_model(input)
        #print(dissipation.shape)
        #dissipation_term = dissipation_undim*std + mean
        
        # physic
        d_U = fc*V + K*TAxnow  - dissipation_undim[0]
        d_V = -fc*U + K*TAynow - dissipation_undim[1]
        # d_U = fc*V + K*( (1-aa)*TAxt[itf] + aa*TAxt[itsup] )  - dissipation[0]
        # d_V = -fc*U + K*( (1-aa)*TAyt[itf] + aa*TAyt[itsup] ) - dissipation[1]
        d_y = d_U,d_V
        
        # def cond_print(it):
        #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
        # jax.lax.cond(it<=10, cond_print, lambda x:None, it)

        return d_y
