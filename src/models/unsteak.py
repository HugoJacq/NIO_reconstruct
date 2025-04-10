"""
This modules gather the definition of a unsteady ekmann solution 
to reconstruct inertial current from wind stress
Each model needs a forcing, from the module 'forcing.py'

The models are written in JAX to allow for automatic differentiation

inspired from:
Wang et al. 2023: https://www.mdpi.com/2072-4292/15/18/4526

By: Hugo Jacquet march 2025

Hypothesis:
- dissipation is done through friction between layer, and at the bottom of the model (with ocean interior)
- if needed, we use a parametrization of turbulent stress with as a diffusion*shear of mean flow (K*dU/dz)

"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, lax
# from functools import partial
import equinox as eqx
from diffrax import ODETerm, diffeqsolve, Euler
import diffrax
from functools import partial


from constants import *
from basis import kt_1D_to_2D, pkt2Kt_matrix
from tools import compute_hgrad

class junsteak(eqx.Module):
    # control
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray
    TAy : jnp.ndarray
    fc : jnp.ndarray
    dt_forcing : jnp.ndarray  
    nl : np.int32            = eqx.static_field() 
    AD_mode : str           
    t0 : jnp.ndarray          
    t1 : jnp.ndarray          
    dt : jnp.ndarray         
    
    use_difx : bool
    
    # variables declaration
    
    def __init__(self, pk, TAx, TAy, fc, dt_forcing, nl, AD_mode, call_args=(0.0,oneday,60.), use_difx=False):
        self.pk = pk
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        
        t0,t1,dt = call_args
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        
        self.use_difx = use_difx 
        
        
    @eqx.filter_jit            
    def __call__(self, save_traj_at = None):
        """
        """
        # run parameters
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = self.dt_forcing // dt
        maxstep = int((t1-t0)//dt) +1  
        
        # control parameters
        K = jnp.exp( jnp.asarray(self.pk) )
        
        # forcing fields
        args = self.fc, K, self.TAx, self.TAy, nsubsteps
        
        # DIFFRAX CODE
        if self.use_difx:
            y0 = jnp.zeros(self.nl), jnp.zeros(self.nl) # 0.0, 0.0 # initialisation
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(0., self.t1-self.t0,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=0., 
                            t1=self.t1-self.t0,  
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(0., self.t1-self.t0+dt, dt)),
                            adjoint=adjoint,    # here this is needed to be able to forward AD
                            max_steps=maxstep,
                            made_jump=False).ys 
        
        # NOT DIFFRAX CODE
        else:
            # saving at specific timesteps
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
                    
            # initialisation at null current
            U, V = jnp.zeros((Nforcing, self.nl)), jnp.zeros((Nforcing, self.nl))
            # inner loop at dt
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            # outer loop at dt_forcing
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, X0
            
            # loop on forcing time steps
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            # reducing the size of output to selected timesteps
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
                    
        return solution
        
    
  
    def vector_field(self, t, C, args):
        """
        """        
        # Definition of each layers
        def Onelayer(args0):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = args0
            ik = 0
            d_U = d_U.at[ik].set(  fc*V[ik] +K[2*ik]*TAxnow - K[2*ik+1]*(U[ik]) )
            d_V = d_U.at[ik].set( - fc*U[ik] +K[2*ik]*TAynow - K[2*ik+1]*(V[ik]))
            return d_U, d_V
            
        def Nlayer_midlayers_for_scan(carry, ik):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = carry
            
            d_U = d_U.at[ik].set(  fc*V[ik]                     # Coriolis
                                - K[2*ik]*(U[ik]-U[ik-1])       # top layer friction
                                - K[2*ik+1]*(U[ik]-U[ik+1]) )   # bottom layer friction
            d_V = d_V.at[ik].set(- fc*U[ik] 
                                - K[2*ik]*(V[ik]-V[ik-1])
                                - K[2*ik+1]*(V[ik]-V[ik+1]) )
            X = U, V, K, fc, TAxnow, TAynow, d_U, d_V
            return X, X
            
        def Nlayer(args0, nl):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = args0
            
            # surface
            ik = 0
            d_U = d_U.at[ik].set(   fc*V[ik]                        # Coriolis
                                    + K[2*ik]*TAxnow                # top layer friction
                                    - K[2*ik+1]*(U[ik]-U[ik+1]) )   # bottom layer friction
            d_V = d_V.at[ik].set( - fc*U[ik] 
                                    + K[2*ik]*TAynow
                                    - K[2*ik+1]*(V[ik]-V[ik+1]))
            # bottom
            ik = -1
            d_U = d_U.at[ik].set(   fc*V[ik] 
                                    - K[2*ik]*(U[ik]-U[ik-1])
                                    - K[2*ik+1]*U[ik] )
            d_V = d_V.at[ik].set( - fc*U[ik] 
                                    - K[2*ik]*(V[ik]-V[ik-1])
                                    - K[2*ik+1]*V[ik] )    
            # in between
            X0 = U, V, K, fc, TAxnow, TAynow, d_U, d_V
            final, _ = lax.scan( lambda carry, it: Nlayer_midlayers_for_scan(carry, it), X0, jnp.arange(1,nl-1) )
            _, _, _, _, _, _, d_U, d_V = final
            
            return d_U, d_V
            
        
        # gather args and variables
        U,V = C
        fc, K, TAx, TAy, nsubsteps = args
        
        # on-the-fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAx), -1, itf+1) 
        TAxnow = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAynow = (1-aa)*TAy[itf] + aa*TAy[itsup]
        
        # initialisation: current RHS of equation
        d_U, d_V = jnp.zeros(self.nl), jnp.zeros(self.nl)
        
        arg2 = U, V, K, fc, TAxnow, TAynow, d_U, d_V
        # loop on layers
        d_U, d_V = lax.cond( self.nl == 1,                  # condition, only 1 layer ?
                            lambda: Onelayer(arg2),         # if 1 layer, ik=0
                            lambda: Nlayer(arg2, self.nl)   # else, loop on layers
                            )  
        
        def cond_print():
            jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0], fc*V[0], K[0]*TAxnow, - K[1]*(U[0]-U[1]))
            jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0], fc*V[0], K[0]*TAxnow, - K[1]*(U[0]))
        #jax.lax.cond(it<=10, cond_print, lambda:None)
        
        
         
        return d_U, d_V
    
class junsteak_kt(eqx.Module):
    # control
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray
    TAy : jnp.ndarray
    fc : jnp.ndarray
    dTK : float 
    NdT : jnp.ndarray
    dt_forcing : jnp.ndarray  
    nl : np.int32            = eqx.static_field() 
    AD_mode : str           
    t0 : jnp.ndarray          
    t1 : jnp.ndarray          
    dt : jnp.ndarray         
    
    use_difx : bool
    k_base : str
    
    def __init__(self, pk, TAx, TAy, fc, dTK, dt_forcing, nl, AD_mode, call_args, use_difx=False, k_base='gauss'):
        t0,t1,dt = call_args
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        
        self.pk = pk
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        
        self.dTK = dTK
        self.NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK)   NdT = 
        
        self.use_difx = use_difx 
        self.k_base = k_base
        
    @eqx.filter_jit            
    def __call__(self, save_traj_at = None):
        """
        """
        # run parameters
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = self.dt_forcing // dt
        maxstep = int((t1-t0)//dt) +1  
        
        # control parameters
        K = jnp.exp( jnp.asarray(self.pk) )
        K = kt_1D_to_2D(K, NdT=self.NdT, npk=2*self.nl)
        M = pkt2Kt_matrix(NdT=self.NdT, dTK=self.dTK, t0=t0, t1=t1, dt_forcing=self.dt_forcing, base=self.k_base)
        Kt = jnp.dot(M,K)
        
        # forcing fields
        args = self.fc, Kt, self.TAx, self.TAy, nsubsteps
        
        # DIFFRAX CODE
        if self.use_difx:
            y0 = jnp.zeros(self.nl), jnp.zeros(self.nl) # 0.0, 0.0 # initialisation
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(0., self.t1-self.t0,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=0., 
                            t1=self.t1-self.t0,  
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(0., self.t1-self.t0+dt, dt)),
                            adjoint=adjoint,    # here this is needed to be able to forward AD
                            max_steps=maxstep,
                            made_jump=False).ys 
        
        # NOT DIFFRAX CODE
        else:
            # saving at specific timesteps
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
                    
            # initialisation at null current
            U, V = jnp.zeros((Nforcing, self.nl)), jnp.zeros((Nforcing, self.nl))
            
            # inner loop at dt
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            # outer loop at dt_forcing
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, X0
            
            # loop on forcing time steps
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            # reducing the size of output to selected timesteps
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
                    
        return solution
        
    
  
    def vector_field(self, t, C, args):
        """
        """        
        # Definition of each layers
        def Onelayer(args0):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = args0
            ik = 0
            d_U = d_U.at[ik].set(  fc*V[ik] +K[2*ik]*TAxnow - K[2*ik+1]*(U[ik]) )
            d_V = d_U.at[ik].set( - fc*U[ik] +K[2*ik]*TAynow - K[2*ik+1]*(V[ik]))
            return d_U, d_V
            
        def Nlayer_midlayers_for_scan(carry, ik):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = carry
            
            d_U = d_U.at[ik].set(  fc*V[ik]                     # Coriolis
                                - K[2*ik]*(U[ik]-U[ik-1])       # top layer friction
                                - K[2*ik+1]*(U[ik]-U[ik+1]) )   # bottom layer friction
            d_V = d_V.at[ik].set(- fc*U[ik] 
                                - K[2*ik]*(V[ik]-V[ik-1])
                                - K[2*ik+1]*(V[ik]-V[ik+1]) )
            X = U, V, K, fc, TAxnow, TAynow, d_U, d_V
            return X, X
            
        def Nlayer(args0, nl):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = args0
            
            # surface
            ik = 0
            d_U = d_U.at[ik].set(   fc*V[ik]                        # Coriolis
                                    + K[2*ik]*TAxnow                # top layer friction
                                    - K[2*ik+1]*(U[ik]-U[ik+1]) )   # bottom layer friction
            d_V = d_V.at[ik].set( - fc*U[ik] 
                                    + K[2*ik]*TAynow
                                    - K[2*ik+1]*(V[ik]-V[ik+1]))
            # bottom
            ik = -1
            d_U = d_U.at[ik].set(   fc*V[ik] 
                                    - K[2*ik]*(U[ik]-U[ik-1])
                                    - K[2*ik+1]*U[ik] )
            d_V = d_V.at[ik].set( - fc*U[ik] 
                                    - K[2*ik]*(V[ik]-V[ik-1])
                                    - K[2*ik+1]*V[ik] )    
            # in between
            X0 = U, V, K, fc, TAxnow, TAynow, d_U, d_V
            final, _ = lax.scan( lambda carry, it: Nlayer_midlayers_for_scan(carry, it), X0, jnp.arange(1,nl-1) )
            _, _, _, _, _, _, d_U, d_V = final
            
            return d_U, d_V
            
        
        # gather args and variables
        U,V = C
        fc, Kt, TAx, TAy, nsubsteps = args
        
        # on-the-fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAx), -1, itf+1) 
        TAxnow = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAynow = (1-aa)*TAy[itf] + aa*TAy[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        
        # initialisation: current RHS of equation
        d_U, d_V = jnp.zeros(self.nl), jnp.zeros(self.nl)
        
        arg2 = U, V, Ktnow, fc, TAxnow, TAynow, d_U, d_V
        # loop on layers
        d_U, d_V = lax.cond( self.nl == 1,                  # condition, only 1 layer ?
                            lambda: Onelayer(arg2),         # if 1 layer, ik=0
                            lambda: Nlayer(arg2, self.nl)   # else, loop on layers
                            )  
        # debug print
        def cond_print():
            jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0], fc*V[0], Ktnow[0]*TAxnow, - Ktnow[1]*(U[0]-U[1]))
            jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0], fc*V[0], Ktnow[0]*TAxnow, - Ktnow[1]*(U[0]))
        #jax.lax.cond(it<=10, cond_print, lambda:None)
        
        return d_U, d_V
    
class junsteak_kt_2D(eqx.Module):
     # control
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray
    TAy : jnp.ndarray
    fc : jnp.ndarray
    dTK : float 
    NdT : jnp.ndarray
    nx : jnp.ndarray
    ny : jnp.ndarray
    dt_forcing : jnp.ndarray  
    nl : np.int32            = eqx.static_field() 
    AD_mode : str           
    t0 : jnp.ndarray          
    t1 : jnp.ndarray          
    dt : jnp.ndarray         
    
    use_difx : bool
    k_base : str
    
    def __init__(self, pk, TAx, TAy, fc, dTK, dt_forcing, nl, AD_mode, call_args, use_difx=False, k_base='gauss'):
        t0,t1,dt = call_args
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        
        self.pk = pk
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        
        self.dTK = dTK
        self.NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK)   NdT = 
        shape = TAx.shape
        self.nx = shape[-1]
        self.ny = shape[-2]
        
        self.use_difx = use_difx 
        self.k_base = k_base
        
    #@eqx.filter_jit            
    def __call__(self, save_traj_at = None):
        """
        """
        # run parameters
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = self.dt_forcing // dt
        maxstep = int((t1-t0)//dt) +1  
        
        # control parameters
        K = jnp.exp( jnp.asarray(self.pk) )
        K = kt_1D_to_2D(K, NdT=self.NdT, npk=2*self.nl)
        M = pkt2Kt_matrix(NdT=self.NdT, dTK=self.dTK, t0=t0, t1=t1, dt_forcing=self.dt_forcing, base=self.k_base)
        Kt = jnp.dot(M,K)
        
        # forcing fields
        args = self.fc, Kt, self.TAx, self.TAy, nsubsteps
        
        # DIFFRAX CODE
        if self.use_difx:
            y0 = jnp.zeros((self.nl,self.ny,self.nx)), jnp.zeros((self.nl,self.ny,self.nx)) # initialisation
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(0., self.t1-self.t0,save_traj_at)) 
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=0., 
                            t1=self.t1-self.t0,  
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(0., self.t1-self.t0+dt, dt)),
                            adjoint=adjoint,    # here this is needed to be able to forward AD
                            max_steps=maxstep,
                            made_jump=False).ys 
        
        # NOT DIFFRAX CODE
        else:
            # saving at specific timesteps
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
                    
            # initialisation at null current
            U, V = jnp.zeros((Nforcing, self.nl, self.ny, self.nx)), jnp.zeros((Nforcing, self.nl, self.ny, self.nx))
            
            # inner loop at dt
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            # outer loop at dt_forcing
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, None #X0, X0
            
            # loop on forcing time steps
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            # reducing the size of output to selected timesteps
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
                    
        return solution      
        
        
    def vector_field(self, t, C, args):
        """
        """        
        # Definition of each layers
        def Onelayer(args0):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = args0
            
            ik = 0
            d_U = d_U.at[ik].set(  fc*V[ik] +K[2*ik]*TAxnow - K[2*ik+1]*(U[ik]) )
            d_V = d_U.at[ik].set( - fc*U[ik] +K[2*ik]*TAynow - K[2*ik+1]*(V[ik]))
            return d_U, d_V
            
        def Nlayer_midlayers_for_scan(carry, ik):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = carry
            
            d_U = d_U.at[ik].set(  fc*V[ik]                     # Coriolis
                                - K[2*ik]*(U[ik]-U[ik-1])       # top layer friction
                                - K[2*ik+1]*(U[ik]-U[ik+1]) )   # bottom layer friction
            d_V = d_V.at[ik].set(- fc*U[ik] 
                                - K[2*ik]*(V[ik]-V[ik-1])
                                - K[2*ik+1]*(V[ik]-V[ik+1]) )
            X = U, V, K, fc, TAxnow, TAynow, d_U, d_V
            return X, None #X
            
        def Nlayer(args0, nl):
            """  """
            U, V, K, fc, TAxnow, TAynow, d_U, d_V = args0
            # surface
            ik = 0
            d_U = d_U.at[ik].set(   fc*V[ik]                        # Coriolis
                                    + K[2*ik]*TAxnow                # top layer friction
                                    - K[2*ik+1]*(U[ik]-U[ik+1]) )   # bottom layer friction
            d_V = d_V.at[ik].set( - fc*U[ik] 
                                    + K[2*ik]*TAynow
                                    - K[2*ik+1]*(V[ik]-V[ik+1]))
            # bottom
            ik = -1
            d_U = d_U.at[ik].set(   fc*V[ik] 
                                    - K[2*ik]*(U[ik]-U[ik-1])
                                    - K[2*ik+1]*U[ik] )
            d_V = d_V.at[ik].set( - fc*U[ik] 
                                    - K[2*ik]*(V[ik]-V[ik-1])
                                    - K[2*ik+1]*V[ik] )    
            # in between
            X0 = U, V, K, fc, TAxnow, TAynow, d_U, d_V
            final, _ = lax.scan( lambda carry, it: Nlayer_midlayers_for_scan(carry, it), X0, jnp.arange(1,nl-1) )
            _, _, _, _, _, _, d_U, d_V = final
            
            return d_U, d_V
            
        
        # gather args and variables
        U,V = C
        fc, Kt, TAx, TAy, nsubsteps = args
        
        # on-the-fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAx), -1, itf+1) 
        TAxnow = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAynow = (1-aa)*TAy[itf] + aa*TAy[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        
        # initialisation: current RHS of equation
        d_U, d_V = jnp.zeros((self.nl, self.ny, self.nx)), jnp.zeros((self.nl, self.ny, self.nx))
        
        arg2 = U, V, Ktnow, fc, TAxnow, TAynow, d_U, d_V
        
        # loop on layers
        d_U, d_V = lax.cond( self.nl == 1,                  # condition, only 1 layer ?
                            lambda: Onelayer(arg2),         # if 1 layer, ik=0
                            lambda: Nlayer(arg2, self.nl)   # else, loop on layers
                            )  
        # debug print
        def cond_print():
            jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0], fc*V[0], Ktnow[0]*TAxnow, - Ktnow[1]*(U[0]-U[1]))
            jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0], fc*V[0], Ktnow[0]*TAxnow, - Ktnow[1]*(U[0]))
        #jax.lax.cond(it<=10, cond_print, lambda:None)
        
        return d_U, d_V
    
class junsteak_kt_2D_adv(eqx.Module):
     # control
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray
    TAy : jnp.ndarray
    fc : jnp.ndarray
    dTK : float 
    NdT : jnp.ndarray
    nx : jnp.ndarray
    ny : jnp.ndarray
    Ug : jnp.ndarray         
    Vg : jnp.ndarray  
    dx : jnp.ndarray
    dy : jnp.ndarray
    dt_forcing : jnp.ndarray  
    nl : np.int32            = eqx.static_field() 
    AD_mode : str           
    t0 : jnp.ndarray          
    t1 : jnp.ndarray          
    dt : jnp.ndarray         
    
    use_difx : bool
    k_base : str
    
    def __init__(self, pk, TAx, TAy, fc, Ug, Vg, dx, dy, dTK, dt_forcing, nl, AD_mode, call_args, use_difx=False, k_base='gauss'):
        t0,t1,dt = call_args
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        
        self.pk = pk
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        self.Ug = Ug
        self.Vg = Vg
        
        self.dx = dx
        self.dy = dy
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        
        self.dTK = dTK
        self.NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK)   NdT = 
        shape = TAx.shape
        self.nx = shape[-1]
        self.ny = shape[-2]
        
        self.use_difx = use_difx 
        self.k_base = k_base
        
    #@eqx.filter_jit            
    def __call__(self, save_traj_at = None):
        """
        """
        # run parameters
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = self.dt_forcing // dt
        maxstep = int((t1-t0)//dt) +1  
        
        # control parameters
        K = jnp.exp( jnp.asarray(self.pk) )
        K = kt_1D_to_2D(K, NdT=self.NdT, npk=2*self.nl)
        M = pkt2Kt_matrix(NdT=self.NdT, dTK=self.dTK, t0=t0, t1=t1, dt_forcing=self.dt_forcing, base=self.k_base)
        Kt = jnp.dot(M,K)
        
        # compute gradient of geostrophy
        gradUgt, gradVgt = compute_hgrad(self.Ug, self.dx, self.dy), compute_hgrad(self.Vg, self.dx, self.dy)
        
        # forcing fields
        args = self.fc, Kt, self.TAx, self.TAy, gradUgt, gradVgt, nsubsteps
        
        # DIFFRAX CODE
        if self.use_difx:
            y0 = jnp.zeros((self.nl,self.ny,self.nx)), jnp.zeros((self.nl,self.ny,self.nx)) # initialisation
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(0., self.t1-self.t0,save_traj_at)) 
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=0., 
                            t1=self.t1-self.t0,  
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(0., self.t1-self.t0+dt, dt)),
                            adjoint=adjoint,    # here this is needed to be able to forward AD
                            max_steps=maxstep,
                            made_jump=False).ys 
        
        # NOT DIFFRAX CODE
        else:
            # saving at specific timesteps
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
                    
            # initialisation at null current
            U, V = jnp.zeros((Nforcing, self.nl, self.ny, self.nx)), jnp.zeros((Nforcing, self.nl, self.ny, self.nx))
            
            # inner loop at dt
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            # outer loop at dt_forcing
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, None #X0, X0
            
            # loop on forcing time steps
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            # reducing the size of output to selected timesteps
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
                    
        return solution      
        
        
    def vector_field(self, t, C, args):
        """
        """        
        # Definition of each layers
        def Onelayer(args0):
            """  """
            U, V, K, fc, TAxnow, TAynow, gradUgnow, gradVgnow, d_U, d_V = args0
            
            ik = 0
            d_U = d_U.at[ik].set(   fc*V[ik] +K[2*ik]*TAxnow - K[2*ik+1]*U[ik] - U[ik]*gradUgnow[0] - V[ik]*gradUgnow[1])
            d_V = d_U.at[ik].set( - fc*U[ik] +K[2*ik]*TAynow - K[2*ik+1]*V[ik] - V[ik]*gradVgnow[0] - V[ik]*gradVgnow[1])
            return d_U, d_V
            
        def Nlayer_midlayers_for_scan(carry, ik):
            """  """
            U, V, K, fc, TAxnow, TAynow, gradUgnow, gradVgnow, d_U, d_V = carry
            
            d_U = d_U.at[ik].set(  fc*V[ik]                     # Coriolis
                                - K[2*ik]*(U[ik]-U[ik-1])       # top layer friction
                                - K[2*ik+1]*(U[ik]-U[ik+1]) )   # bottom layer friction
            d_V = d_V.at[ik].set(- fc*U[ik] 
                                - K[2*ik]*(V[ik]-V[ik-1])
                                - K[2*ik+1]*(V[ik]-V[ik+1]) )
            X = U, V, K, fc, TAxnow, TAynow, gradUgnow, gradVgnow, d_U, d_V
            return X, None #X
            
        def Nlayer(args0, nl):
            """  """
            U, V, K, fc, TAxnow, TAynow, gradUgnow, gradVgnow, d_U, d_V = args0
            # surface
            ik = 0
            d_U = d_U.at[ik].set(   fc*V[ik]                            # Coriolis
                                    + K[2*ik]*TAxnow                    # top layer friction
                                    - K[2*ik+1]*(U[ik]-U[ik+1])         # bottom layer friction
                                    - U[ik]*gradUgnow[0] - V[ik]*gradUgnow[1])  # advection              
            d_V = d_V.at[ik].set( - fc*U[ik] 
                                    + K[2*ik]*TAynow
                                    - K[2*ik+1]*(V[ik]-V[ik+1])
                                    - V[ik]*gradVgnow[0] - V[ik]*gradVgnow[1])
            # bottom
            ik = -1
            d_U = d_U.at[ik].set(   fc*V[ik] 
                                    - K[2*ik]*(U[ik]-U[ik-1])
                                    - K[2*ik+1]*U[ik] )
            d_V = d_V.at[ik].set( - fc*U[ik] 
                                    - K[2*ik]*(V[ik]-V[ik-1])
                                    - K[2*ik+1]*V[ik] )    
            # in between
            X0 = U, V, K, fc, TAxnow, TAynow, gradUgnow, gradVgnow, d_U, d_V
            final, _ = lax.scan( lambda carry, it: Nlayer_midlayers_for_scan(carry, it), X0, jnp.arange(1,nl-1) )
            _, _, _, _, _, _, _, _, d_U, d_V = final
            
            return d_U, d_V
            
        
        # gather args and variables
        U,V = C
        fc, Kt, TAx, TAy, gradUg, gradVg, nsubsteps = args
        
        # on-the-fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAx), -1, itf+1) 
        TAxnow = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAynow = (1-aa)*TAy[itf] + aa*TAy[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        gradUgnow = (1-aa)*gradUg[0][itf] + aa*gradUg[0][itsup], (1-aa)*gradUg[1][itf] + aa*gradUg[1][itsup]
        gradVgnow = (1-aa)*gradVg[0][itf] + aa*gradVg[0][itsup], (1-aa)*gradVg[1][itf] + aa*gradVg[1][itsup]
        
        # initialisation: current RHS of equation
        d_U, d_V = jnp.zeros((self.nl, self.ny, self.nx)), jnp.zeros((self.nl, self.ny, self.nx))
        
        arg2 = U, V, Ktnow, fc, TAxnow, TAynow, gradUgnow, gradVgnow, d_U, d_V
        
        # loop on layers
        d_U, d_V = lax.cond( self.nl == 1,                  # condition, only 1 layer ?
                            lambda: Onelayer(arg2),         # if 1 layer, ik=0
                            lambda: Nlayer(arg2, self.nl)   # else, loop on layers
                            )  
        # debug print
        def cond_print():
            jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0], fc*V[0], Ktnow[0]*TAxnow, - Ktnow[1]*(U[0]-U[1]))
            jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0], fc*V[0], Ktnow[0]*TAxnow, - Ktnow[1]*(U[0]))
        #jax.lax.cond(it<=10, cond_print, lambda:None)
        
        return d_U, d_V
