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

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, lax
# from functools import partial
import equinox as eqx
from diffrax import ODETerm, diffeqsolve, Euler
import diffrax

from basis import kt_1D_to_2D, pkt2Kt_matrix
from constants import *

class jslab(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray
    TAy : jnp.ndarray
    fc : jnp.ndarray
    dt_forcing : jnp.ndarray  
    nl : jnp.ndarray         
    AD_mode : str           
    t0 : jnp.ndarray          
    t1 : jnp.ndarray          
    dt : jnp.ndarray         
    
    use_difx : bool
    
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
        
        self.use_difx = use_difx # use_difx
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None):
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = self.dt_forcing // dt
        
        y0 = 0.0, 0.0 # self.U0,self.V0
        # control
        K = jnp.exp( jnp.asarray(self.pk) )
  
        args = self.fc, K, self.TAx, self.TAy, nsubsteps
        
        maxstep = int((t1-t0)//dt) +1 
        
        
        if self.use_difx:
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=t0, 
                            t1=t1, 
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys # here this is needed to be able to forward AD
        else:
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
            
            # initialisation at null current
            U, V = jnp.zeros(Nforcing), jnp.zeros(Nforcing)
            
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
            
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
            
            
        return solution

    # vector field is common whether we use diffrax or not
    def vector_field(self, t, C, args):
        U,V = C
        fc, K, TAxt, TAyt, nsubsteps = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAxt), -1, itf+1) 
        TAx = (1-aa)*TAxt[itf] + aa*TAxt[itsup]
        TAy = (1-aa)*TAyt[itf] + aa*TAyt[itsup]
        # physic
        d_U = fc*V + K[0]*TAx - K[1]*U
        d_V = -fc*U + K[0]*TAy - K[1]*V
        d_y = d_U,d_V
        
        # def cond_print(it):
        #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
        
        # jax.lax.cond(it<=10, cond_print, lambda x:None, it)

        return d_y

class jslab_fft(eqx.Module):
     # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # forcing
    TAx : jnp.ndarray         
    TAy : jnp.ndarray         
    fc : jnp.ndarray              
    dt_forcing : jnp.ndarray  
    # model parameters
    nl : jnp.ndarray         
    AD_mode : str          
    NdT : jnp.ndarray    
    # run time parameters    
    t0 : jnp.ndarray         
    t1 : jnp.ndarray         
    dt : jnp.ndarray         
    
    use_difx : bool 
    k_base : str
    
    def __init__(self, pk, TAx, TAy, fc, dt_forcing, nl, AD_mode, call_args, use_difx=False, k_base='gauss'):
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
        
        self.use_difx = use_difx
        self.k_base = k_base
    
    
    
    @eqx.filter_jit
    def __call__(self, save_traj_at = None): 
        """
        """ 
        
        if save_traj_at is None:
                step_save_out = 1
        else:
            if save_traj_at<self.dt_forcing:
                raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
            else:
                step_save_out = int(save_traj_at//self.dt_forcing)
        
        # forcing: time -> fourier space 
        TA      = self.TAx + 1j*self.TAy
        time    = jnp.arrange(self.t0, self.t1, self.dt)
        nt      = len(time)
        ffl     = jnp.arange(nt)/(nt*self.dt)
        wind_fft = jnp.fft(TA)
        # solving in fourier
        U1f = self.H(ffl) * wind_fft
        # going back to time space
        U,V = jnp.ifft(U1f)

        # get solution at 'save_traj_at'
        solution = U[::step_save_out], V[::step_save_out]
        
        return solution
        
    def H(self,s):
        K = self.pk
        return K[0] / (1j*(2*jnp.pi*s +self.fc) + K[1])
      
class jslab_kt(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray         
    TAy : jnp.ndarray         
    fc : jnp.ndarray         
    dTK : float         
    dt_forcing : jnp.ndarray  
    nl : jnp.ndarray         
    AD_mode : str          
    NdT : jnp.ndarray        
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
        
        self.dTK = dTK
        self.NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK)   NdT = 
        self.pk = pk #self.kt_ini( jnp.asarray(pk) )
        
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        
        self.use_difx = use_difx
        self.k_base = k_base
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None): #call_args, 

        y0 = 0.0,0.0 # self.U0,self.V0
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = int(self.dt_forcing // dt)
        # control
        K = jnp.exp( self.pk) 
        K = kt_1D_to_2D(K, NdT=self.NdT, npk=2*self.nl)
        M = pkt2Kt_matrix(NdT=self.NdT, dTK=self.dTK, t0=t0, t1=t1, dt_forcing=self.dt_forcing, base=self.k_base)
        Kt = jnp.dot(M,K)
        args = self.fc, Kt, self.TAx, self.TAy, nsubsteps
        
        maxstep = int((t1-t0)//dt) +1 
        
        
        if self.use_difx:
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10) # <- number of checkpoint still WIP
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=t0, 
                            t1=t1, 
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys # here this is needed to be able to forward AD
        else:
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
            
            U, V = jnp.zeros(Nforcing), jnp.zeros(Nforcing)
            
            
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, X0
            
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
            
        return solution

    def vector_field(self, t, C, args):
        U,V = C
        fc, Kt, TAx, TAy, nsubsteps = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAx), -1, itf+1) 
        TAx = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAy = (1-aa)*TAy[itf] + aa*TAy[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        # def cond_print(it):
        #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
        # jax.lax.cond(it<=10, cond_print, lambda x:None, it)
        
        # physic
        d_U = fc*V + Ktnow[0]*TAx - Ktnow[1]*U
        d_V = -fc*U + Ktnow[0]*TAy - Ktnow[1]*V
        d_y = d_U,d_V
        return d_y
    
    """
    Note sur pourquoi j'ai enlevé t0, t1, dt du __call__:
    Si on veut faire des opérations sur K qui font un changement de base, il faut savoir le nombre de timestep, et donc il faut que nt=(t1-t0)//dt soit static !
    
    Note sur pourquoi je n'utilise pas diffrax:
    les perfs sont entre x3 et x5 comparé à la version sans diffrax (use_difx=False)... Mais on perd la possibilité de faire du reverse AD avec checkpoints automatiques.
    """
    
class jslab_kt_2D(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # forcing
    TAx : jnp.ndarray         
    TAy : jnp.ndarray         
    fc : jnp.ndarray         
    dTK : jnp.ndarray        
    dt_forcing : jnp.ndarray  
    # model parameters
    nl : jnp.ndarray         
    AD_mode : str          
    NdT : jnp.ndarray    
    nx : jnp.ndarray
    ny : jnp.ndarray
    # run time parameters    
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
        
        self.dTK = dTK
        self.NdT = len(jnp.arange(t0, t1,dTK)) # int((t1-t0)//dTK)   NdT = 
        self.pk = pk #self.kt_ini( jnp.asarray(pk) )
        
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        shape = TAx.shape
        self.nx = shape[-1]
        self.ny = shape[-2]
        
        self.use_difx = use_difx
        self.k_base = k_base
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None): #call_args, 

        y0 = jnp.zeros((self.ny,self.nx)), jnp.zeros((self.ny,self.nx)) # self.U0,self.V0
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = int(self.dt_forcing // dt)
        # control
        K = jnp.exp( self.pk) 
        K = kt_1D_to_2D(K, NdT=self.NdT, npk=2*self.nl)
        M = pkt2Kt_matrix(NdT=self.NdT, dTK=self.dTK, t0=t0, t1=t1, dt_forcing=self.dt_forcing, base=self.k_base)
        Kt = jnp.dot(M,K)
        args = self.fc, Kt, self.TAx, self.TAy, nsubsteps
        
        maxstep = int((t1-t0)//dt) +1 
        
        
        if self.use_difx:
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10) # <- number of checkpoint still WIP
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=t0, 
                            t1=t1, 
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys # here this is needed to be able to forward AD
        else:
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
            
            U, V = jnp.zeros((Nforcing, self.ny, self.nx)), jnp.zeros((Nforcing, self.ny, self.nx))
            
            
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, X0
            
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
            
        return solution

    def vector_field(self, t, C, args):
        U,V = C
        fc, Kt, TAx, TAy, nsubsteps = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=TAx.shape[0], -1, itf+1) 
        TAxt = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAyt = (1-aa)*TAy[itf] + aa*TAy[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        # def cond_print(it):
        #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
        # jax.lax.cond(it<=10, cond_print, lambda x:None, it)
        # print(U.shape, TAx.shape)
        # physic
        d_U = fc*V + Ktnow[0]*TAxt - Ktnow[1]*U
        d_V = -fc*U + Ktnow[0]*TAyt - Ktnow[1]*V
        d_y = d_U,d_V
        return d_y 
   
  
class jslab_rxry(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray
    TAy : jnp.ndarray
    fc : jnp.ndarray
    dt_forcing : jnp.ndarray  
    nl : jnp.ndarray         
    AD_mode : str           
    t0 : jnp.ndarray          
    t1 : jnp.ndarray          
    dt : jnp.ndarray         
    
    use_difx : bool
    
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
        
        self.use_difx = use_difx # use_difx
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None):
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = self.dt_forcing // dt
        
        y0 = 0.0, 0.0 # self.U0,self.V0
        # control
        K = jnp.exp( jnp.asarray(self.pk) )
  
        args = self.fc, K, self.TAx, self.TAy, nsubsteps
        
        maxstep = int((t1-t0)//dt) +1 
        
        
        if self.use_difx:
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=t0, 
                            t1=t1, 
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys # here this is needed to be able to forward AD
        else:
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
            
            # initialisation at null current
            U, V = jnp.zeros(Nforcing), jnp.zeros(Nforcing)
            
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
            
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
            
            
        return solution

    # vector field is common whether we use diffrax or not
    def vector_field(self, t, C, args):
            U,V = C
            fc, K, TAx, TAy, nsubsteps = args
            
            # on the fly interpolation
            it = jnp.array(t//self.dt, int)
            itf = jnp.array(it//nsubsteps, int)
            
            aa = jnp.mod(it,nsubsteps)/nsubsteps
            itsup = jnp.where(itf+1>=len(TAx), -1, itf+1) 
            TAx = (1-aa)*TAx[itf] + aa*TAx[itsup]
            TAy = (1-aa)*TAy[itf] + aa*TAy[itsup]
            # physic
            d_U = fc*V + K[0]*TAx - K[1]*U
            d_V = -fc*U + K[0]*TAy - K[2]*V
            d_y = d_U,d_V
            
            # def cond_print(it):
            #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
            
            # jax.lax.cond(it<=10, cond_print, lambda x:None, it)

            return d_y
 
class jslab_Ue_Unio(eqx.Module):
    """
    see D'Asaro 1985 https://doi.org/10.1175/1520-0485(1985)015<1043:TEFFTW>2.0.CO;2
    """
    
    
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray
    TAy : jnp.ndarray
    fc : jnp.ndarray
    dt_forcing : jnp.ndarray  
    nl : jnp.ndarray         
    AD_mode : str           
    t0 : jnp.ndarray          
    t1 : jnp.ndarray          
    dt : jnp.ndarray         
    
    use_difx : bool
    
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
        
        self.use_difx = use_difx # use_difx
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None):
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = self.dt_forcing // dt
        
        y0 = 0.0, 0.0 # self.U0,self.V0
        # control
        K = jnp.exp( jnp.asarray(self.pk) )
  
        
        # slow, ekman flow
        T = (self.TAx + 1j*self.TAy) /rho
        Ce = K[0] * T / (K[1] + 1j*self.fc)
        Ue = jnp.real(Ce)
        Ve = jnp.imag(Ce) 
        dCedt = jnp.gradient(Ce, self.dt, axis=0)
        dUedt = jnp.real(dCedt)
        dVedt = jnp.imag(dCedt)
        # forcing is already filtered at fc
        args = self.fc, K, self.TAx, self.TAy, dUedt, dVedt, nsubsteps 
        
        maxstep = int((t1-t0)//dt) +1 
        
        if self.use_difx:
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=t0, 
                            t1=t1, 
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys # here this is needed to be able to forward AD
        else:
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
            
            # initialisation at null current
            U, V = jnp.zeros(Nforcing), jnp.zeros(Nforcing)
            
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
            
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            Unio,Vnio = final
            
            
            
            
            
            # Ue = 1/self.fc * (K[2]*self.TAy + K[3]*self.TAx)
            # Ve = - 1/self.fc * (K[3]*self.TAy - K[2]*self.TAx)
            
            U,V = Ue+Unio, Ve+Vnio
            
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
            
            
        return solution

    # vector field is common whether we use diffrax or not
    def vector_field(self, t, C, args):
        U,V = C
        fc, K, TAx, TAy, dUedt, dVedt, nsubsteps = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAx), -1, itf+1) 
        
        
        # TAx = (1-aa)*TAx[itf] + aa*TAx[itsup]
        # TAy = (1-aa)*TAy[itf] + aa*TAy[itsup]
        # physic
        # fast, NIO flow. 
        # How to get the equation for Unio ? -> apply a fc filter on the slab model equation. 
        #   hypothesis: filtered K are null as they evolve slowly compared to inertial period
        #   here TAx and TAy are filtered at fc !
        # Note: theses equations are the same as the slab model (jslab)
        # d_U = fc*V + K[0]*TAx - K[1]*U
        # d_V = -fc*U + K[0]*TAy - K[1]*V
        
        # new test, d'Asaro 1985
        dUedt = (1-aa)*dUedt[itf] + aa*dUedt[itsup]
        dVedt = (1-aa)*dVedt[itf] + aa*dVedt[itsup]
        
        d_U = -K[1]*U + fc*V - dUedt
        d_V = -K[1]*V - fc*U - dVedt
        
        d_y = d_U,d_V
        
        # def cond_print(it):
        #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
        
        # jax.lax.cond(it<=10, cond_print, lambda x:None, it)

        return d_y
 
class jslab_kt_Ue_Unio(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # parameters
    TAx : jnp.ndarray         
    TAy : jnp.ndarray         
    fc : jnp.ndarray         
    dTK : float         
    dt_forcing : jnp.ndarray  
    nl : jnp.ndarray         
    AD_mode : str          
    NdT : jnp.ndarray        
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
        
        self.dTK = dTK
        self.NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK)   NdT = 
        self.pk = pk #self.kt_ini( jnp.asarray(pk) )
        
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        
        self.use_difx = use_difx
        self.k_base = k_base
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None): #call_args, 

        y0 = 0.0,0.0 # self.U0,self.V0
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = int(self.dt_forcing // dt)
        # control
        K = jnp.exp( self.pk) 
        # K = kt_1D_to_2D(K, NdT=self.NdT, npk = 4*self.nl)
        K = kt_1D_to_2D(K, NdT=self.NdT, npk = 2*self.nl)
        M = pkt2Kt_matrix(NdT=self.NdT, dTK=self.dTK, t0=t0, t1=t1, dt_forcing=self.dt_forcing, base=self.k_base)
        Kt = jnp.dot(M,K)
        
        # slow, ekman flow
        T = (self.TAx + 1j*self.TAy) #/ rho # PAPA is already tau/rho, Croco is only tau, To be fixed uniformly
        Ce = Kt[:,0] * T / (Kt[:,1] + 1j*self.fc)
        Ue = jnp.real(Ce)
        Ve = jnp.imag(Ce) 
        dCedt = jnp.gradient(Ce, self.dt, axis=0)
        dUedt = jnp.real(dCedt)
        dVedt = jnp.imag(dCedt)
        # forcing is already filtered at fc
        args = self.fc, Kt, self.TAx, self.TAy, dUedt, dVedt, nsubsteps 
        
        
        maxstep = int((t1-t0)//dt) +1 
        
        
        if self.use_difx:
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10) # <- number of checkpoint still WIP
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=t0, 
                            t1=t1, 
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys # here this is needed to be able to forward AD
        else:
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
            
            U, V = jnp.zeros(Nforcing), jnp.zeros(Nforcing)
            
            
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, X0
            
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            Unio, Vnio = final
            
            
            # # slow, ekman flow
            # Ue = 1/self.fc * (Kt[:, 2]*self.TAy + Kt[:, 3]*self.TAx)
            # Ve = - 1/self.fc * (Kt[:, 3]*self.TAy - Kt[:, 2]*self.TAx)
            
            U,V = Ue+Unio, Ve+Vnio
            
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
            
        return solution

    def vector_field(self, t, C, args):
        U,V = C
        fc, Kt, TAx, TAy, dUedt, dVedt, nsubsteps = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=len(TAx), -1, itf+1) 
        # TAx = (1-aa)*TAx[itf] + aa*TAx[itsup]
        # TAy = (1-aa)*TAy[itf] + aa*TAy[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        # def cond_print(it):
        #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
        # jax.lax.cond(it<=10, cond_print, lambda x:None, it)
        
        # physic
        # d_U = fc*V + Ktnow[0]*TAx - Ktnow[1]*U
        # d_V = -fc*U + Ktnow[0]*TAy - Ktnow[1]*V
        
        
        # new test, d'Asaro 1985
        dUedt = (1-aa)*dUedt[itf] + aa*dUedt[itsup]
        dVedt = (1-aa)*dVedt[itf] + aa*dVedt[itsup]
        
        d_U = -Ktnow[1]*U + fc*V - dUedt
        d_V = -Ktnow[1]*V - fc*U - dVedt
        
        d_y = d_U,d_V
        return d_y


# TO BE TESTED
class jslab_kt_2D_adv(eqx.Module):
   # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # forcing
    TAx : jnp.ndarray         
    TAy : jnp.ndarray     
    Ug : jnp.ndarray         
    Vg : jnp.ndarray  
    dx : jnp.ndarray
    dy : jnp.ndarray    
    fc : jnp.ndarray         
    dTK : jnp.ndarray        
    dt_forcing : jnp.ndarray  
    # model parameters
    nl : jnp.ndarray         
    AD_mode : str           = eqx.static_field() 
    NdT : jnp.ndarray    
    nx : jnp.ndarray
    ny : jnp.ndarray
    # run time parameters    
    t0 : jnp.ndarray         
    t1 : jnp.ndarray         
    dt : jnp.ndarray         
    
    use_difx : bool         = eqx.static_field()
    k_base : str            = eqx.static_field()
    
    def __init__(self, pk, TAx, TAy, Ug, Vg, dx, dy, fc, dTK, dt_forcing, nl, AD_mode, call_args, use_difx=False, k_base='gauss'):
        t0,t1,dt = call_args
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        
        self.dTK = dTK
        self.NdT = len(jnp.arange(t0, t1,dTK)) # int((t1-t0)//dTK)   NdT = 
        self.pk = pk #self.kt_ini( jnp.asarray(pk) )
        
        self.TAx = TAx
        self.TAy = TAy
        self.Ug = Ug
        self.Vg = Vg
        self.dx = dx
        self.dy = dy
        self.fc = fc
        
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        shape = TAx.shape
        self.nx = shape[-1]
        self.ny = shape[-2]
        
        self.use_difx = use_difx
        self.k_base = k_base
        
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None): #call_args, 

        y0 = jnp.zeros((self.ny,self.nx)), jnp.zeros((self.ny,self.nx)) # self.U0,self.V0
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = int(self.dt_forcing // dt)
        # control
        K = jnp.exp( self.pk) 
        K = kt_1D_to_2D(K, NdT=self.NdT, npk=2*self.nl)
        M = pkt2Kt_matrix(NdT=self.NdT, dTK=self.dTK, t0=t0, t1=t1, dt_forcing=self.dt_forcing, base=self.k_base)
        Kt = jnp.dot(M,K)
        
        # compute gradient of geostrophy
        gradUgt, gradVgt = compute_hgrad(self.Ug, self.dx, self.dy), compute_hgrad(self.Vg, self.dx, self.dy)
        
        args = self.fc, Kt, self.TAx, self.TAy, gradUgt, gradVgt, nsubsteps
        
        maxstep = int((t1-t0)//dt) +1 
        
        
        if self.use_difx:
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10) # <- number of checkpoint still WIP
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=t0, 
                            t1=t1, 
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys # here this is needed to be able to forward AD
        else:
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
            
            U, V = jnp.zeros((Nforcing, self.ny, self.nx)), jnp.zeros((Nforcing, self.ny, self.nx))
            
            
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, X0
            
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
            
        return solution

    #@partial(jax.jit, donate_argnames=['C']) # not much memory save bc C is small
    def vector_field(self, t, C, args):
        U,V = C
        fc, Kt, TAx, TAy, gradUg, gradVg, nsubsteps = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=TAx.shape[0], -1, itf+1) 
        TAxt = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAyt = (1-aa)*TAy[itf] + aa*TAy[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        gradUgt = (1-aa)*gradUg[0][itf] + aa*gradUg[0][itsup], (1-aa)*gradUg[1][itf] + aa*gradUg[1][itsup]
        gradVgt = (1-aa)*gradVg[0][itf] + aa*gradVg[0][itsup], (1-aa)*gradVg[1][itf] + aa*gradVg[1][itsup]
       
        
        
        # physic
        d_U =  fc*V + Ktnow[0]*TAxt - Ktnow[1]*U - (U*gradUgt[0] + V*gradUgt[1]) # Ktnow[0]*
        d_V = -fc*U + Ktnow[0]*TAyt - Ktnow[1]*V - (U*gradVgt[0] + V*gradVgt[1]) # Ktnow[0]*
        d_y = d_U,d_V
        
        def cond_print(it):
            jax.debug.print("d_U, Coriolis, stress, damping, adv: {}, {}, {}, {}, {}", d_U[0,0], fc[0]*V[0,0], Ktnow[0]*TAxt[0,0], - Ktnow[1]*U[0,0], - Ktnow[0]*(U[0,0]*gradUgt[0][0,0] + V[0,0]*gradUgt[1][0,0]))
        jax.lax.cond(it<=70, cond_print, lambda x:None, it)
        # print(U.shape, TAx.shape)
        
        return d_y 

# TO BE TESTED
class jslab_kt_2D_adv_Ut(eqx.Module):
    """
    Solving dU/dt = -ifU - K0*Tau - U*gradUg + ifUg
    U = total current (ageo + geo)
    
    """
    
    
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.ndarray
    # forcing
    TAx : jnp.ndarray         
    TAy : jnp.ndarray        
    Ug : jnp.ndarray         
    Vg : jnp.ndarray  
    dx : jnp.ndarray
    dy : jnp.ndarray  
    fc : jnp.ndarray         
    dTK : jnp.ndarray        
    dt_forcing : jnp.ndarray  
    # model parameters
    nl : jnp.ndarray         
    AD_mode : str          
    NdT : jnp.ndarray    
    nx : jnp.ndarray
    ny : jnp.ndarray
    # run time parameters    
    t0 : jnp.ndarray         
    t1 : jnp.ndarray         
    dt : jnp.ndarray         
    
    use_difx : bool 
    k_base : str
    
    def __init__(self, pk, TAx, TAy, Ug, Vg, dx, dy, fc, dTK, dt_forcing, nl, AD_mode, call_args, use_difx=False, k_base='gauss'):
        t0,t1,dt = call_args
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        
        self.dTK = dTK
        self.NdT = len(jnp.arange(t0, t1,dTK)) # int((t1-t0)//dTK)   NdT = 
        self.pk = pk #self.kt_ini( jnp.asarray(pk) )
        
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
        shape = TAx.shape
        self.nx = shape[-1]
        self.ny = shape[-2]
        
        self.use_difx = use_difx
        self.k_base = k_base
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None): #call_args, 

        y0 = jnp.zeros((self.ny,self.nx)), jnp.zeros((self.ny,self.nx)) # self.U0,self.V0
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = int(self.dt_forcing // dt)
        # control
        K = jnp.exp( self.pk) 
        K = kt_1D_to_2D(K, NdT=self.NdT, npk=2*self.nl)
        M = pkt2Kt_matrix(NdT=self.NdT, dTK=self.dTK, t0=t0, t1=t1, dt_forcing=self.dt_forcing, base=self.k_base)
        Kt = jnp.dot(M,K)
        
        # compute gradient of geostrophy
        gradUgt, gradVgt = compute_hgrad(self.Ug, self.dx, self.dy), compute_hgrad(self.Vg, self.dx, self.dy)
        
        
        args = self.fc, Kt, self.TAx, self.TAy, self.Ug, self.Vg, gradUgt, gradVgt, nsubsteps
        
        maxstep = int((t1-t0)//dt) +1 
        
        
        if self.use_difx:
            solver = Euler()
            if save_traj_at is None:
                saveat = diffrax.SaveAt(steps=True)
            else:
                saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
            # Auto-diff mode
            if self.AD_mode=='F':
                adjoint = diffrax.ForwardMode()
            else:
                adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10) # <- number of checkpoint still WIP
            
            solution = diffeqsolve(terms=ODETerm(self.vector_field), 
                            solver=solver, 
                            t0=t0, 
                            t1=t1, 
                            y0=y0, 
                            args=args, 
                            dt0=None, #dt, #dt, None
                            saveat=saveat,
                            stepsize_controller=diffrax.StepTo(jnp.arange(t0, t1+dt, dt)),
                            adjoint=adjoint,
                            max_steps=maxstep,
                            made_jump=False).ys # here this is needed to be able to forward AD
        else:
            Nforcing = int((t1-t0)//self.dt_forcing)
            if save_traj_at is None:
                step_save_out = 1
            else:
                if save_traj_at<self.dt_forcing:
                    raise Exception('You want to save at dt<dt_forcing, this is not available.\n Choose a bigger dt')
                else:
                    step_save_out = int(save_traj_at//self.dt_forcing)
            
            U, V = jnp.zeros((Nforcing, self.ny, self.nx)), jnp.zeros((Nforcing, self.ny, self.nx))
            
            
            def __inner_loop(carry, iin):
                Uold, Vold, iout = carry
                t = iout*self.dt_forcing + iin*self.dt
                C = Uold, Vold
                d_U,d_V = self.vector_field(t, C, args)
                newU,newV = Uold + self.dt*d_U, Vold + self.dt*d_V # Euler hard coded
                X1 = newU,newV,iout
                return X1, X1
            
            def __outer_loop(carry, iout):
                U,V = carry
                X1 = U[iout], V[iout], iout
                final, _ = lax.scan(__inner_loop, X1, jnp.arange(0,nsubsteps)) #jnp.arange(0,self.nt-1))
                newU, newV, _ = final
                U = U.at[iout+1].set(newU)
                V = V.at[iout+1].set(newV)
                X0 = U,V
                return X0, X0
            
            X1 = U, V
            final, _ = lax.scan(__outer_loop, X1, jnp.arange(0,Nforcing))
            U,V = final
            
            if save_traj_at is None:
                solution = U,V
            else:
                solution = U[::step_save_out], V[::step_save_out]
            
        return solution

    def vector_field(self, t, C, args):
        U,V = C
        fc, Kt, TAx, TAy, Ug, Vg, gradUgt, gradVgt, nsubsteps = args
        
        # on the fly interpolation
        it = jnp.array(t//self.dt, int)
        itf = jnp.array(it//nsubsteps, int)
        aa = jnp.mod(it,nsubsteps)/nsubsteps
        itsup = jnp.where(itf+1>=TAx.shape[0], -1, itf+1) 
        TAxt = (1-aa)*TAx[itf] + aa*TAx[itsup]
        TAyt = (1-aa)*TAy[itf] + aa*TAy[itsup]
        Ktnow = (1-aa)*Kt[it-1] + aa*Kt[itsup]
        Ugnow = (1-aa)*Ug[it-1] + aa*Ug[itsup]
        Vgnow = (1-aa)*Vg[it-1] + aa*Vg[itsup]
        gradUgnow = (1-aa)*gradUgt[0][itf] + aa*gradUgt[0][itsup], (1-aa)*gradUgt[1][itf] + aa*gradUgt[1][itsup]
        gradVgnow = (1-aa)*gradVgt[0][itf] + aa*gradVgt[0][itsup], (1-aa)*gradVgt[1][itf] + aa*gradVgt[1][itsup]
        
        # physic
        d_U = fc*V + Ktnow[0]*TAxt - Ktnow[1]*U - fc*Vgnow #- (U*gradUgnow[0] + V*gradUgnow[1])
        d_V = -fc*U + Ktnow[0]*TAyt - Ktnow[1]*V  + fc*Ugnow #- (U*gradVgnow[0] + V*gradVgnow[1])
        
        # def cond_print(it):
        #     #jax.debug.print("d_U, Coriolis, stress, damping, adv: {}, {}, {}, {}, {}", d_U[0,0], fc[0]*V[0,0], Ktnow[0]*TAxt[0,0], - Ktnow[1]*U[0,0], - Ktnow[0]*(U[0,0]*gradUgt[0][0,0] + V[0,0]*gradUgt[1][0,0]))
        #     jax.debug.print("d_U, Coriolis, stress, damping: {}, {}, {}, {}", d_U[0,0], fc[0]*V[0,0], Ktnow[0]*TAxt[0,0], - Ktnow[1]*U[0,0])
        # jax.lax.cond(it<=70, cond_print, lambda x:None, it)
        
        d_y = d_U, d_V
        return d_y 



# class jslab_kt_fft(eqx.Module):
# -> k(t) donc FFT(K*tau) <=> FFT(K) convolution FFT(tau)
# et pareil pour un potentiel terme en grad Ug
 

# this could be moved to a different file
def compute_hgrad(Ug, dx, dy):
    # we assume that geostrophic current is on regular lat longrid
    dUgdx = jnp.gradient(Ug, dx, axis=-1)
    dUgdy = jnp.gradient(Ug, dy, axis=-2)
    return dUgdx, dUgdy