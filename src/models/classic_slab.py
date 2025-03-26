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
from diffrax import Dopri5, ODETerm, diffeqsolve, Euler
import diffrax

from constants import *

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
    dt_forcing : jnp.array  
    nl : jnp.array         
    AD_mode : str           
    t0 : jnp.array          
    t1 : jnp.array          
    dt : jnp.array         
    
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
            itsup = lax.select(itf+1>=len(TAx), -1, itf+1) 
            TAx = (1-aa)*TAx[itf] + aa*TAx[itsup]
            TAy = (1-aa)*TAy[itf] + aa*TAy[itsup]
            # physic
            d_U = fc*V + K[0]*TAx - K[1]*U
            d_V = -fc*U + K[0]*TAy - K[1]*V
            d_y = d_U,d_V
            
            # def cond_print(it):
            #     jax.debug.print('it,itf, TA, {}, {}, {}',it,itf,(TAx,TAy))
            
            # jax.lax.cond(it<=10, cond_print, lambda x:None, it)

            return d_y
        
class jslab_kt(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.array
    # parameters
    TAx : jnp.array         
    TAy : jnp.array         
    fc : jnp.array         
    dTK : float         
    dt_forcing : jnp.array  
    nl : jnp.array         
    AD_mode : str          
    NdT : jnp.array        
    t0 : jnp.array         
    t1 : jnp.array         
    dt : jnp.array         
    
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
        K = kt_1D_to_2D(K, NdT=self.NdT, nl=self.nl)
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
        itsup = lax.select(itf+1>=len(TAx), -1, itf+1) 
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
    pk : jnp.array
    # forcing
    TAx : jnp.array         
    TAy : jnp.array         
    fc : jnp.array         
    dTK : jnp.array        
    dt_forcing : jnp.array  
    # model parameters
    nl : jnp.array         
    AD_mode : str          
    NdT : jnp.array    
    nx : jnp.array
    ny : jnp.array
    # run time parameters    
    t0 : jnp.array         
    t1 : jnp.array         
    dt : jnp.array         
    
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
        K = kt_1D_to_2D(K, NdT=self.NdT, nl=self.nl)
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
        itsup = lax.select(itf+1>=TAx.shape[0], -1, itf+1) 
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
   
    
# K(t)
def kt_ini(pk, NdT):
    a_2D = jnp.repeat(pk, NdT)
    return kt_2D_to_1D(a_2D)

def kt_1D_to_2D(vector_kt_1D, NdT, nl):
    return vector_kt_1D.reshape((NdT,nl*2))

def kt_2D_to_1D(vector_kt):
    return vector_kt.flatten()

@partial(jax.jit, static_argnames=['NdT', 'dTK', 't0', 't1', 'dt_forcing', 'base']) # 
def pkt2Kt_matrix(NdT, dTK, t0, t1, dt_forcing, base='gauss'):
        """
        Reduced basis matrix
        
        original numpy function:
        
            def pkt2Kt_matrix(dT, gtime):
                if dT<gtime[-1]-gtime[0] : gptime = numpy.arange(gtime[0], gtime[-1]+dT,dT)
                else: gptime = numpy.array([gtime[0]])
                nt=len(gtime)
                npt = len(gptime)
                M = numpy.zeros((nt,npt))
                # Ks=numpy.zeros((ny,nx))
                S=numpy.zeros((nt))
                for ip in range(npt):
                    distt = (gtime-gptime[ip])
                    iit = numpy.where((numpy.abs(distt) < (3*dT)))[0]
                    tmp = numpy.exp(-distt[iit]**2/dT**2)
                    S[iit] += tmp
                    M[iit,ip] += tmp
                M = (M.T / S.T).T
                return M
        
        To transform in JAX compatible code, we need to work on array dimensions.
        Short description of what is done in the first 'if' statement:
            if we have a period dT shorter than the total time 'gtime',
                then we create an array with time values spaced by dT.
                last value if 'gtime[-1]'.
            else
                no variation of K along 'gtime' so only 1 vector K
                
        Example with nt = 22, with 1 dot is 1 day, dT = 5 days:
        vector_K :
            X * * * * * * * * * * * * * * * * * * * * X  # 22 values
        vector_Kt :
            X - - - - * - - - - * - - - - * - - - - * X  # 6 values
        """
        if NdT>0:    
            gptime = jnp.arange(t0+ dTK/2, t1+ dTK/2,dTK) # here +dTK/2 so that gaussian are at the middle of dTK intervals
            #gptime = jnp.arange(t0, t1,dTK)
        else:
            gptime = jnp.array([t0])
        # gptime = lax.select(NdT>0, jnp.arange(t0+ dTK/2, t1,dTK), jnp.array([t0]))
        gtime = jnp.arange(t0,t1,dt_forcing)
        
        # # see : https://docs.kidger.site/equinox/faq/#how-to-use-non-array-modules-as-inputs-to-scancondwhile-etc
        # # |   so you need to use scan with a custom made lambda function
        # # v
        # #_, _, S, M = lax.fori_loop(0, npt, self.__step_pkt2Kt_matrix, arg0) 
        # arg0,_ = lax.scan(lambda it,arg0: __step_pkt2Kt_matrix(it,arg0), arg0, xs=jnp.arange(0,npt))
        
        if base=='gauss':
            # CLAUDE VERSION
            # Vectorize the distance calculation
            # Shape: (nt, npt)
            distt = gtime[:, jnp.newaxis] - gptime[jnp.newaxis, :]
            
            # Vectorized condition and exponential calculation
            #cond = jnp.abs(distt) < 3 * dTK
            cond = jnp.ones(distt.shape) * True
            #tmp = jnp.exp(-2*distt**2 / dTK**2) * cond  # The condition zeros out values outside range
            coeff = 0.5
            tmp = jnp.exp(-coeff*distt**2 / dTK**2) * cond
            # Sum across the appropriate axis for S
            S = jnp.sum(tmp, axis=1)
            
            # Division with proper broadcasting
            M = tmp / S[:, jnp.newaxis]      
        elif base=='id':
            
            M = jnp.zeros((len(gtime),len(gptime)))
            
            nsteps = int(dTK//dt_forcing)
            slice_ones = jnp.ones(nsteps)
            if NdT*nsteps>len(gtime):
                last_slice = jnp.ones(len(gtime)-(NdT-1)*nsteps )
            else:
                last_slice = slice_ones
            def __fn_scan(arg0, it):
                """
                """
                M = arg0
                imin = jnp.array(it*nsteps,int)
                myslice = lax.select( it*nsteps> len(gtime), last_slice, slice_ones)
                #jax.debug.print('M[:,it] {} \n myslice {} \n imin {}/{} \n', M[:,it], myslice, imin, len(gtime))
                #jax.debug.print('len(myslice) {}, imin= {} / {}', len(myslice), imin, len(gtime))
                update = lax.dynamic_update_slice(M[:,it], myslice, (imin,))                            
                M = M.at[:,it].set(update)
                
                return M, M
            
            arg0 = M
            final, _ = lax.scan( __fn_scan, arg0, xs=jnp.arange(0,len(gptime)))
            
            M = final
        elif base=='linearInterp':
            """
            """
        else:
            raise Exception(f'You want to use the reduced basis {base} but it is not coded in pkt2Kt_matrix, aborting ...')
        return M
