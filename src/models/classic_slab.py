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
    dt_forcing : jnp.array  = eqx.static_field()
    nl : jnp.array          = eqx.static_field()
    AD_mode : str           = eqx.static_field()
    t0 : jnp.array          = eqx.static_field()
    t1 : jnp.array          = eqx.static_field()
    dt : jnp.array          = eqx.static_field()
    
    
    def __init__(self, pk, TAx, TAy, fc, dt_forcing, nl, AD_mode, call_args):
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
        
    @eqx.filter_jit
    def __call__(self, save_traj_at = None):
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
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
        
class jslab_kt(eqx.Module):
    # variables
    # U0 : np.float64
    # V0 : np.float64
    # control vector
    pk : jnp.array
    # parameters
    # TAx : jnp.array         = eqx.static_field()
    # TAy : jnp.array         = eqx.static_field()
    # fc : jnp.array          = eqx.static_field()
    # dTK : jnp.array         = eqx.static_field()
    # dt_forcing : jnp.array  = eqx.static_field()
    # nl : jnp.array          = eqx.static_field()
    # AD_mode : str           = eqx.static_field()
    # NdT : jnp.array         = eqx.static_field()
    # t0 : jnp.array          = eqx.static_field()
    # t1 : jnp.array          = eqx.static_field()
    # dt : jnp.array          = eqx.static_field()
    TAx : jnp.array         
    TAy : jnp.array         
    fc : jnp.array         
    dTK : jnp.array        
    dt_forcing : jnp.array  
    nl : jnp.array         
    AD_mode : str          
    NdT : jnp.array        
    t0 : jnp.array         
    t1 : jnp.array         
    dt : jnp.array         
    
    
    def __init__(self, pk, TAx, TAy, fc, dTK, dt_forcing, nl, AD_mode, call_args):
        t0,t1,dt = call_args
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        
        self.dTK = dTK
        self.NdT = int((t1-t0)//dTK) # jnp.array((t1-t0)//self.dTK,int)
        self.pk = pk #self.kt_ini( jnp.asarray(pk) )
        
        self.TAx = TAx
        self.TAy = TAy
        self.fc = fc
        
        self.dt_forcing = dt_forcing
        self.nl = nl
        self.AD_mode = AD_mode
        
        
        
    #@eqx.filter_jit
    #@partial(jax.jit, static_argnames=['save_traj_at'])
    @eqx.filter_jit
    def __call__(self, save_traj_at = None): #call_args, 

        # Auto-diff mode
        if self.AD_mode=='F':
            adjoint = diffrax.ForwardMode()
        else:
            adjoint = diffrax.RecursiveCheckpointAdjoint(checkpoints=10)

        solver = Euler()
        y0 = 0.0,0.0 # self.U0,self.V0
        t0, t1, dt = self.t0, self.t1, self.dt # call_args
        nsubsteps = self.dt_forcing // dt
        # control
        # K = jnp.exp( jnp.asarray(self.pk) )
        # K = self.kt_ini(K)
        K = jnp.exp( self.pk) 
        K = kt_1D_to_2D(K, NdT=self.NdT, nl=self.nl)
        forcing_time = jnp.arange(t0,t1,self.dt_forcing)
        M = pkt2Kt_matrix(gtime=forcing_time, NdT=self.NdT, dTK=self.dTK)
        Kt = jnp.dot(M,K)
        args = self.fc, Kt, self.TAx, self.TAy
        
        if save_traj_at is None:
            saveat = diffrax.SaveAt(steps=True)
        else:
            saveat = diffrax.SaveAt(ts=jnp.arange(t0,t1,save_traj_at)) # slower than above (no idea why)
        
        maxstep = int((t1-t0)//dt) +1 
        
        def vector_field(t, C, args):
            U,V = C
            fc, Kt, TAx, TAy = args
            
            # on the fly interpolation
            it = jnp.array(t//dt, int)
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
        
        return diffeqsolve(terms=ODETerm(vector_field), 
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
                           made_jump=False) # here this is needed to be able to forward AD
    
    """
    Note sur pourquoi j'ai enlevé t0, t1, dt du __call__:
    Si on veut faire des opérations sur K qui font un changement de base, il faut savoir le nombre de timestep, et donc il faut que nt=(t1-t0)//dt soit static !
    """
    
    
    
# K(t)
def kt_ini(pk, NdT):
    a_2D = jnp.repeat(pk, NdT)
    return kt_2D_to_1D(a_2D)

def kt_1D_to_2D(vector_kt_1D, NdT, nl):
    return vector_kt_1D.reshape((NdT,nl*2))

def kt_2D_to_1D(vector_kt):
    return vector_kt.flatten()


def pkt2Kt_matrix(NdT, dTK, gtime):
        """
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
        gptime = jnp.zeros(NdT) #self.ntm//step+1 )        

        nt = len(gtime)
        npt = len(gptime)
        M = jnp.zeros((nt,npt))
        S = jnp.zeros((nt))

        def __step_pkt2Kt_matrix(arg0, ip):
            gtime,gptime,S,M = arg0
            distt = (gtime-gptime[ip])
            tmp =  jnp.exp(-distt**2/dTK**2)
            S = lax.add( S, tmp )
            M = M.at[:,ip].set( M[:,ip] + tmp )
            arg0 = gtime,gptime,S,M
            return arg0,arg0

        # print('M.shape',M.shape)
        # print('S.shape',S.shape)
        # loop over each dT
        arg0 = gtime, gptime, S, M
        # This should work but there is a bug in jax
        # see : https://docs.kidger.site/equinox/faq/#how-to-use-non-array-modules-as-inputs-to-scancondwhile-etc
        # |
        # v
        #_, _, S, M = lax.fori_loop(0, npt, self.__step_pkt2Kt_matrix, arg0) 
        arg0,_ = lax.scan(lambda it,arg0: __step_pkt2Kt_matrix(it,arg0), arg0, xs=jnp.arange(0,npt))
        _, _, S, M = arg0
        M = (M.T / S.T).T
        return M
