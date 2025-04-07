"""
This module gather functions related to basis changes of control parameters

used in the models definitions ./models/
"""

import jax.numpy as jnp
from functools import partial
import jax
from jax import lax


# K(t)
# all of this could be moved to a different file ...
def kt_ini(pk, NdT):
    a_2D = jnp.repeat(pk, NdT)
    return kt_2D_to_1D(a_2D)

def kt_1D_to_2D(vector_kt_1D, NdT, npk):
    return vector_kt_1D.reshape((NdT,npk))

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
            coeff = 1 # if =2, exp do not recover
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
