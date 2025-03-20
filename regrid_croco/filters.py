import numpy as np
from joblib import Parallel, delayed
import scipy as sp
import sys
sys.path.insert(0, '../src')
from tools import ParallelTqdm
import tqdm

def my2dfilter(s,sigmax,sigmay, ns=2):
    """
    Spatial 2D filter, using gaussian kernel.
    This is used to get large scale trend of SSH

    INPUT:
        - s : signal to smooth
        - sigmax : for gaussian kernel, std in x direction
        - sigmay : for gaussian kernel, std in y direction
        - ns : width of the gaussian filter, in number of std
    OUTPUT:
        - smoothed signal 'sf', same dimensions as 's'
    """

    x, y = np.meshgrid(np.arange(-int(ns*sigmax), int(ns*sigmax)+1), np.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
    cf=np.exp(-(x**2/sigmax**2+y**2/sigmay**2))
    m=~np.isnan(s)*1.
    s = np.where(np.isnan(s),0,s)
    # s_sum = (sp.signal.convolve2d(s, cf, mode='same'))
    # w_sum = (sp.signal.convolve2d(m, cf, mode='same'))
    s_sum = sp.signal.oaconvolve(s, cf, mode='same')
    w_sum = sp.signal.oaconvolve(m, cf, mode='same')
    sf = s*0.
    sf[w_sum!=0] = s_sum[w_sum!=0] / w_sum[w_sum!=0]
    return sf

def my2dfilter_over_time(s,sigmax,sigmay, nt, N_CPU=1, ns=2, show_progress=False):
    """
    'my2dfilter' but over each time step and in parallel with Joblib
    
    # to do: wrap this with xarray ufunc to keep attrs/dims/coords
    """  
    if N_CPU==1:
        list_results = []
        for it in range(nt):
            print(it,'/',nt)
            list_results.append( my2dfilter(s[it,:,:], sigmax, sigmay) )
    else:
        
        if show_progress:
            results = ParallelTqdm(n_jobs=N_CPU)([delayed(my2dfilter)(s[it,:,:], sigmax, sigmay) for it in range(nt)])
        else:
            results = Parallel(n_jobs=N_CPU)(delayed(my2dfilter)(s[it,:,:], sigmax, sigmay) for it in range(nt))
    return np.array(results)

def mytimefilter(Hf0):
    """
    This is a time smoothing operator
    
    INPUT:
        - Hf0: array 3D (time,y,x)
    OUTPUT:
        - Hf: smoothed field, same dimensions as Hf0
    """
    
    _, ny, nx = np.shape(Hf0) # shape of array
    time_conv = np.arange(-1*86400,1*86400+3600,3600) # 1D array, time boundaries for kernel (in sec)
    Hf = Hf0*0. # initialization
    
    # time kernel for convolution
    taul = 2*86400
    gl = np.exp(-taul**-2 * time_conv**2) 
    gl = (gl / np.sum(np.abs(gl)))
    # doing the convolution in time
    for ix in range(nx):
        for iy in range(ny):
            print(ix,iy)
            #Hf[:,iy,ix] = np.convolve(Hf0[:,iy,ix],gl,'same')
            #Hf[:,iy,ix] = sp.signal.oaconvolve(Hf0[:,iy,ix],gl, mode='same')
            Hf[:,iy,ix] = mytimefilter1D(Hf0[:,iy,ix])
    return Hf

def mytimefilter1D(Hf0):
    """
    Smoothing filter on 1D array
    input is dt=1h
    """
    dt = 3600
    time_conv = np.arange(-1*86400,1*86400+dt,dt) # 1D array, time boundaries for kernel (in sec)
    # time kernel for convolution
    taul = 2*86400
    gl = np.exp(-taul**-2 * time_conv**2) 
    gl = (gl / np.sum(np.abs(gl)))
    #Hf = np.convolve(Hf0[:],gl,'same')
    Hf = sp.signal.oaconvolve(Hf0[:],gl, mode='same')
    return Hf

def mytimefilter_over_spatialXY(Hf0, N_CPU=1, show_progress=False):
    """
    'mytimefilter1D' but over XY
    
    Hf0 of the form (t,y,x), 
    dt=1h, if not 1h, you need to change in mytimefilter1D
    
    # to do: wrap this with xarray ufunc to keep attrs/dims/coords
    """
    Hf = Hf0*0. # initialization
    nt, ny, nx = np.shape(Hf0) # shape of array
    
    
    list_index = [(ix,iy) for ix in range(nx) for iy in range(ny)]
    if N_CPU<=1:
        if show_progress:
            for ind in tqdm.tqdm(list_index):
                Hf[:,ind[1],ind[0]] = mytimefilter1D(Hf0[:,ind[1],ind[0]])
        else:
            for ix in range(nx):
                for iy in range(ny):
                    Hf[:,iy,ix] = mytimefilter1D(Hf0[:,iy,ix]) 
        
    else:
        #raise Exception('// mytimefilter_over_spatialXY is bugged, exiting')
        # To expand a bit on this:
        #   when i use the parallel version (see undernearth) of the double for loop 
        #   I get strange lines with unphysical values like 10e10 for SSH.
        #   
        # In the end it is not very important as the full North Atlantic Croco sim
        # is like 1400 * 1900, it takes around 20min to apply this filter !
    
        if show_progress:
            #results = ParallelTqdm(n_jobs=N_CPU)([delayed(mytimefilter1D)(Hf0[:,ind[1],ind[0]]) for ind in list_index])
            results = []
            # for ix in tqdm.tqdm(range(nx)):
            #     results.append( Parallel(n_jobs=N_CPU)([delayed(mytimefilter1D)(Hf0[:,iy,ix]) for iy in range(ny)]) )
            for iy in tqdm.tqdm(range(ny)):
                results.append( Parallel(n_jobs=N_CPU)([delayed(mytimefilter1D)(Hf0[:,iy,ix]) for ix in range(nx)]) )
            
        else:
            results = Parallel(n_jobs=N_CPU)(delayed(mytimefilter1D)(Hf0[:,ind[1],ind[0]]) for ind in list_index)
        
        
        Hf = np.asarray(results, dtype=float).reshape((nt,ny,nx))
        
    return Hf
#
# ATTEMPT TO USE XARRAY AND BUILD A XR CONVOLUTION
#


#convolve2D_vectorized = np.vectorize( sp.signal.convolve2d,
 #                       signature='(k,m,n),(i,j)->(m,n)')




# def convolve2D_ufunc(array,cf,mode='same'):
#     shape_cf = cf.shape
#     cf2 = np.zeros((shape_cf[0],shape_cf[1],1))
#     print('SIZE BEFORE CONVOLVE',array.shape,cf2.shape)
#     ufunc = lambda x: sp.signal.convolve(x,cf2,mode)
#     return xr.apply_ufunc(
# 		ufunc,	# func to use
# 		array,				# input of func, usually numpy array
# 		dask="parallelized", # parallelized, to allow // computing if array is already chuncked
# 		input_core_dims=[['time']],	# axis of work of func
# 		output_core_dims=[['time']],
# 		#kwargs={'sigmax':sigmax,'sigmay':sigmay,'ns':ns}, # kwargs of func
# 		output_dtypes=[array.dtype],				# this is passed to Dask
# 		dask_gufunc_kwargs={'allow_rechunk':False},	# this is passed to Dask, if core dim is chuncked
#         vectorize=False,
# 	        ).transpose("time",... )
    
    

# def my2dfilter_xr(array,sigmax,sigmay,ns=2):
#     print('entering my2dfilter_xr')
#     x, y = np.meshgrid(np.arange(-int(ns*sigmax), int(ns*sigmax)+1), np.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
#     #x, y = meshgrid(np.arange(-int(ns*sigmax), int(ns*sigmax)+1), np.arange(-int(ns*sigmay), int(ns*sigmay)+1), indexing='ij')
#     cf=np.exp(-(x**2/sigmax**2+y**2/sigmay**2))
    
    
#     #m= ~np.isnan(s)*1.
#     m = xr.where( xr.ufuncs.isnan(array), 0., 1.0)
    
#     #s = np.where(np.isnan(s),0,s)
#     s = xr.where( xr.ufuncs.isnan(array), 0., array)

#     print(s)
#     print(m)

#     print('begin of convolve')
#     print('s.shape,cf.shape,cf.shape',s.shape,cf.shape,cf.shape)
#     s_sum = convolve2D_ufunc(s, cf)
#     w_sum = convolve2D_ufunc(m, cf)
#     print('s_sum.shape,w_sum.shape',s_sum.shape,w_sum.shape)
#     print('end of convolve')
#     #sf = s*0.
#     #sf[w_sum!=0] = s_sum[w_sum!=0] / w_sum[w_sum!=0]
#     sf = xr.where( w_sum!=0, s_sum/w_sum, 0)
    
#     print('s',s)
#     print('s_sum',s_sum)
#     print('w_sum',w_sum)
#     print('sf',sf)

#     return sf




