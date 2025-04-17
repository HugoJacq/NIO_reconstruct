

"""
Functions that tile the Croco simulation domain into a map.
"""
import jax.numpy as jnp
import numpy as np
import os
from Listes_models import L_slabs

import models.classic_slab as classic_slab
import models.unsteak as unsteak
from basis import kt_ini, kt_1D_to_2D

def model_instanciation(model_name, forcing, args_model, call_args, extra_args):
    """
    Wrapper of every model instanciation.
    
    
    TBD: doc of this, with typing
    """
    t0, t1, dt = call_args
    dTK = args_model['dTK']
    Nl = args_model['Nl']
    
    # MODELS FROM MODULE classic_slab        
    if model_name=='jslab_kt_2D':
        pk = jnp.asarray([-11., -10.])   
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = classic_slab.jslab_kt_2D(pk, dTK, forcing, call_args, extra_args)
        
    elif model_name=='jslab_kt_2D_adv':
        pk = jnp.asarray([-11., -10.])   
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = classic_slab.jslab_kt_2D_adv(pk, dTK, forcing, call_args, extra_args)
        
    elif model_name=='jslab_kt_2D_adv_Ut':
        pk = jnp.asarray([-11., -10.])   
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = classic_slab.jslab_kt_2D_adv_Ut(pk, dTK, forcing, call_args, extra_args)
        
    
    
    # MODELS FROM MODULE unsteak        
    elif model_name=='junsteak_kt_2D':
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-10.,-10., -9., -9.]) 
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = unsteak.junsteak_kt_2D(pk, dTK, forcing, Nl, call_args, extra_args)   
    
    elif model_name=='junsteak_kt_2D_adv':
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-10.,-10., -9., -9.]) 
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = unsteak.junsteak_kt_2D_adv(pk, dTK, forcing, Nl, call_args, extra_args)   
        
    else:
        raise Exception(f'You want to test the mode {model_name} but it is not recognized')

    
    return model


def number_of_tile(R, lon, lat):
    Side = 2*R 
    Lx = np.max(lon) - np.min(lon)
    Ly = np.max(lat) - np.min(lat)
    Nx = int(Lx//Side)
    Ny = int(Ly//Side)
    return Nx, Ny
    
def iter_bounds_mapper(R, dx, lon, lat):
    """
    This function is an iterator: each time it is called it will give dimensions of a new box of side 2*R 
        that fit inside the rectangle formed by lat,lon.
        
        The tuple returned is:
            (
            (lon of center, lat of center),
            [min lon of rectangle, max lon of rectangle],
            [min lat of rectangle, max lat of rectangle],
            )
    INPUTS:
        - R     : float, half of a side of a tile
        - dx    : horizontal resolution
        - lon   : 1D array with longitudes
        - lat   : 1D array with latitudes
    """
    #point_loc, LON_bounds, LAT_bounds = 1,1,1
    
    # first lets find the index of lower left corner
    #   of a rectangle compose of square tiles
    #   of side R/2.
    
    Side = 2*R 
    N = int(Side//dx) + 1
    Lx = np.max(lon) - np.min(lon)
    Ly = np.max(lat) - np.min(lat)
    Nx, reste_x = int(Lx//Side), Lx%Side
    Ny, reste_y = int(Ly//Side), Ly%Side
    indx_start = int(reste_x/2//dx)  # nearest(lon, lon[0+])
    indy_start = int(reste_y/2//dx) #nearest(lat, lat[])
    
    for kx in range(Nx):
        for ky in range(Ny):
            LAT_bounds = [lat[indy_start+ky*N],lat[indy_start+(ky+1)*N]]
            LON_bounds = [lon[indx_start+kx*N],lon[indx_start+(kx+1)*N]]
            point_loc = ( (LON_bounds[1]+LON_bounds[0])/2, (LAT_bounds[1]+LAT_bounds[0])/2 )
            yield point_loc, LON_bounds, LAT_bounds

def save_pk_tile(model, 
            tile_infos,
            path_save):
    """
    This function saves the control parameters model.pk of tile 'k' at 'point_loc' position 
        in the folder 'path_save'+model_name. 
        Also, a text file is saved to save 'tile_infos'
    
    INPUTS:
        - model     : eqx.Module
        - tile_infos : index of tile, coords of center, lon bounds, lat bounds
        - path_save : where to save .npy and link file
        
    OUPUTS:
        - .npy file for tile 'k' with control parameter
        - a txt file that link 'k' with 'infos_loc'
    """
    model_name = type(model).__name__
    k, point_loc, LON_bounds, LAT_bounds = tile_infos
    
    os.system('mkdir -p '+path_save+model_name)
    
    # write in a file
    if model_name in L_slabs:
        npk=2
    else:
        npk=2*model.nl
    mypk = kt_1D_to_2D(model.pk, model.NdT, npk)
    np.save(path_save+model_name+f'/{k}.npy', mypk)
    
    # write 1 file with link indice k with lat/lon bounds
    file_name_link = 'link'
    if k==0:
        os.system('mv -f '+path_save+model_name+'/'+file_name_link+'.txt '+path_save+model_name+'/'+file_name_link+'_previous.txt') # save previous if exist
        mode = 'w'
    else:
        mode = 'a'
    with open(path_save+model_name+'/'+file_name_link+".txt", mode) as f:
        # structure is:
        # indice of tile, lon center, lat center, min lon, max lon, min lat, max lat
        f.write(f'{k},{point_loc[0]},{point_loc[1]},{LON_bounds[0]},{LON_bounds[1]},{LAT_bounds[0]},{LAT_bounds[1]}\n')
        
        
def compute_and_save_pk(model, var, mini_args, tile_infos, path_save):
    """
    Nice wrapper of minimization process and saving_pk_tile
    """
    maxiter = mini_args['maxiter']
    


    # test if pk is here

    # minimize to find new pk
    mymodel, _ = var.scipy_lbfgs_wrapper(model, maxiter, verbose=False)   
    # mymodel = model
    
    # save pk
    save_pk_tile(mymodel, tile_infos, path_save)
            