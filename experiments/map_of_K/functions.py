

"""
Functions that tile the Croco simulation domain into a map.
"""
import jax.numpy as jnp
import numpy as np

from tools import nearest



def iter_bounds_mapper(R, dx, lon, lat):
    """
    
    - R : float, half of a side of a tile
    
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
    
    print(Nx, Ny, N, indx_start, indy_start)
    
    for kx in range(Nx):
        for ky in range(Ny):
            LAT_bounds = [lat[indy_start+ky*N],lat[indy_start+(ky+1)*N]]
            LON_bounds = [lon[indx_start+kx*N],lon[indx_start+(kx+1)*N]]
            point_loc = ( (LON_bounds[1]+LON_bounds[0])/2, (LAT_bounds[1]+LAT_bounds[0])/2 )
            yield point_loc, LON_bounds, LAT_bounds
            
            
            
    #return point_loc, LON_bounds, LAT_bounds

def save_pk(model):
    """
    """
    
    # write in a file
    