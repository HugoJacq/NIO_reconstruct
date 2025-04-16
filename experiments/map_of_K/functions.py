

"""
Functions that tile the Croco simulation domain into a map.
"""
import jax.numpy as jnp





def iter_bounds_mapper(R):
    """
    
    - R : float, half of a side of a tile
    
    """
    point_loc, LON_bounds, LAT_bounds = 1,1,1
    return point_loc, LON_bounds, LAT_bounds

def save_pk(pk, model_name):
    """
    """
    
    # write in a file