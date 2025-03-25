import matplotlib.pyplot as plt

from regridder import mf_regridder
from plot_functions import *

path_in_local = '/home/jacqhugo/Datlas_2025/DATA_Crocco/'
path_in_hpc = '/data2/nobackup/clement/Data/Lionel_coupled_run/'
path_in_hpc = '/ODYSEA2/Lionel_coupled_run/'
#L_filename = ['croco_1h_inst_surf_2006-02-01-2006-02-28']
L_filename = ['croco_1h_inst_surf_2005-01-01-2005-01-31.nc',
              'croco_1h_inst_surf_2005-02-01-2005-02-28.nc',
              'croco_1h_inst_surf_2005-03-01-2005-03-31.nc',
              'croco_1h_inst_surf_2005-04-01-2005-04-30.nc',
              'croco_1h_inst_surf_2005-05-01-2005-05-31.nc',
              'croco_1h_inst_surf_2005-06-01-2005-06-30.nc',
              'croco_1h_inst_surf_2005-07-01-2005-07-31.nc',
              'croco_1h_inst_surf_2005-08-01-2005-08-31.nc',
              'croco_1h_inst_surf_2005-09-01-2005-09-30.nc',
              'croco_1h_inst_surf_2005-10-01-2005-10-31.nc',
              'croco_1h_inst_surf_2005-11-01-2005-11-30.nc',
              'croco_1h_inst_surf_2005-12-01-2005-12-31.nc']

path_save = '../data_regrid/'
ON_HPC = True              
N_CPU = 1                   # for // of spatial filter. Bug in spatial filter if N_CPU>1 
                            # but thats ok because this filter is applied on the new
                            # regridded file which is way smaller than the input file


new_dx = 0.1                # °, new resolution
method = 'conservative'     # conservative (long but more accurate) or bilinear (fast)

SHOW_DIFF           = False        # show a map with before/after
SHOW_SPACE_DIFF     = False        # show diff along a spatial dimension
SHOW_TIME_DIFF      = False        # show diff along time
SHOW_BILI_VS_CONS   = False        # if CHECK_RESULTS: show diff bilinear - conservative

if __name__ == "__main__": 
    if ON_HPC:
        path_in = path_in_hpc
    else:
        path_in = path_in_local
        
    # producing the files
    print('* Regridding process')
    mf_regridder(path_in, L_filename, method, new_dx, path_save, N_CPU=N_CPU)

    if SHOW_DIFF:
        print('* Looking at new fields')
        show_diff(path_in, L_filename, new_dx, method, path_save)
     
    if SHOW_SPACE_DIFF:
        print('* Looking at =/= along spatial dimension')
        show_diff_along_space(path_in, L_filename, new_dx, method, path_save)
     
    if SHOW_TIME_DIFF:
        print('* Looking at =/= along time dimension')
        show_diff_along_time(path_in, L_filename, new_dx, method, path_save)  
        
    if SHOW_BILI_VS_CONS:
        print('* Looking at =/= between conservative and bilinear methods')
        show_bili_vs_cons(L_filename, new_dx, path_save)
    
    plt.show()