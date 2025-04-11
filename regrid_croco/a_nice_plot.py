import xarray as xr
import matplotlib.pyplot as plt

namefile = 'croco_1h_inst_surf_2005-07-01-2005-07-31.nc'
file_path_HR = '/ODYSEA2/Lionel_coupled_run/' + namefile
file_path_LR = '../data_regrid/' + namefile


dsHR = xr.open_dataset(file_path_HR)
dsLR = xr.open_dataset(file_path_LR)

print(dsHR)

print(dsLR)

fig, ax = plt.subplots(1,2,figsize = (10,10),constrained_layout=True,dpi=200)

plt.show()