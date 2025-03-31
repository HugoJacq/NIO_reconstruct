import os
import matplotlib.pyplot as plt
from functions_benchmark import get_data_from_file
import glob


result_local = 'benchmark_results.txt'
result_jackz = 'benchmark_results_jackz.txt'

L_results = [result_local,result_jackz]

# PLOT
dpi=200
path_save_png = './png_benchmark/'

os.system('mkdir -p '+path_save_png)

# getting data
nbparams = {}
tforward = {}
tcost = {}
tgrad = {}


jackz_files = glob.glob('*jackz.txt')
print(sorted(jackz_files))

# sort list by increasing delta T
L_delta_T = []
for file in jackz_files:
    name = file.split('_')
    t0 = float(name[2][2:])
    t1 = float(name[3][2:])
    delta_T = t1 - t0 # in days
    L_delta_T.append(delta_T)
L_delta_T, jackz_files = zip(*sorted(zip(L_delta_T, jackz_files)))
    
    
for file in jackz_files:
    namePC = 'jackz'
    nbparams[namePC], tforward[namePC], tcost[namePC], tgrad[namePC] = get_data_from_file()
            
# plotting

fig, ax = plt.subplots(1,3,figsize = (15,5),constrained_layout=True,dpi=dpi)

# ax[0].plot()
plt.show()
