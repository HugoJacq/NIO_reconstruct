import os
import matplotlib.pyplot as plt
from functions_benchmark import get_data_from_file
import glob
from pprint import pprint
import numpy as np


# PLOT
dpi=200
path_save_png = './png_benchmark/'

os.system('mkdir -p '+path_save_png)




files = {'jackz':glob.glob('*jackz.txt')}

# sort list by increasing delta T
L_delta_T = []
for file in files['jackz']:
    name = file.split('_')
    t0 = float(name[2][2:])
    t1 = float(name[3][2:])
    delta_T = t1 - t0 # in days
    L_delta_T.append(delta_T)
L_delta_T, files['jackz'] = zip(*sorted(zip(L_delta_T, files['jackz'])))


dict_, _, _, _ = get_data_from_file(files['jackz'][0])

list_model = dict_.keys()



# getting data
nbparams = {}
tforward = {}
tcost = {}
tgrad = {}
for file in files['jackz']:
    namePC = 'jackz'
    nbparams[namePC], tforward[namePC], tcost[namePC], tgrad[namePC] = get_data_from_file(file)
      
data = {}
for machine in ['jackz']: # 'local'
    data[machine] = {}
    for nmodel in list_model:
        data[machine][nmodel] = {'forward':[],
                                 'cost':[],
                                 'grad':[],
                                 'grad/cost':[]}
        for file in files['jackz']:
            nbparams, tforward, tcost, tgrad = get_data_from_file(file)
            data[machine][nmodel]['forward'].append(tforward[nmodel])
            data[machine][nmodel]['cost'].append(tcost[nmodel])
            data[machine][nmodel]['grad'].append(tgrad[nmodel])
            data[machine][nmodel]['grad/cost'].append(tgrad[nmodel]/tcost[nmodel])
        data[machine][nmodel]['nbparams'] = nbparams[nmodel]

# plotting
lc = ['k','b']

for nmodel in data['jackz'].keys():
    for k,machine in enumerate(['jackz']): # 'local',
        
        fig, ax = plt.subplots(1,3,figsize = (15,5),constrained_layout=True,dpi=dpi)
        ax[0].plot(L_delta_T, data[machine][nmodel]['forward'], c=lc[k], label=f"{machine} ({data[machine][nmodel]['nbparams']})" )
        ax[0].set_title('forward')
        ax[0].set_ylabel('wall time (s)')
        ax[1].plot(L_delta_T, data[machine][nmodel]['cost'], c=lc[k], label=f"{machine} ({data[machine][nmodel]['nbparams']})" )
        ax[1].set_title('cost')
        #ax[2].plot(L_delta_T[0], data[machine][nmodel]['grad'][0], c=lc[k], ls='--', label=f"{machine} ({data[machine][nmodel]['nbparams']})" ) # dummy
        ax[2].plot(L_delta_T, data[machine][nmodel]['grad'], c=lc[k], label=f"{machine} ({data[machine][nmodel]['nbparams']})" ) 
        ax[2].set_title('grad')
        axbis = ax[2].twinx()
        axbis.set_ylabel('ratio grad/cost')
        axbis.plot(L_delta_T, np.asarray(data[machine][nmodel]['cost'])/np.asarray(data[machine][nmodel]['grad']), c=lc[k], ls='--')
        fig.suptitle(nmodel)
        for axe in ax:
            axe.set_xlabel('Days of runtime')    
            axe.legend(loc='upper left')
        fig.savefig(path_save_png+'benchmark_'+nmodel+'.png')
plt.show()
