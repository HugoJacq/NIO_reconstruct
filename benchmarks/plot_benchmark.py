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



L_machine = ['local','jackz']
files = {'jackz':glob.glob('*jackz.txt'),
         'local':glob.glob('*0.txt')}

# sort list by increasing delta T
L_delta_T_old = []
for file in files['jackz']:
    name = file.split('_')
    t0 = float(name[2][2:])
    t1 = float(name[3][2:])
    delta_T = t1 - t0 # in days
    L_delta_T_old.append(delta_T) 
    
L_delta_T, files['jackz'] = zip(*sorted(zip(L_delta_T_old, files['jackz'])))


L_delta_T_old = []
for file in files['local']:
    name = file.split('_')
    t0 = float(name[2][2:])
    t1 = float(name[3][2:-4])
    delta_T = t1 - t0 # in days
    L_delta_T_old.append(delta_T) 

L_delta_T, files['local'] = zip(*sorted(zip(L_delta_T_old, files['local'])))
dict_, _, _, _ = get_data_from_file(files['jackz'][0])

list_model = dict_.keys()

L_delta_T = np.asarray(L_delta_T)

# getting data
nbparams = {}
tforward = {}
tcost = {}
tgrad = {}
# for file in files['jackz']:
#     namePC = 'jackz'
#     nbparams[namePC], tforward[namePC], tcost[namePC], tgrad[namePC] = get_data_from_file(file)
      
data = {}
for machine in L_machine: # 'local'
    data[machine] = {}
    for nmodel in list_model:
        data[machine][nmodel] = {'forward':[],
                                 'cost':[],
                                 'grad':[],
                                 'grad/cost':[],
                                 'nbparams':[]}
        for file in files[machine]:
            nbparams, tforward, tcost, tgrad = get_data_from_file(file)
            data[machine][nmodel]['forward'].append(tforward[nmodel])
            data[machine][nmodel]['cost'].append(tcost[nmodel])
            data[machine][nmodel]['grad'].append(tgrad[nmodel])
            data[machine][nmodel]['grad/cost'].append(tgrad[nmodel]/tcost[nmodel])
            data[machine][nmodel]['nbparams'].append(nbparams[nmodel])
            
# plotting
lc = ['k','b']
for nmodel in data['jackz'].keys():
    fig, ax = plt.subplots(1,4,figsize = (15,5),constrained_layout=True,dpi=dpi)
    
    axbis = ax[2].twinx()
    axbis.set_ylabel('ratio grad/cost')
    
    for k,machine in enumerate(L_machine): # 'local',
        coeff = 1000
        time_forward = np.asarray(data[machine][nmodel]['forward']) / L_delta_T * coeff
        time_cost = np.asarray(data[machine][nmodel]['cost']) / L_delta_T * coeff
        time_grad = np.asarray(data[machine][nmodel]['grad']) / L_delta_T * coeff
        ratio = np.asarray(data[machine][nmodel]['grad/cost']) 
        Nparam = data[machine][nmodel]['nbparams']
        
        ax[0].plot(L_delta_T, time_forward, c=lc[k], label=f"{machine}" )
        ax[0].set_title('forward')
        ax[0].set_ylabel('wall time per day of run (ms/days)')
        ax[1].plot(L_delta_T, time_cost, c=lc[k], label=f"{machine}" )
        ax[1].set_title('cost')
        #ax[2].plot(L_delta_T[0], data[machine][nmodel]['grad'][0], c=lc[k], ls='--', label=f"{machine} ({data[machine][nmodel]['nbparams']})" ) # dummy
        ax[2].plot(L_delta_T, time_grad, c=lc[k], label=f"{machine}" ) 
        ax[2].set_title('grad')
        axbis.plot(L_delta_T, ratio, c=lc[k], ls='--')
        ax[3].plot(L_delta_T, Nparam, c=lc[k], label=f"{machine}" )
        ax[3].set_title('nb parameters')
        # ax[0].plot(Nparam, time_forward, c=lc[k], label=f"{machine}" )
        # ax[0].set_title('forward')
        # ax[0].set_ylabel('wall time per day of run (s/days)')
        # ax[1].plot(Nparam, time_cost, c=lc[k], label=f"{machine}" )
        # ax[1].set_title('cost')
        # #ax[2].plot(L_delta_T[0], data[machine][nmodel]['grad'][0], c=lc[k], ls='--', label=f"{machine} ({data[machine][nmodel]['nbparams']})" ) # dummy
        # ax[2].plot(Nparam, time_grad, c=lc[k], label=f"{machine}" ) 
        # ax[2].set_title('grad')
        # axbis.plot(Nparam, ratio, c=lc[k], ls='--')
        # ax[3].plot(Nparam, Nparam, c=lc[k], label=f"{machine}" )
        # ax[3].set_title('nb parameters')
        
        fig.suptitle(nmodel)
        for axe in ax:
            axe.set_xlabel('Days of runtime')    
            axe.legend(loc='upper left')
        fig.savefig(path_save_png+'benchmark_'+nmodel+'.png')
plt.show()
