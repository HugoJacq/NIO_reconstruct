"""
Here function that runs the tests
"""
# import sys
# sys.path.insert(0, '../src')
import time as clock
import matplotlib.pyplot as plt
import numpy as np

import tools

def run_forward_cost_grad(mymodel, var_dfx, call_args):
    """
    """
    dynamic_model, static_model = var_dfx.my_partition(mymodel)
    
    time1 = clock.time()
    C = mymodel(call_args)
    print(' time, forward model (with compile)',clock.time()-time1)
    
    time2 = clock.time()
    C = mymodel(call_args)
    print(' time, forward model (no compile)',clock.time()-time2)
    
    time3 = clock.time()
    J = var_dfx.cost(dynamic_model, static_model, call_args)
    print(' time, cost (with compile)',clock.time()-time3)

    time4 = clock.time()
    J = var_dfx.cost(dynamic_model, static_model, call_args)
    print(' time, cost (no compile)',clock.time()-time4)

    time5 = clock.time()
    J, dJ = var_dfx.grad_cost(dynamic_model, static_model, call_args)
    print(' time, gradcost (with compile)',clock.time()-time5)

    time6 = clock.time()
    J, dJ = var_dfx.grad_cost(dynamic_model, static_model, call_args)
    print(' time, gradcost (no compile)',clock.time()-time6)
    
def plot_traj(mymodel, var_dfx, call_args, forcing1D, observations1D, name_save, path_save_png, dpi):
    Ua = mymodel(call_args, save_traj_at=mymodel.dt_forcing).ys[0]
    U = forcing1D.data.U.values
    Uo, _ = observations1D.get_obs()
    RMSE = tools.score_RMSE(Ua, U) 
    dynamic_model, static_model = var_dfx.my_partition(mymodel)
    final_cost = var_dfx.cost(dynamic_model, static_model, call_args)
    print('RMSE is',RMSE)
    # PLOT trajectory
    fig, ax = plt.subplots(1,1,figsize = (10,3),constrained_layout=True,dpi=dpi)
    ax.plot(forcing1D.time/86400, U, c='k', lw=2, label='Croco')
    ax.plot(forcing1D.time/86400, Ua, c='g', label='slab')
    ax.scatter(observations1D.time_obs/86400,Uo, c='r', label='obs')
    ax.set_ylim([-0.3,0.4])
    ax.set_title('RMSE='+str(np.round(RMSE,4))+' cost='+str(np.round(final_cost,4)))
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Ageo zonal current (m/s)')
    ax.legend(loc=1)
    fig.savefig(path_save_png+name_save+'.png')