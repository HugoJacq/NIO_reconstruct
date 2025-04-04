"""
Here function that runs the tests
"""
# import sys
# sys.path.insert(0, '../src')
import time as clock
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, '../src')

import tools
import models.classic_slab as classic_slab
from basis import kt_ini, kt_1D_to_2D, pkt2Kt_matrix
from constants import *
from inv import my_partition

def run_forward_cost_grad(mymodel, var_dfx):
    """
    """
    dynamic_model, static_model = my_partition(mymodel)
    
    time1 = clock.time()
    _ = mymodel() # call_args
    print(' time, forward model (with compile)',clock.time()-time1)

    time2 = clock.time()
    _ = mymodel()
    print(' time, forward model (no compile)',clock.time()-time2)

    time3 = clock.time()
    _ = var_dfx.cost(dynamic_model, static_model)
    print(' time, cost (with compile)',clock.time()-time3)

    time4 = clock.time()
    _ = var_dfx.cost(dynamic_model, static_model)
    print(' time, cost (no compile)',clock.time()-time4)

    time5 = clock.time()
    _, _ = var_dfx.grad_cost(dynamic_model, static_model)
    print(' time, gradcost (with compile)',clock.time()-time5)

    time6 = clock.time()
    _, _ = var_dfx.grad_cost(dynamic_model, static_model)
    print(' time, gradcost (no compile)',clock.time()-time6)
    
def plot_traj_1D(mymodel, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi):
    Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
    U = forcing1D.data.U.values
    V = forcing1D.data.V.values
    t0 = mymodel.t0
    Uo, _ = observations1D.get_obs()
    RMSE = tools.score_RMSE(Ua, U) 
    dynamic_model, static_model = my_partition(mymodel)
    final_cost = var_dfx.cost(dynamic_model, static_model)
        
    SHOW_BASE_TRANSFORM = False    
        
    print('RMSE is',RMSE)
    # PLOT trajectory
    fig, ax = plt.subplots(3,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
    if var_dfx.filter_at_fc:
        ax[0].plot((t0 + forcing1D.time)/oneday, U, c='k', lw=2, label='Croco', alpha=0.3)
        ax[0].plot((t0 + forcing1D.time)/oneday, Ua, c='g', label='slab', alpha = 0.3)
        (Ut_nio,Vt_nio) = tools.my_fc_filter(mymodel.dt_forcing,U+1j*V, mymodel.fc)
        ax[0].plot((t0 + forcing1D.time)/oneday, Ut_nio, c='k', lw=2, label='Croco at fc', alpha=1)
        (Unio,Vnio) = tools.my_fc_filter(mymodel.dt_forcing, Ua+1j*Va, mymodel.fc)
        ax[0].plot((t0 + forcing1D.time)/oneday, Unio, c='b', label='slab at fc')
    else:
        ax[0].plot((t0 + forcing1D.time)/oneday, U, c='k', lw=2, label='Croco', alpha=1)
        ax[0].plot((t0 + forcing1D.time)/oneday, Ua, c='g', label='slab')
    ax[0].scatter((t0 + observations1D.time_obs)/oneday,Uo, c='r', label='obs', marker='x')
    ax[0].set_ylim([-0.6,0.6])
    #ax.set_xlim([15,25])
    #ax.set_title('RMSE='+str(np.round(RMSE,4))+' cost='+str(np.round(final_cost,4)))
    
    ax[0].set_ylabel('Ageo zonal current (m/s)')
    ax[0].legend(loc=1)
    ax[0].grid()
    # plot forcing
    ax[1].plot((t0 + forcing1D.time)/oneday, forcing1D.TAx, c='b', lw=2, label=r'$\tau_x$', alpha=1)
    ax[1].plot((t0 + forcing1D.time)/oneday, forcing1D.TAy, c='orange', lw=2, label=r'$\tau_y$', alpha=1)
    ax[1].set_ylabel('surface stress (N/m2)')
    ax[1].legend(loc=1)
    ax[1].grid()
    # plot MLD
    if hasattr(mymodel, 'dTK'):
        NdT = len(np.arange(mymodel.t0, mymodel.t1, mymodel.dTK))
        M = classic_slab.pkt2Kt_matrix(NdT, mymodel.dTK, mymodel.t0, mymodel.t1, mymodel.dt_forcing, base=mymodel.k_base)
        kt2D = classic_slab.kt_1D_to_2D(mymodel.pk, NdT, npk=len(mymodel.pk)//NdT*mymodel.nl)
        new_kt = np.dot(M,kt2D)
        myMLD = 1/np.exp(new_kt[:,0])/rho
        myR = np.transpose(np.exp(new_kt[:,1:])) * myMLD
        if SHOW_BASE_TRANSFORM and mymodel.k_base !='id':
            M2 = classic_slab.pkt2Kt_matrix(NdT, mymodel.dTK, mymodel.t0, mymodel.t1, mymodel.dt_forcing, base='id')
            new_kt2 = np.dot(M2,kt2D)
            myMLD_cst = 1/np.exp(new_kt2[:,0])/rho
            ax[2].plot((t0 + forcing1D.time)/oneday, myMLD_cst, label='no base transform')
        
        if SHOW_BASE_TRANSFORM:
            fig2, ax2 = plt.subplots(1,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
            for k in range(M.shape[-1]):
                ax2.plot((t0 + forcing1D.time)/oneday, M[:,k])
                #ax2.plot(M2[:,k], ls='--')
    else:
        myMLD = 1/np.exp(mymodel.pk[0])/rho * np.ones(len(forcing1D.time))
        myR = np.stack( [np.exp(mymodel.pk[1:]) for _ in range(len(forcing1D.time))], axis=1) * myMLD
    ax[2].plot((t0 + forcing1D.time)/oneday, myMLD, label='estimated')
    #ax[2].plot((t0 + forcing1D.time)/oneday, forcing1D.MLD, c='k', label='true')
    ax[2].set_ylabel('MLD (m)')
    ax[2].legend(loc=1)
    ax[2].grid()
    if False:
        ax2bis = ax[2].twinx()
        ax2bis.set_ylabel('r')
        lls = ['--','-.']
        for kr in range(len(myR)):
            ax2bis.plot((t0 + forcing1D.time)/oneday, myR[kr],ls=lls[kr])
    ax[2].set_xlabel('Time (days)')
    fig.savefig(path_save_png+name_save+'_t'+str(int(mymodel.t0//oneday))+'_'+str(int(mymodel.t1//oneday))+'.png')
    
def plot_traj_2D(mymodel, var_dfx, forcing2D, observations2D, name_save, point_loc, LON_bounds, LAT_bounds, path_save_png, dpi):
    Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
    U = forcing2D.data.U
    V = forcing2D.data.V
    Ug = forcing2D.data.Ug
    Vg = forcing2D.data.Vg
    TAx, TAy = forcing2D.data.oceTAUX, forcing2D.data.oceTAUY
    
    t0, t1 = mymodel.t0, mymodel.t1
    
    indx = tools.nearest(forcing2D.data.lon.values, point_loc[0])
    indy = tools.nearest(forcing2D.data.lat.values, point_loc[1])
    if type(mymodel).__name__ in ['jslab_kt_2D_adv_Ut']:
        U1D = U.values + Ug.values
        V1D = V.values + Vg.values
        Uo, _ = observations2D.get_obs(is_utotal=True)
    else:
        U1D = U.values
        V1D = V.values
        Uo, _ = observations2D.get_obs(is_utotal=False)
    U1D = U1D[:,indx,indy]
    V1D = V1D[:,indx,indy]
    TAx1D = TAx.values[:,indx,indy]
    TAy1D = TAy.values[:,indx,indy]
    Ua1D = Ua[:,indx,indy]
    Va1D = Va[:,indx,indy]
    Uo1D = Uo[:,indx,indy]
    
    RMSE = tools.score_RMSE(Ua1D, U1D) 

    print('RMSE at point_loc '+str(point_loc)+' is',RMSE)
    # PLOT trajectory at point_loc
    if True:
        fig, ax = plt.subplots(2,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
        Nxy = U[0,:,:].size
        # U_xy_flat = np.reshape(U.values,(U.shape[0],Nxy))
        # for k in range(Nxy):
        #     ax[0].plot((t0 + forcing2D.time)/oneday, U_xy_flat[:,k], c='k', alpha=0.1)
            
        if var_dfx.filter_at_fc:
            ax[0].plot((t0 + forcing2D.time)/oneday, U1D, c='k', lw=2, label='Croco', alpha=0.3)
            ax[0].plot((t0 + forcing2D.time)/oneday, Ua1D, c='g', label='slab', alpha = 0.3)
            (Ut_nio,Vt_nio) = tools.my_fc_filter(mymodel.dt_forcing,U1D+1j*V1D, mymodel.fc)
            ax[0].plot((t0 + forcing2D.time)/oneday, Ut_nio, c='k', lw=2, label='Croco at fc', alpha=1)
            (Unio,Vnio) = tools.my_fc_filter(mymodel.dt_forcing, Ua1D+1j*Va1D, mymodel.fc)
            ax[0].plot((t0 + forcing2D.time)/oneday, Unio, c='b', label='slab at fc')
        else:
            ax[0].plot((t0 + forcing2D.time)/oneday, U1D, c='k', lw=2, label='Croco at loc', alpha=1)
            ax[0].plot((t0 + forcing2D.time)/oneday, Ua1D, c='g', label='slab at loc')
        ax[0].scatter((t0 +observations2D.time_obs)/oneday, Uo1D, c='r', label='obs at loc', marker='x')
        ax[0].set_ylim([-0.6,0.6])
        ax[0].set_ylabel('Ageo zonal current (m/s)')
        ax[0].legend(loc=1)
        ax[0].grid()
        
        ax[1].plot((t0 + forcing2D.time)/oneday, TAx1D, c='b', lw=2, label=r'$\tau_x$', alpha=1)
        ax[1].plot((t0 + forcing2D.time)/oneday, TAy1D, c='orange', lw=2, label=r'$\tau_y$', alpha=1)
        ax[1].set_ylabel('surface stress (N/m2)')
        ax[1].legend(loc=1)
        ax[1].grid()
        
        # # budget terms
        # if type(mymodel).__name__ in ['jslab_kt_2D_adv','jslab_kt_2D']:
        #     if hasattr(mymodel, 'dTK'):
        #         NdT = len(np.arange(mymodel.t0, mymodel.t1, mymodel.dTK))
        #         M = pkt2Kt_matrix(NdT, mymodel.dTK, mymodel.t0, mymodel.t1, mymodel.dt_forcing, base=mymodel.k_base)
        #         kt2D = kt_1D_to_2D(mymodel.pk, NdT, npk=len(mymodel.pk)//NdT*mymodel.nl)
        #         new_kt = np.dot(M,kt2D)
        #     else:
        #         new_kt = mymodel.pk
        #     coriolis = mymodel.fc.isel(lat=indy) * Ua1D
        #     wind = 
        
        # #ax[2].
        # ax[2].grid()
        ax[-1].set_xlabel('Time (days)')
        fig.savefig(path_save_png+name_save+'.png')
    
    # 2D plot
    # U total comparison
    nx, ny = U[0,:,:].shape
    indt = -1
    x = np.linspace(LON_bounds[0],LON_bounds[1], nx+1)
    y = np.linspace(LAT_bounds[0],LAT_bounds[1], ny+1)

    fig, ax = plt.subplots(2,1,figsize = (6,10),constrained_layout=True,dpi=dpi) 
    s = ax[0].pcolormesh(x, y, (np.array(Ua)+Ug)[indt], vmin=-0.6, vmax=0.6)
    plt.colorbar(s, ax=ax[0])
    ax[0].set_title('Ua+Ug')
    ax[0].set_aspect(1.0)
    s = ax[1].pcolormesh(x, y, (U+Ug)[indt], vmin=-0.6, vmax=0.6)
    plt.colorbar(s, ax=ax[1])
    ax[1].set_title('U+Ug')
    ax[1].set_aspect(1.0)
    
    fig, ax = plt.subplots(2,1,figsize = (7,10),constrained_layout=True,dpi=dpi) 
    s = ax[0].pcolormesh(x, y, Ua[indt], vmin=-0.6, vmax=0.6)
    plt.colorbar(s, ax=ax[0])
    ax[0].set_title('Ua')
    ax[0].set_aspect(1.0)
    s = ax[1].pcolormesh(x, y, U[indt], vmin=-0.6, vmax=0.6)
    plt.colorbar(s, ax=ax[1])
    ax[1].set_title('U')
    ax[1].set_aspect(1.0)
    