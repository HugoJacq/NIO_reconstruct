"""
Here function that runs the tests
"""
import time as clock
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../src')
import os
import xarray as xr

import tools
import models.classic_slab as classic_slab
import models.unsteak as unsteak
from basis import kt_ini, kt_1D_to_2D, pkt2Kt_matrix
from constants import *
from inv import my_partition
from Listes_models import L_slabs, L_unsteaks, L_variable_Kt, L_nlayers_models, L_models_total_current

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
    
def plot_traj_1D(mymodel, call_args, var_dfx, forcing1D, observations1D, name_save, path_save_png, dpi):
    
    Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
    if type(mymodel).__name__ in L_nlayers_models:
        Ua, Va = Ua[:,0], Va[:,0]
    U = forcing1D.data.U.values
    V = forcing1D.data.V.values
    t0, t1, _ = call_args
    Uo, _ = observations1D.get_obs()
    RMSE = tools.score_RMSE(Ua, U) 
    dynamic_model, static_model = my_partition(mymodel)
    #final_cost = var_dfx.cost(dynamic_model, static_model)
        
    SHOW_BASE_TRANSFORM = False   
    SHOW_MLD = False
    
    nlines = 3 if SHOW_MLD else 2
    
    if type(mymodel).__name__ in L_unsteaks:
        modeltype = 'unsteak'
    elif type(mymodel).__name__ in L_slabs:
        modeltype = 'slab'
    
    
    print('RMSE is',RMSE)
    # PLOT trajectory
    fig, ax = plt.subplots(nlines,1,figsize = (10,10),constrained_layout=True,dpi=dpi)
    if var_dfx.filter_at_fc:
        ax[0].plot((t0 + forcing1D.time)/oneday, U, c='k', lw=2, label='Croco', alpha=0.3)
        ax[0].plot((t0 + forcing1D.time)/oneday, Ua, c='g', label=modeltype, alpha = 0.3)
        (Ut_nio,Vt_nio) = tools.my_fc_filter(mymodel.dt_forcing,U+1j*V, mymodel.fc)
        ax[0].plot((t0 + forcing1D.time)/oneday, Ut_nio, c='k', lw=2, label='Croco at fc', alpha=1)
        (Unio,Vnio) = tools.my_fc_filter(mymodel.dt_forcing, Ua+1j*Va, mymodel.fc)
        ax[0].plot((t0 + forcing1D.time)/oneday, Unio, c='b', label=modeltype+' at fc')
    else:
        ax[0].plot((t0 + forcing1D.time)/oneday, U, c='k', lw=2, label='Croco', alpha=1)
        ax[0].plot((t0 + forcing1D.time)/oneday, Ua, c='g', label=modeltype)
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
    if SHOW_MLD:
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
    fig.savefig(path_save_png+name_save+'_t'+str(int(t0//oneday))+'_'+str(int(t1//oneday))+'.png')
    
def plot_traj_2D(mymodel, var_dfx, forcing2D, observations2D, name_save, point_loc, LON_bounds, LAT_bounds, attime, path_save_png, dpi):
    Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
    if type(mymodel).__name__ in L_nlayers_models:
        Ua, Va = Ua[:,0], Va[:,0]
    U = forcing2D.data.U
    V = forcing2D.data.V
    Ug = forcing2D.data.Ug
    Vg = forcing2D.data.Vg
    TAx, TAy = forcing2D.data.oceTAUX, forcing2D.data.oceTAUY
    
    t0, t1 = mymodel.t0, mymodel.t1
    
    indx = tools.nearest(forcing2D.data.lon.values, point_loc[0])
    indy = tools.nearest(forcing2D.data.lat.values, point_loc[1])
    if type(mymodel).__name__ in L_models_total_current:
        U1D = U.values + Ug.values
        V1D = V.values + Vg.values
        Uo, _ = observations2D.get_obs(is_utotal=True)
    else:
        U1D = U.values
        V1D = V.values
        Uo, _ = observations2D.get_obs(is_utotal=False)
        
    U1D = U1D[:,indy,indx]
    V1D = V1D[:,indy,indx]
    TAx1D = TAx.values[:,indy,indx]
    TAy1D = TAy.values[:,indy,indx]
    Ua1D = Ua[:,indy,indx]
    Va1D = Va[:,indy,indx]
    Uo1D = Uo[:,indy,indx]
    
    RMSE = tools.score_RMSE(Ua1D, U1D) 

    if type(mymodel).__name__ in L_unsteaks:
        modeltype = 'unsteak'
    elif type(mymodel).__name__ in L_slabs:
        modeltype = 'slab'
    else:
        raise Exception(f'Your model {type(mymodel).__name__} is not recognied.')

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
            ax[0].plot((t0 + forcing2D.time)/oneday, Ua1D, c='g', label=modeltype, alpha = 0.3)
            (Ut_nio,Vt_nio) = tools.my_fc_filter(mymodel.dt_forcing,U1D+1j*V1D, mymodel.fc)
            ax[0].plot((t0 + forcing2D.time)/oneday, Ut_nio, c='k', lw=2, label='Croco at fc', alpha=1)
            (Unio,Vnio) = tools.my_fc_filter(mymodel.dt_forcing, Ua1D+1j*Va1D, mymodel.fc)
            ax[0].plot((t0 + forcing2D.time)/oneday, Unio, c='b', label=modeltype+' at fc')
        else:
            ax[0].plot((t0 + forcing2D.time)/oneday, U1D, c='k', lw=2, label='Croco at loc', alpha=1)
            ax[0].plot((t0 + forcing2D.time)/oneday, Ua1D, c='g', label=modeltype+' at loc')
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
    #attime = 70 # day number of the year
    indt = tools.nearest( (t0 + forcing2D.time)/oneday, attime)
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
    
    
def idealized_run(mymodel, forcing1D, name_save, path_save_png, dpi):

    Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
    if type(mymodel).__name__ in ['junsteak','junsteak_kt']:
        Ua, Va = Ua[:,0], Va[:,0]
    stressX = forcing1D.TAx
    fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
    ax.plot((mymodel.t0 + forcing1D.time)/oneday, Ua, c='k')
    ax.set_ylabel('U current m/s')
    ax2 = ax.twinx()
    ax2.plot((mymodel.t0 + forcing1D.time)/oneday, stressX, c='b')
    ax2.set_ylabel('Step stress')
    ax.grid()
    ax.set_xlabel('days')
    fig.savefig(path_save_png + name_save + 'U.png')
    
    fig, ax = plt.subplots(1,1,figsize = (10,5),constrained_layout=True,dpi=dpi)
    ax.plot(Ua, Va, c='k')
    ax.set_xlabel('U m/s')
    ax.set_ylabel('V m/s')
    fig.savefig(path_save_png + name_save + 'UV.png')
    
    
def make_film(mymodel, forcing2D, LON_bounds, LAT_bounds, namesave_loc_area, path_save_png):
    
    # getting data
    Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
    if type(mymodel).__name__ in L_nlayers_models:
        Ua, Va = Ua[:,0], Va[:,0]
    U = forcing2D.data.U
    V = forcing2D.data.V
    Ug = forcing2D.data.Ug
    Vg = forcing2D.data.Vg
    TAx, TAy = forcing2D.data.oceTAUX, forcing2D.data.oceTAUY
    t0, t1 = mymodel.t0, mymodel.t1
    nx, ny = U[0,:,:].shape
    x = np.linspace(LON_bounds[0],LON_bounds[1], nx+1)
    y = np.linspace(LAT_bounds[0],LAT_bounds[1], ny+1)
    
    vmin, vmax = -0.6, 0.6
    
    subfolder = path_save_png + type(mymodel).__name__ + namesave_loc_area
    os.system('mkdir -p '+subfolder)
    
    
    for it in range(len(forcing2D.time)): 
        
        fig, ax = plt.subplots(2,2,figsize = (12,10),constrained_layout=True,dpi=100)

        # Ua
        s = ax[0,0].pcolormesh(x, y, np.array(Ua)[it], vmin=-0.6, vmax=0.6)
        plt.colorbar(s, ax=ax[0,0])
        ax[0,0].set_title('Ua')
        ax[0,0].set_aspect(1.0)
        # U
        s = ax[0,1].pcolormesh(x, y, U[it], vmin=-0.6, vmax=0.6)
        plt.colorbar(s, ax=ax[0,1])
        ax[0,1].set_title('U')
        ax[0,1].set_aspect(1.0)
        # Ua+Ug
        s = ax[1,0].pcolormesh(x, y, (np.array(Ua)+Ug)[it], vmin=-0.6, vmax=0.6)
        plt.colorbar(s, ax=ax[1,0])
        ax[1,0].set_title('Ua+Ug')
        ax[1,0].set_aspect(1.0)
        # U+Ug
        s = ax[1,1].pcolormesh(x, y, (U+Ug)[it], vmin=-0.6, vmax=0.6)
        plt.colorbar(s, ax=ax[1,1])
        ax[1,1].set_title('U+Ug')
        ax[1,1].set_aspect(1.0)

        fig.savefig(subfolder + f'/frame_{it}.png')
        plt.close(fig)
        
def save_output_as_nc(mymodel, forcing2D, LON_bounds, LAT_bounds, name_save, where_to_save):
    
    # getting data
    Ua,Va = mymodel(save_traj_at=mymodel.dt_forcing)
    if type(mymodel).__name__ in L_nlayers_models:
        Ua, Va = Ua[:,0], Va[:,0]
    U = forcing2D.data.U
    V = forcing2D.data.V
    Ug = forcing2D.data.Ug
    Vg = forcing2D.data.Vg
    
    t0, t1 = mymodel.t0, mymodel.t1
    x = np.arange(LON_bounds[0],LON_bounds[1], 0.1)
    y = np.arange(LAT_bounds[0],LAT_bounds[1], 0.1)  
    
    #time = np.arange(t0,t1,forcing2D.dt_forcing)
    
    """
    TO BE ADDED
    - the pk vector
    - has the parameters been through minimization ?
    - U truth
    - model static parameters: time window, dt, ...
    - origin of truth -> Croco
    
    - let the possibility for the user to add a note in the terminal window ??
    
    """
    
    os.system('mkdir -p ' + where_to_save)
    
    # making the dataset
    ds = xr.Dataset({"Ua": (["time", "lat", "lon"], np.asarray(Ua)),
                     "Ug": (["time", "lat", "lon"], Ug.data),
                     "Uat": (["time", "lat", "lon"], np.asarray(Ua)+Ug.values),
                     "Ut": (["time", "lat", "lon"], (U+Ug).data),
                    },
                    coords={
                        "lon": (["lon"], x),
                        "lat": (["lat"], y),
                        "time": forcing2D.data.time,
                            },
                    attrs={
                        "made_from":"save_output_as_nc",
                        "model_used":type(mymodel).__name__
                    }

                    )
    ds.to_netcdf(where_to_save + name_save + '.nc')
    
def memory_profiler(mymodel):
    
    U,V = mymodel()
    U.block_until_ready()
    V.block_until_ready()

    name_out = 'memory_profile_'+type(mymodel).__name__+'_forward.prof'

    jax.profiler.save_device_memory_profile(name_out)
    # then run: pprof --svg memory.prof
    
    # THIS IS WIP    
    # traced = jax.jit(mymodel).trace() 
    # print(traced.jaxpr)
    
    

def model_instanciation(model_name, forcing, args_model, call_args, extra_args):
    """
    Wrapper of every model instanciation.
    
    
    TBD: doc of this, with typing
    """
    t0, t1, dt = call_args
    dTK = args_model['dTK']
    Nl = args_model['Nl']
    
    # MODELS FROM MODULE classic_slab
    if model_name=='jslab':
        pk = jnp.asarray([-11., -10.]) 
        model = classic_slab.jslab(pk, forcing, call_args, extra_args)
        
    elif model_name=='jslab_fft':
        pk = jnp.asarray([-11., -10.]) 
        model = classic_slab.jslab_fft(pk, forcing, call_args, extra_args)
        
    elif model_name=='jslab_kt':
        pk = jnp.asarray([-11., -10.])   
        NdT = len(np.arange(t0, t1,dTK))
        pk = kt_ini(pk, NdT)
        model = classic_slab.jslab_kt(pk, dTK, forcing, call_args, extra_args)
        
    elif model_name=='jslab_kt_2D':
        pk = jnp.asarray([-11., -10.])   
        NdT = len(np.arange(t0, t1, dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = classic_slab.jslab_kt_2D(pk, dTK, forcing, call_args, extra_args)
        
    elif model_name=='jslab_rxry':
        pk = jnp.asarray([-11., -10., -9.]) 
        model = classic_slab.jslab_rxry(pk, forcing, call_args, extra_args)
        
    elif model_name=='jslab_Ue_Unio':
        TA = forcing.TAx + 1j*forcing.TAy
        TAx_f,TAy_f = tools.my_fc_filter( forcing.dt_forcing, TA, forcing.fc)
        forcing.TAx = TAx_f
        forcing.TAy = TAy_f
        pk = jnp.asarray([-11., -10.]) 
        model = classic_slab.jslab_Ue_Unio(pk, forcing, call_args, extra_args)
        
    elif model_name=='jslab_kt_Ue_Unio':
        TA = forcing.TAx + 1j*forcing.TAy
        TAx_f,TAy_f = tools.my_fc_filter( forcing.dt_forcing, TA, forcing.fc)
        forcing.TAx = TAx_f
        forcing.TAy = TAy_f
        pk = jnp.asarray([-11., -10.])   
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = classic_slab.jslab_kt_Ue_Unio(pk, dTK, forcing, call_args, extra_args)
        
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
        
    
    
    # # MODELS FROM MODULE unsteak
    elif model_name=='junsteak':
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-15.,-10., -10., -10.])   
        model = unsteak.junsteak(pk, forcing, Nl, call_args, extra_args)   
        
    elif model_name=='junsteak_kt':
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-15.,-10., -10., -10.]) 
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = unsteak.junsteak_kt(pk, dTK, forcing, Nl, call_args, extra_args)   
        
    elif model_name=='junsteak_kt_2D':
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-15.,-10., -10., -10.]) 
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = unsteak.junsteak_kt_2D(pk, dTK, forcing, Nl, call_args, extra_args)   
    
    elif model_name=='junsteak_kt_2D_adv':
        if Nl==1:
            pk = jnp.asarray([-11.31980127, -10.28525189])    
        elif Nl==2:
            pk = jnp.asarray([-15.,-10., -10., -10.]) 
        NdT = len(np.arange(t0, t1,dTK)) # int((t1-t0)//dTK) 
        pk = kt_ini(pk, NdT)
        model = unsteak.junsteak_kt_2D_adv(pk, dTK, forcing, Nl, call_args, extra_args)   
        
    else:
        raise Exception(f'You want to test the mode {model_name} but it is not recognized')

    
    return model
    