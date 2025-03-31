import numpy as np
import time as clock
import os
import jax.numpy as jnp

from inv import *
from models.classic_slab import kt_ini
from constants import oneday

def benchmark_model(model, var=None, fun='forward', Nexec=10):
    # selecting the function to benchmark
    if fun=='forward':
        func = model
    else:
        if fun in ['cost','gradcost']:
            if var is None:
                raise Exception(f'You want to benchmark {fun} but no variational class has been supplied, aborting...')
            dynamic_model, static_model = my_partition(model)
            if fun=='cost':
                func = lambda :var.cost(dynamic_model, static_model)
            elif fun=='gradcost':
                func = lambda :var.grad_cost(dynamic_model, static_model)
        else:
            raise Exception(f'I dont recognize the function ({fun}) you want to benchmark, aborting ...')
    
    # running the benchmark    
    Ltimes = []
    for _ in range(Nexec):
        t1 = clock.time()
        _ = func()
        Ltimes.append(clock.time()-t1)
    return np.array(Ltimes)
    
def benchmark_all(Lmodel, Lobservations, Nexec=10):
    """
    """
    L_model_slab = ['jslab','jslab_kt','jslab_kt_2D','jslab_rxry','jslab_Ue_Unio','jslab_kt_Ue_Unio']
    L_model_kt = ['jslab_kt','jslab_kt_2D','jslab_kt_Ue_Unio']
    L_model_2D = ['jslab_kt_2D']
    
    NB_dec = 8
    SAVE_PREVIOUS = False
    name_bench = 'benchmark_results' #+type(model).__name__
    
    Ltimes_forward = np.zeros((len(Lmodel),Nexec))
    Ltimes_cost = np.zeros((len(Lmodel),Nexec))
    Ltimes_grad = np.zeros((len(Lmodel),Nexec))
    Nb_param = np.zeros(len(Lmodel))
    for k in range(len(Lmodel)):
        is1D,is2D = True,False
        model = Lmodel[k]
        observations = Lobservations[k]
        print('     running:',type(model).__name__)
        var = Variational(model, observations)
        
        if type(model).__name__ in L_model_slab:
            NL = 1
        else:
            NL = model.nl
        
        if type(model).__name__ in L_model_2D:
            is1D,is2D = False,True
            LAT_bounds = [np.amin(observations.data.lat),np.amax(observations.data.lat)]
            LON_bounds = [np.amin(observations.data.lon),np.amax(observations.data.lon)]
        else:
            point_loc = [observations.data.lon.values,observations.data.lat.values]
        
        if type(model).__name__ in L_model_kt:
            NdT = len(np.arange(model.t0, model.t1, model.dTK)) # int((t1-t0)//dTK) 
            pk = kt_ini(model.pk, NdT)
        else:
            pk = model.pk
         
        Nb_param[k] = len(pk)        
        
        
        print('         forward')
        Ltimes_forward[k] = benchmark_model(model, fun='forward', Nexec=Nexec)       
        print('         cost')
        Ltimes_cost[k] = benchmark_model(model, var, fun='cost', Nexec=Nexec)
        print('         gradcost')
        Ltimes_grad[k] = benchmark_model(model, var, fun='gradcost', Nexec=Nexec)
    
    # writing in file
    if SAVE_PREVIOUS:
        os.system('mv -f '+name_bench+'.txt '+name_bench+'_previous.txt')
    
    with open(name_bench+".txt", "w") as f:
        f.write("*=============================================\n")
        f.write('* BENCHMARK: '+str(Nexec)+ ' runs of each func\n')
        if is1D:
            f.write(f'for 1D models: at LON,LAT={point_loc}\n')
        if is2D:
            f.write(f'for 2D models: LON={LON_bounds},LAT={LAT_bounds}\n')
        f.write('* if multilayers, N layer = '+str(NL)+'\n')
        f.write(f'* length of runtime = {(Lmodel[0].t1-Lmodel[0].t0)/oneday}\n')
        f.write(f'* dt = {Lmodel[0].dt}\n')       
        f.write("* C0: MODEL, Nb param\n")
        f.write("* C1: Mean Execution time (s)\n")
        f.write("* C2: Std Execution time (s)\n")
        f.write("* C3: Compilation time (s)\n")
        
        for k in range(len(Lmodel)):
            name = type(Lmodel[k]).__name__
            # compilation times
            txt_f = str(np.round( Ltimes_forward[k,0]-Ltimes_forward[k,1],NB_dec ))
            txt_c = str(np.round( Ltimes_cost[k,0]-Ltimes_cost[k,1],NB_dec ))
            txt_g = str(np.round( Ltimes_grad[k,0]-Ltimes_grad[k,1],NB_dec ))
            ind0 = 1
                
            f.write(name + ', '+str(int(Nb_param[k]))+'\n')
            f.write('   - forward,'+str(np.round( np.mean(Ltimes_forward[k,ind0:]),NB_dec ))+','+
                                        str(np.round( np.std(Ltimes_forward[k,ind0:]),NB_dec ))+','+
                                        txt_f+'\n')
            f.write('   - cost   ,'+str(np.round(np.mean(Ltimes_cost[k,ind0:]),    NB_dec ))+','+
                                        str(np.round( np.std(Ltimes_cost[k,ind0:]),NB_dec ))+   ','+
                                        txt_c+'\n')
            f.write('   - grad   ,'+str(np.round( np.mean(Ltimes_grad[k,ind0:]),   NB_dec ))+','+
                                        str(np.round( np.std(Ltimes_grad[k,ind0:]),NB_dec ))+   ','+
                                        txt_g+'\n')
            f.write('   - grad/cost,'+str(np.round( np.mean(Ltimes_grad[k,ind0:]/Ltimes_cost[k,ind0:]),2))+'\n')
        
        
        f.write("*=============================================\n")
        f.write("MODELS DEFINITIONS\n")
        for k in range(len(Lmodel)):
            name = type(Lmodel[k]).__name__
            f.write(name +'\n')
            attrs = Lmodel[k].__dict__
            for attr in attrs.keys():
                if attr=='TAx' or attr=='TAy':
                    f.write('   '+attr+' : '+str(attrs[attr].shape)+' terms\n')
                else:
                    f.write('   '+attr+' = '+str(attrs[attr])+'\n')
            f.write(''+'\n')    
        
    f = open(name_bench+'.txt')
    print('Results:')
    for line in f:
        print(line[:-1]),
    f.close()