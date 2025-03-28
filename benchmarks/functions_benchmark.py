import numpy as np
import time as clock
import os
import jax.numpy as jnp

from inv import *

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
                func = var.cost(dynamic_model, static_model)
            elif fun=='gradcost':
                func = var.grad_cost(dynamic_model, static_model)
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
    
    
    NB_dec = 8
    SAVE_PREVIOUS = False
    name_bench = 'benchmark_'+type(model).__name__
    
    Ltimes_forward = np.zeros((len(Lmodel),Nexec))
    Ltimes_cost = np.zeros((len(Lmodel),Nexec))
    Ltimes_grad = np.zeros((len(Lmodel),Nexec))
    Nb_param = np.zeros(len(Lmodel))
    for k in range(len(Lmodel)):
        model = Lmodel[k]
        observations = Lobservations[k]
        print('     running:',type(model).__name__)
        var = Variational(model, observations)
        
        if type(model).__name__ in L_model_slab:
            NL = 1
        else:
            NL = model.nl
        
        # if type(model).__name__ in L_model_kt:
        #     jpk = model.kt_2D_to_1D(model.kt_ini(pk))
            
        Nb_param[k] = len(model.pk)        
        
        # if type(model).__name__ == 'jUnstek1D_Kt':
        #     jpk = model.kt_2D_to_1D(model.kt_ini(jpk))
        #     Nb_param[k] = model.kt_ini(jpk).shape[0]*model.nl*2
        
        Ltimes_forward[k] = benchmark_model(model, fun='forward', Nexec=Nexec)       
        Ltimes_cost[k] = benchmark_model(model, var, fun='cost', Nexec=Nexec)
        Ltimes_grad[k] = benchmark_model(model, var, fun='gradcost', Nexec=Nexec)
    
    # writing in file
    if SAVE_PREVIOUS:
        os.system('mv -f '+name_bench+'.txt '+name_bench+'_previous.txt')
    
    with open(name_bench+".txt", "w") as f:
        f.write("*=============================================\n")
        f.write('* BENCHMARK: '+str(Nexec)+ ' runs of each func\n')
        f.write('* N layer = '+str(NL)+'\n')
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
                
            f.write(name + ', '+str(Nb_param[k])+'\n')
            f.write('   - forward, '+str(np.round( np.mean(Ltimes_forward[ind0:]),NB_dec ))+', '+
                                        str(np.round( np.std(Ltimes_forward[ind0:]),NB_dec ))+', '+
                                        txt_f+'\n')
            f.write('   - cost   , '+str(np.round(np.mean(Ltimes_cost[ind0:]),    NB_dec ))+', '+
                                        str(np.round( np.std(Ltimes_cost[ind0:]),NB_dec ))+   ', '+
                                        txt_c+'\n')
            f.write('   - grad   , '+str(np.round( np.mean(Ltimes_grad[ind0:]),   NB_dec ))+', '+
                                        str(np.round( np.std(Ltimes_grad[ind0:]),NB_dec ))+   ', '+
                                        txt_g+'\n')
        
    f = open(name_bench+'.txt')
    print('Results:')
    for line in f:
        print(line[:-1]),
    f.close()