import xarray as xr
import optax 

d_base_config = {
        # "slab":{'optimizer':'adam',         # adam or lbfgs
        #         'linear_lr': (1e-1, 1e-3, 40, 40), # lr_start, lr_end, ntr, start_tr
        #         'MAX_STEP':100,             # number of epochs
        #         'PRINT_EVERY':1,            # print infos every 'PRINT_EVERY' epochs
        #         'features_names':[],        # what features to use in the NN (U,V in by default)
        #         'forcing_names':[],         # U,V,TAx,TAy already in by default
        #         'BATCH_SIZE':300,            # size of a batch (time), set to -1 for no batching
        #         'L_to_be_normalized':''},
        "slab":{'optimizer':'lbfgs',         # adam or lbfgs
                'linear_lr': (1e-1, 1e-1, 40, 40), # lr_start, lr_end, ntr, start_tr
                'MAX_STEP':100,             # number of epochs
                'PRINT_EVERY':1,            # print infos every 'PRINT_EVERY' epochs
                'features_names':[],        # what features to use in the NN (U,V in by default)
                'forcing_names':[],         # U,V,TAx,TAy already in by default
                'BATCH_SIZE':-1,            # size of a batch (time), set to -1 for no batching
                'L_to_be_normalized':''},
        
        "CNN":{'optimizer':'adam',
                'linear_lr': (1e-4, 1e-6, 20, 50), # lr_start, lr_end, ntr, start_tr
                'MAX_STEP':50,
                'PRINT_EVERY':1,
                'features_names':['TAx','TAy'],
                'forcing_names':[],
                'BATCH_SIZE':300,
                'L_to_be_normalized':'features'},
        
        "MLP":{'optimizer':'adam',
                'linear_lr': (1e-5, 1e-6, 10, 10), # lr_start, lr_end, ntr, start_tr
                'MAX_STEP':50,
                'PRINT_EVERY':1,
                'features_names':[],
                'forcing_names':[],
                'BATCH_SIZE':300,
                'L_to_be_normalized':'features'},
        # "MLP_linear":{'optimizer':'sgd',
        #         'linear_lr': (10, 1, 5, 5), # lr_start, lr_end, ntr, start_tr
        #         'MAX_STEP':20,
        #         'PRINT_EVERY':1,
        #         'features_names':['TAx','TAy'],
        #         'forcing_names':[],
        #         'BATCH_SIZE':-1, # 300
        #         'L_to_be_normalized':'features'},
        "MLP_linear":{'optimizer':'lbfgs',
                'linear_lr': (1e-1, 1e-1, 40, 40), # lr_start, lr_end, ntr, start_tr
                'MAX_STEP':20,
                'PRINT_EVERY':1,
                'features_names':['TAx','TAy'],
                'forcing_names':[],
                'BATCH_SIZE':-1, # 300
                'L_to_be_normalized':''},
        # "MLP_linear":{'optimizer':'adam', # <- only last layer init as N(0,1e-4)
        #         'linear_lr': (1e-6, 1e-8, 0, 10), # lr_start, lr_end, ntr, start_tr
        #         'MAX_STEP':100,
        #         'PRINT_EVERY':1,
        #         'features_names':[],
        #         'forcing_names':[],
        #         'BATCH_SIZE':300,
        #         'L_to_be_normalized':''},
        # "MLP_linear":{'optimizer':'adam',
        #         'linear_lr': (1e-5, 1e-6, 10, 10), # lr_start, lr_end, ntr, start_tr
        #         'MAX_STEP':50,
        #         'PRINT_EVERY':1,
        #         'features_names':['TAx','TAy'],
        #         'forcing_names':[],
        #         'BATCH_SIZE':300,
        #         'L_to_be_normalized':''},
            }
 
d_training_config = {
        "offline":{'N_integration_steps':2,             # 2 for offline, more for online
                    'N_integration_steps_verif':2},     # number of time step to integrate during in use cases of the model
        "online":{'N_integration_steps':-1,
                    'N_integration_steps_verif':-1},
                        }

"""
Note: 

-> N_integration_steps and N_integration_steps_verif start at 2 because you need the initial solution AND at least one time step
-> features_names has already U and V if set to []
"""










def get_config(name_model, mode):
    
    if d_base_config[name_model]['optimizer']=='adam':
        lr_start, lr_end, ntr, start_tr = d_base_config[name_model]['linear_lr']
        OPTI = optax.adam(optax.linear_schedule(lr_start, lr_end, transition_steps=ntr, transition_begin=start_tr))
    elif d_base_config[name_model]['optimizer']=='lbfgs':
        OPTI = optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch( max_linesearch_steps=55, verbose=True))
    elif d_base_config[name_model]['optimizer']=='sgd':
        lr_start, lr_end, ntr, start_tr = d_base_config[name_model]['linear_lr']
        OPTI = optax.sgd(optax.linear_schedule(lr_start, lr_end, transition_steps=ntr, transition_begin=start_tr))
    else:
        raise Exception(f'Optimizer {d_base_config[name_model]["optimizer"]} is not recognized')
    MAX_STEP = d_base_config[name_model]['MAX_STEP']
    PRINT_EVERY = d_base_config[name_model]['PRINT_EVERY']
    FEATURES_NAMES = d_base_config[name_model]['features_names']
    FORCING_NAMES = d_base_config[name_model]['forcing_names']
    BATCH_SIZE = d_base_config[name_model]['BATCH_SIZE']
    L_TO_BE_NORMALIZED = d_base_config[name_model]['L_to_be_normalized']
    
    N_integration_steps = d_training_config[mode]['N_integration_steps']
    N_integration_steps_verif = d_training_config[mode]['N_integration_steps_verif']
    
    my_base_config = OPTI, MAX_STEP, PRINT_EVERY, FEATURES_NAMES, FORCING_NAMES, BATCH_SIZE, L_TO_BE_NORMALIZED
    my_training_config = N_integration_steps, N_integration_steps_verif
    return my_base_config, my_training_config





def my_warnings(dataset        : xr.core.dataset, 
                config         : dict, 
                train_config   : dict,
                test_ratio     : float):
    
    OPTI, MAX_STEP, PRINT_EVERY, FEATURES_NAMES, FORCING_NAMES, BATCH_SIZE, L_TO_BE_NORMALIZED = config
    N_integration_steps, N_integration_steps_verif = train_config
    Ntests = test_ratio*len(dataset.time)//100 # how many instants used for test
    
    if BATCH_SIZE<0:
        BATCH_SIZE = len(dataset.time) - Ntests 
        if N_integration_steps <0:
            N_integration_steps = BATCH_SIZE
        if N_integration_steps_verif<0:
            N_integration_steps_verif = BATCH_SIZE
        
    if N_integration_steps > BATCH_SIZE and BATCH_SIZE>0:
        print(f'You have chosen to do online training but the number of integration step ({N_integration_steps}) is greater than the batch_size ({BATCH_SIZE})')
        print(f'N_integration_steps has been reduced to the batch size value ({BATCH_SIZE})')
        N_integration_steps = BATCH_SIZE
    if N_integration_steps<0:
        print(f'N_integration_steps is < 0, N_integration_steps is set to = BATCH_SIZE ({BATCH_SIZE})')
        N_integration_steps = BATCH_SIZE
    if BATCH_SIZE%N_integration_steps!=0:
        raise Exception(f'N_integration_steps is not a divider of BATCH_SIZE: {N_integration_steps}%{BATCH_SIZE}={BATCH_SIZE%N_integration_steps}, try again')
    
    my_base_config = OPTI, MAX_STEP, PRINT_EVERY, FEATURES_NAMES, FORCING_NAMES, BATCH_SIZE, L_TO_BE_NORMALIZED
    my_training_config = N_integration_steps, N_integration_steps_verif
    return my_base_config, my_training_config 


def prepare_config(dataset, name_model, training_mode, test_ratio):
    my_base_config, my_training_config = get_config(name_model, training_mode)
    my_base_config, my_training_config = my_warnings(dataset, my_base_config, my_training_config, test_ratio)
    return my_base_config, my_training_config