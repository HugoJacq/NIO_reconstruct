


from constants import *
from Listes_models import *

from functions import save_pk

def compute_and_save_pk(model, var, mini_args, save_name):
    
    maxiter = mini_args['maxiter']
    


    # test if pk is here

    # minimize to find new pk
    mymodel, _ = var.scipy_lbfgs_wrapper(model, maxiter, verbose=True)   
    
    # save pk
    save_pk(model)
            