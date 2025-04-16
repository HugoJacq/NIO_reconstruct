


from constants import *
from Listes_models import *


def compute_and_save_pk(model, var, mini_args):
    
    maxiter = mini_args['maxiter']

    # minimize
    mymodel, _ = var.scipy_lbfgs_wrapper(model, maxiter, verbose=True)   
    
    # get pk
    
    # save pk
            
            
