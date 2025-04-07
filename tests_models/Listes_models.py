"""
This module gathers list of names of modeles
"""


"""
L_variable_Kt

This list gather models that have control parameters that changes with time, 
    possibly with a reduced basis
"""
L_variable_Kt = []




"""
L_nlayers_models

This list gather models that have more than 1 layer, 
    so we need to tell the code to use the surface current in the cost function
"""
L_nlayers_models = ['junsteak','junsteak_kt','junsteak_kt_2D']




"""
L_models_total_current

This list gather models that computes total curren
                U = Ug + Uag
    and so we compare the output to total current from observations
"""
L_models_total_current = ['jslab_kt_2D_press']




"""
L_unsteak

This list gather all unsteak models
"""
L_unsteaks = ['junsteak','junsteak_kt','junsteak_kt_2D']




"""
L_slab

This list gather all unsteak models
"""
L_slabs = ['jslab','jslab_fft','jslab_kt','jslab_kt_2D','jslab_rxry','jslab_Ue_Unio','jslab_kt_Ue_Unio','jslab_kt_2D_adv','jslab_kt_2D_adv_Ut']

