"""
This module gathers list of names of modeles
"""
import sys, inspect
from pprint import pprint

import models.classic_slab as classic_slab
import models.unsteak as unsteak

import tools

# List of modules where models are declared.
#   you also need to import them
#   for e.g. : import unsteak
L_modules = ['models.classic_slab','models.unsteak']
"""
L_all

This list gather all models
"""
L_all = []
for module in L_modules:
    L_all = L_all + tools.get_models_name_from_module(module)




"""
L_unsteak

This list gather all unsteak models
"""
L_unsteaks = tools.get_models_name_from_module('models.unsteak')



"""
L_slab

This list gather all slab models
"""
L_slabs = tools.get_models_name_from_module('models.classic_slab')




"""
L_1D_models

This list gather models that are 1D (only time)
"""
L_1D_models = ['jslab',
               'jslab_fft',
               'jslab_kt',
               'jslab_rxry',
               'jslab_Ue_Unio',
               'jslab_kt_Ue_Unio',
               'junsteak',
               'junsteak_kt',]

"""
L_2D_models

This list gather models that are 2D (time and space XY)
"""
L_2D_models = ['jslab_kt_2D',
               'jslab_kt_2D_adv',
               'jslab_kt_2D_adv_Ut',
               'junsteak_kt_2D',
               'junsteak_kt_2D_adv']


"""
L_variable_Kt

This list gather models that have control parameters that changes with time, 
    possibly with a reduced basis
"""
L_variable_Kt = ['jslab_kt',
                 'jslab_kt_2D',
                 'jslab_kt_Ue_Unio',
                 'jslab_kt_2D_adv',
                 'jslab_kt_2D_adv_Ut',
                 'junsteak_kt',
                 'junsteak_kt_2D',
                 'junsteak_kt_2D_adv']




"""
L_nlayers_models

This list gather models that have more than 1 layer, 
    so we need to tell the code to use the surface current in the cost function
"""
L_nlayers_models = ['junsteak',
                    'junsteak_kt',
                    'junsteak_kt_2D',
                    'junsteak_kt_2D_adv']




"""
L_models_total_current

This list gather models that computes total curren
                U = Ug + Uag
    and so we compare the output to total current from observations
"""
L_models_total_current = ['jslab_kt_2D_press']




"""
L_filtered_forcing

This list gather all models that takes as inputs a filtered (at fc) forcing
"""
L_filtered_forcing = ['jslab_Ue_Unio']