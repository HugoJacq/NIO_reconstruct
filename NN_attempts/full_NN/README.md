# Using a NN to predict next time step

This is a example to show:
- how to used JAX and Equinox to build a NN
- how to train a NN
- demonstrate that a CNN (convolutional NN) can reconstructe the total current given maps of SSH and wind stress
        
But this example is also:
- a proof of concept, results should not be taken as publication ready
- a toy example to play with different NN architectures
- a preliminary example for a more complex NN: modeling dissipation inside a equation


INPUT of the NN: Ug, Vg, Taux, Tauy

OUPUT of the NN: U total,V total

I give the inputs at time t, and i want to predict U,V at time t+dt.
My data is 1 month, so i give 'Ntime' input (from t0 to t0+Ntime*dt) 
and i train on the ouputs at t0+dt to Ntime*(dt+1).
So that my ouput are shifted in time by 1 dt.

My test data is using the same structure on the rest of the data in the month

Note: when running the script, JAX/XLA print many warnings in the terminal. Is it important ?

Author: Hugo Jacquet, April 2025
