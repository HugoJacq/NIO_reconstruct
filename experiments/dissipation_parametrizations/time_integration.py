"""
Euler integration scheme here
"""

def Integration_Euler(X0, forcing, features, RHS, dt, Nsteps):
    X = X0
    for k in range(Nsteps):
        X = X + dt*RHS(X, forcing, features)
    return X