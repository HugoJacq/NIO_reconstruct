from jax.experimental.ode import odeint

def solve_ode(U0, t_span, f, K0, tau, dissipation_model):
    # Solve the differential equation
    return odeint(rhs, U0, t_span, f, K0, tau, dissipation_model)

def rhs(U, t, f, K0, tau, dissipation_model):
    dissipation = dissipation_model(U, tau)
    return -1j * f * U + K0 * tau - dissipation