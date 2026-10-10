import casadi as ca


def rk4(rhs, nx, nu, np_, dt, substeps):
    """Fixed-step RK4; control and online parameters held constant per interval."""
    x, u, p = ca.SX.sym("x", nx), ca.SX.sym("u", nu), ca.SX.sym("p", np_)
    y, h = x, dt/substeps
    for _ in range(substeps):
        k1 = rhs(y, u, p)
        k2 = rhs(y+h*k1/2, u, p)
        k3 = rhs(y+h*k2/2, u, p)
        k4 = rhs(y+h*k3, u, p)
        y += h*(k1+2*k2+2*k3+k4)/6
    step = ca.Function("step", [x,u,p], [y])
    sensitivity = ca.Function("step_jac", [x,u,p], [ca.jacobian(y,x),ca.jacobian(y,u)])
    return step, sensitivity
