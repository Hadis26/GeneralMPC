import numpy as np


def violations(values,lower,upper):
    return np.maximum(lower-values,0.)+np.maximum(values-upper,0.)


def complementarity(values,lower,upper,dual):
    """Equality multipliers are free; inequality multipliers use OSQP signs."""
    unequal = lower != upper
    hi = np.isfinite(upper) & unequal
    lo = np.isfinite(lower) & unequal
    errors = [0.]
    if np.any(hi):
        errors.extend(np.abs(np.maximum(dual[hi],0.)*(upper[hi]-values[hi])))
    if np.any(lo):
        errors.extend(np.abs(np.minimum(dual[lo],0.)*(values[lo]-lower[lo])))
    # For a missing upper/lower bound, the corresponding signed dual must vanish.
    errors.extend(np.maximum(dual[unequal & ~np.isfinite(upper)],0.))
    errors.extend(np.maximum(-dual[unequal & ~np.isfinite(lower)],0.))
    return float(max(errors))


def line_search(evaluate,w,theta,d,objective,gradient,g,lbg,ubg,lbx,ubx,dual,options):
    """L1 exact-penalty merit and Armijo backtracking; MATMPC Eq. 7."""
    violation0 = violations(g,lbg,ubg).sum()+violations(w,lbx,ubx).sum()
    rho = max(
        10.0,
        1.1 * float(np.max(np.abs(dual), initial=0.0)),
    )
    descent = float(gradient@d)-rho*violation0
    if violation0 > 1e-12 and descent >= 0.:
        rho = max(rho,(float(gradient@d)+abs(float(gradient@d))+1.)/violation0)
        descent = float(gradient@d)-rho*violation0
    if descent >= 0.:
        raise RuntimeError("SQP direction is not a descent direction for the merit function")
    merit0 = objective+rho*violation0
    alpha = 1.
    for _ in range(options.max_line_search_iterations):
        candidate = w+alpha*d
        f,gnew = evaluate(candidate,theta)
        gnew = np.array(gnew).ravel()
        merit = float(f)+rho*(violations(gnew,lbg,ubg).sum()+violations(candidate,lbx,ubx).sum())
        if np.isfinite(merit) and merit <= merit0+1e-4*alpha*descent:
            return candidate,alpha
        alpha *= 0.5
    raise RuntimeError("SQP line search failed; inspect scaling, initialization or horizon")
