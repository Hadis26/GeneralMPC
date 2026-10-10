"""MATMPC Eq. 1 -> sparse NLP; Eqs. 4-5 -> Gauss-Newton QP data."""
import casadi as ca
import numpy as np
from .model import bound_vector
from .integrators import rk4


class Transcription:
    def __init__(self, model, bounds, options):
        self.model, self.options = model, options
        nx, nu, np_, N = model.nx, model.nu, model.np, options.N
        self.step, self.step_jac = rk4(model.rhs,nx,nu,np_,options.dt,options.integrator_substeps)
        X, U = ca.SX.sym("X",nx,N+1), ca.SX.sym("U",nu,N)
        P, measured = ca.SX.sym("P",np_,N+1), ca.SX.sym("measured",nx)
        w = ca.vertcat(ca.vec(X),ca.vec(U))
        theta = ca.vertcat(measured,ca.vec(P))
        self.nw, self.n_state_variables = w.numel(), nx*(N+1)
        eq = [X[:,0]-measured]
        residuals, path, low, high = [], [], [], []
        ng = model.path.numel_out(0) if model.path is not None else 0
        ngN = model.terminal_path.numel_out(0) if model.terminal_path is not None else 0
        for k in range(N):
            eq.append(X[:,k+1]-self.step(X[:,k],U[:,k],P[:,k]))
            residuals.append(model.stage(X[:,k],U[:,k],P[:,k]))
            if ng:
                path.append(model.path(X[:,k],U[:,k],P[:,k]))
                low.extend(bound_vector(bounds.path_lower,ng,-np.inf))
                high.extend(bound_vector(bounds.path_upper,ng,np.inf))
        residuals.append(model.terminal(X[:,N],P[:,N]))
        if ngN:
            path.append(model.terminal_path(X[:,N],P[:,N]))
            low.extend(bound_vector(bounds.terminal_lower,ngN,-np.inf))
            high.extend(bound_vector(bounds.terminal_upper,ngN,np.inf))
        r = ca.vertcat(*residuals)
        g = ca.vertcat(*eq,*path)
        n_eq = nx*(N+1)
        self.lbg = np.r_[np.zeros(n_eq), low]
        self.ubg = np.r_[np.zeros(n_eq), high]
        self.lbx = np.r_[np.tile(bound_vector(bounds.x_lower,nx,-np.inf),N+1),
                         np.tile(bound_vector(bounds.u_lower,nu,-np.inf),N)]
        self.ubx = np.r_[np.tile(bound_vector(bounds.x_upper,nx,np.inf),N+1),
                         np.tile(bound_vector(bounds.u_upper,nu,np.inf),N)]
        if np.any(self.lbx>self.ubx) or np.any(self.lbg>self.ubg):
            raise ValueError("Lower bounds exceed upper bounds")
        J = ca.jacobian(r,w)
        H = ca.mtimes(J.T,J)+options.hessian_regularization*ca.SX.eye(self.nw)
        objective = ca.dot(r,r)/2
        self.evaluate = ca.Function("evaluate",[w,theta],[objective,g])
        self.linearize = ca.Function("linearize",[w,theta],
            [ca.triu(H),ca.mtimes(J.T,r),ca.jacobian(g,w),g,objective])
        self.nlp = {"x": w,"p":theta,"f":objective,"g":g}

    def parameters(self, measured, parameters=None):
        m, N = self.model, self.options.N
        measured = np.asarray(measured,dtype=float).reshape(m.nx)
        if parameters is None:
            if m.np:
                raise ValueError("This model requires online parameters")
            parameters = np.zeros((0,N+1))
        p = np.asarray(parameters,dtype=float)
        if p.shape == (m.np,):
            p = np.repeat(p[:,None],N+1,axis=1)
        if p.shape != (m.np,N+1):
            raise ValueError(f"Parameters need shape ({m.np},) or ({m.np},{N+1})")
        if not np.isfinite(measured).all() or not np.isfinite(p).all():
            raise ValueError("States and parameters must be finite")
        if np.any(measured<self.lbx[:m.nx]) or np.any(measured>self.ubx[:m.nx]):
            raise ValueError("Measured state violates hard state bounds")
        return np.r_[measured,p.ravel(order="F")], p

    def pack(self,X,U):
        return np.r_[X.ravel(order="F"),U.ravel(order="F")]

    def unpack(self,w):
        m,N = self.model,self.options.N
        return (w[:self.n_state_variables].reshape((m.nx,N+1),order="F"),
                w[self.n_state_variables:].reshape((m.nu,N),order="F"))

    def rollout(self,measured,U,P):
        X = np.zeros((self.model.nx,self.options.N+1))
        X[:,0] = measured
        for k in range(self.options.N):
            X[:,k+1] = np.array(self.step(X[:,k],U[:,k],P[:,k])).ravel()
        return X
