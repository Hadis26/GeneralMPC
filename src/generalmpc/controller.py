from dataclasses import dataclass,field
import time
import numpy as np
import scipy.sparse as sp
from .model import Bounds
from .options import Options
from .transcription import Transcription
from .qp import QPSolver
from .globalization import violations,line_search,complementarity


@dataclass
class Result:
    control: np.ndarray
    X: np.ndarray
    U: np.ndarray
    status: str
    iterations: int
    objective: float
    constraint_violation: float
    stationarity: float
    complementarity: float
    solve_time: float
    history: list = field(default_factory=list)


class Controller:
    def __init__(self,model,bounds=None,options=None):
        self.options = options or Options()
        self.problem = Transcription(model,bounds or Bounds(),self.options)
        self.qp = QPSolver(self.options)
        self.previous_U = None

    def reset(self):
        self.previous_U = None
        self.qp = QPSolver(self.options)

    def solve(self,measured,parameters=None,initial_controls=None):
        start = time.perf_counter()
        p,o,m = self.problem,self.options,self.problem.model
        theta,P = p.parameters(measured,parameters)
        if initial_controls is not None:
            U = np.asarray(initial_controls,dtype=float)
            if U.shape != (m.nu,o.N) or not np.isfinite(U).all():
                raise ValueError("initial_controls has incorrect shape or nonfinite values")
        elif self.previous_U is not None:
            U = np.column_stack([self.previous_U[:,1:],self.previous_U[:,-1]])
        else:
            U = np.zeros((m.nu,o.N))
        U = np.clip(U,p.lbx[p.n_state_variables:].reshape((m.nu,o.N),order="F"),
                    p.ubx[p.n_state_variables:].reshape((m.nu,o.N),order="F"))
        # Re-integrate shifted controls from the measured state for a feasible dynamics guess.
        w = p.pack(p.rollout(theta[:m.nx],U,P),U)
        history,status,stationarity = [],"max_iterations",float("inf")
        for iteration in range(1,(1 if o.mode=="rti" else o.max_sqp_iterations)+1):
            H,gradient,A,g,f = p.linearize(w,theta)
            H,A = H.sparse().tocsc(),A.sparse().tocsc()
            gradient,g = np.array(gradient).ravel(),np.array(g).ravel()
            Aq = sp.vstack([A,sp.eye(p.nw,format="csc")],format="csc")
            low,high = np.r_[p.lbg-g,p.lbx-w],np.r_[p.ubg-g,p.ubx-w]            
            d,dual,qp_info = self.qp.solve(H,gradient,Aq,low,high)
            stationarity = float(np.max(np.abs(gradient+Aq.T@dual)))
            comp = complementarity(np.r_[g,w],np.r_[p.lbg,p.lbx],np.r_[p.ubg,p.ubx],dual)
            violation = max(float(np.max(violations(g,p.lbg,p.ubg))),
                            float(np.max(violations(w,p.lbx,p.ubx))))
            if o.mode=="sqp" and max(stationarity,violation,comp) <= o.tolerance:
                status = "converged"
                history.append({"iteration":iteration,"alpha":0.,"objective":float(f),
                                "stationarity":stationarity,"complementarity":comp,"violation":violation,"qp":qp_info})
                break
            if o.mode=="rti":
                w,alpha,status = w+d,1.,"rti_step"
            else:
                w,alpha = line_search(p.evaluate,w,theta,d,float(f),gradient,g,
                                     p.lbg,p.ubg,p.lbx,p.ubx,dual,o)
            # Report stationarity at the UPDATED iterate using the current QP multipliers.
            _,grad_new,Anew,gnew,fnew = p.linearize(w,theta)
            stationarity = float(np.max(np.abs(np.array(grad_new).ravel()
                                  + Anew.sparse().T@dual[:len(p.lbg)]+dual[len(p.lbg):])))
            violation = max(float(np.max(violations(np.array(gnew).ravel(),p.lbg,p.ubg))),
                            float(np.max(violations(w,p.lbx,p.ubx))))
            comp = complementarity(np.r_[np.array(gnew).ravel(),w],np.r_[p.lbg,p.lbx],np.r_[p.ubg,p.ubx],dual)
            history.append({"iteration":iteration,"alpha":alpha,"objective":float(fnew),
                            "stationarity":stationarity,"complementarity":comp,"violation":violation,"qp":qp_info})

            print(
                f"  objective = {float(fnew):.6e}"
                f" | stationarity = {stationarity:.3e}"
                f" | violation = {violation:.3e}"
                f" | comp = {comp:.3e}"
                f" | alpha = {alpha:.3e}"
            )
            
            if o.mode=="sqp" and max(stationarity,violation,comp) <= o.tolerance:
                status = "converged"
                break
        X,U = p.unpack(w)
        objective,g = p.evaluate(w,theta)
        violation = max(float(np.max(violations(np.array(g).ravel(),p.lbg,p.ubg))),
                        float(np.max(violations(w,p.lbx,p.ubx))))
        if not np.isfinite(w).all():
            raise RuntimeError("Nonfinite MPC solution")
        # Cache only completed solves; failures above stop instead of applying an unknown action.
        self.previous_U = U.copy()
        return Result(U[:,0].copy(),X,U,status,iteration,float(objective),violation,
                      stationarity,comp,time.perf_counter()-start,history)
