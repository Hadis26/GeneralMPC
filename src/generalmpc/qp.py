"""Native OSQP backend, retaining sparse matrix structure between SQP steps."""
import numpy as np
import scipy.sparse as sp
import osqp


class QPSolver:
    def __init__(self,options):
        self.options = options
        self.solver = None
        self.pattern = None

    def solve(self,H,gradient,A,lower,upper):
        H,A = H.tocsc(),A.tocsc()
        H.sort_indices(); A.sort_indices()
        pattern = (H.indices.tobytes(),H.indptr.tobytes(),A.indices.tobytes(),A.indptr.tobytes())
        if self.solver is None or pattern != self.pattern:
            self.solver = osqp.OSQP()
            self.solver.setup(P=H,q=gradient,A=A,l=lower,u=upper,
                verbose=False,eps_abs=self.options.qp_tolerance,
                eps_rel=self.options.qp_tolerance,max_iter=self.options.qp_max_iterations,
                polishing=True,warm_starting=True)
            self.pattern = pattern
        else:
            self.solver.update(Px=H.data,q=gradient,Ax=A.data,l=lower,u=upper)
        result = self.solver.solve(raise_error=False)
        if (
            result.info.status_val not in (1, 2)
            or result.x is None
        ):
            raise RuntimeError(
                "QP failed: "
                f"{result.info.status}. "
                f"iterations={result.info.iter}, "
                f"prim_res={result.info.prim_res:.3e}, "
                f"dual_res={result.info.dual_res:.3e}"
            )
        if not np.isfinite(result.x).all() or not np.isfinite(result.y).all():
            raise RuntimeError("QP returned nonfinite values")
        if A.shape[0] == 0:
            error = 0.0
        else:
            constraint_values = A @ result.x
            error = float(
                np.max(
                    np.maximum(
                        lower - constraint_values,
                        constraint_values - upper,
                    )
                )
            )
        if error > max(1e-6,100*self.options.qp_tolerance):
            raise RuntimeError(f"QP solution violates constraints by {error:.3g}")

        
        return (
            result.x,
            result.y,
            {
                "status": result.info.status,
                "iterations": result.info.iter,
                "constraint_violation": max(
                    0.0,
                    error,
                ),
                "primal_residual": float(
                    result.info.prim_res
                ),
                "dual_residual": float(
                    result.info.dual_res
                ),
            },
        )
