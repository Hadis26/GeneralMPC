from dataclasses import dataclass


@dataclass(frozen=True)
class Options:
    N: int = 40
    dt: float = 0.025
    integrator_substeps: int = 2
    mode: str = "sqp"  # "sqp": converge with line search; "rti": one full step
    max_sqp_iterations: int = 30
    tolerance: float = 1e-6
    hessian_regularization: float = 1e-8
    qp_tolerance: float = 1e-8
    qp_max_iterations: int = 20000
    max_line_search_iterations: int = 20

    def __post_init__(self):
        if self.mode not in ("sqp", "rti"):
            raise ValueError("mode must be 'sqp' or 'rti'")
        if min(self.N, self.integrator_substeps, self.max_sqp_iterations,
               self.qp_max_iterations, self.max_line_search_iterations) < 1:
            raise ValueError("Iteration counts and horizon must be positive")
        if min(self.dt, self.tolerance, self.qp_tolerance,
               self.hessian_regularization) <= 0:
            raise ValueError("Time step, tolerances and regularization must be positive")
