"""Application contract. Every function uses CasADi-compatible expressions.

Model.rhs(x,u,p): continuous dynamics, nx outputs.
Model.stage(x,u,p): weighted least-squares residual h, arbitrary outputs.
Model.terminal(x,p): weighted terminal residual hN, arbitrary outputs.
Model.path(x,u,p): optional nonlinear path constraints, ng outputs.
Model.terminal_path(x,p): optional terminal constraints, ngN outputs.

Weights belong in the residuals: sqrt(W) @ (h-reference).
"""
from dataclasses import dataclass
import casadi as ca
import numpy as np


@dataclass(frozen=True)
class Model:
    nx: int
    nu: int
    np: int
    rhs: ca.Function
    stage: ca.Function
    terminal: ca.Function
    path: ca.Function | None = None
    terminal_path: ca.Function | None = None

    def __post_init__(self):
        if self.nx < 1 or self.nu < 1 or self.np < 0:
            raise ValueError("Invalid model dimensions")
        for name, function, inputs in (
            ("rhs", self.rhs, [self.nx, self.nu, self.np]),
            ("stage", self.stage, [self.nx, self.nu, self.np]),
            ("terminal", self.terminal, [self.nx, self.np]),
            ("path", self.path, [self.nx, self.nu, self.np]),
            ("terminal_path", self.terminal_path, [self.nx, self.np]),
        ):
            if function is None:
                continue
            if function.n_in() != len(inputs) or function.n_out() != 1:
                raise ValueError(f"{name} has incorrect input/output count")
            if any(function.numel_in(i) != n for i,n in enumerate(inputs)):
                raise ValueError(f"{name} has incorrect input dimensions")
            if function.size2_out(0) != 1:
                raise ValueError(f"{name} must return a column vector")
        if self.rhs.numel_out(0) != self.nx:
            raise ValueError("rhs must return nx derivatives")


@dataclass
class Bounds:
    x_lower: object = None
    x_upper: object = None
    u_lower: object = None
    u_upper: object = None
    path_lower: object = None
    path_upper: object = None
    terminal_lower: object = None
    terminal_upper: object = None


def bound_vector(value, size, default):
    if value is None:
        return np.full(size, default)
    result = np.broadcast_to(np.asarray(value, dtype=float), (size,)).copy()
    if np.isnan(result).any():
        raise ValueError("Bounds cannot contain NaN")
    return result
