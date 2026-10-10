
import numpy as np
import scipy.sparse as sp
import pytest

from generalmpc.options import Options
from generalmpc.qp import QPSolver


def make_solver():
    return QPSolver(
        Options(
            qp_tolerance=1e-9,
            qp_max_iterations=20000,
        )
    )


def test_unconstrained_qp():
    """
    Minimize:
        0.5*x.T@H@x + q.T@x

    H = diag(2, 4)
    q = [-2, -8]

    Analytical optimum: x = [1, 2].
    """
    solver = make_solver()

    H = sp.diags([2.0, 4.0], format="csc")
    q = np.array([-2.0, -8.0])

    A = sp.csc_matrix((0, 2))
    lower = np.array([])
    upper = np.array([])

    x, dual, info = solver.solve(
        H, q, A, lower, upper
    )

    np.testing.assert_allclose(
        x, [1.0, 2.0], atol=1e-7
    )

    assert info["status"] in (
        "solved", "solved inaccurate"
    )
    assert len(dual) == 0


def test_equality_constrained_qp():
    """
    Minimize:
        0.5*(x1**2 + x2**2)

    Subject to:
        x1 + x2 = 2

    Analytical optimum: x = [1, 1].
    """
    solver = make_solver()

    H = sp.eye(2, format="csc")
    q = np.zeros(2)

    A = sp.csc_matrix([[1.0, 1.0]])
    lower = np.array([2.0])
    upper = np.array([2.0])

    x, dual, _ = solver.solve(
        H, q, A, lower, upper
    )

    np.testing.assert_allclose(
        x, [1.0, 1.0], atol=1e-7
    )

    np.testing.assert_allclose(
        A @ x, [2.0], atol=1e-7
    )

    # Stationarity: H*x + q + A.T*y = 0
    stationarity = H @ x + q + A.T @ dual

    np.testing.assert_allclose(
        stationarity, 0.0, atol=1e-7
    )


def test_inequality_constrained_qp():
    """
    Minimize:
        0.5*(x1**2 + x2**2)

    Subject to:
        x1 >= 1
        x2 >= 2

    Analytical optimum: x = [1, 2].
    """
    solver = make_solver()

    H = sp.eye(2, format="csc")
    q = np.zeros(2)

    A = sp.eye(2, format="csc")
    lower = np.array([1.0, 2.0])
    upper = np.array([np.inf, np.inf])

    x, dual, _ = solver.solve(
        H, q, A, lower, upper
    )

    np.testing.assert_allclose(
        x, [1.0, 2.0], atol=1e-7
    )

    assert np.all(x >= lower - 1e-7)
    assert np.all(dual <= 1e-7)


def test_qp_update_same_sparsity():
    """
    Check that changing q does not recreate
    the OSQP solver when the sparsity stays
    unchanged.
    """
    solver = make_solver()

    H = sp.eye(2, format="csc")
    A = sp.eye(2, format="csc")

    lower = np.array([-10.0, -10.0])
    upper = np.array([10.0, 10.0])

    x1, _, _ = solver.solve(
        H,
        np.array([-1.0, -2.0]),
        A,
        lower,
        upper,
    )

    original_solver = solver.solver

    x2, _, _ = solver.solve(
        H,
        np.array([-3.0, -4.0]),
        A,
        lower,
        upper,
    )

    assert solver.solver is original_solver

    np.testing.assert_allclose(
        x1, [1.0, 2.0], atol=1e-7
    )

    np.testing.assert_allclose(
        x2, [3.0, 4.0], atol=1e-7
    )


def test_infeasible_qp():
    """
    Contradictory constraints:
        x >= 2
        x <= 1
    """
    solver = make_solver()

    H = sp.eye(1, format="csc")
    q = np.zeros(1)

    A = sp.csc_matrix([[1.0], [1.0]])
    lower = np.array([2.0, -np.inf])
    upper = np.array([np.inf, 1.0])

    with pytest.raises(RuntimeError, match="QP failed"):
        solver.solve(H, q, A, lower, upper)
