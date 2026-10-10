
import casadi as ca
import numpy as np
import pytest

from generalmpc.model import Model, Bounds
from generalmpc.options import Options
from generalmpc.transcription import Transcription


def make_problem():
    """Two-state nonlinear system with one control."""
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 1)

    rhs = ca.Function(
        "test_rhs",
        [x, u, p],
        [ca.vertcat(
            x[1],
            -ca.sin(x[0]) - 0.2 * x[1] + u[0] + p[0]
        )],
    )

    stage = ca.Function(
        "test_stage",
        [x, u, p],
        [ca.vertcat(x[0], x[1], 0.5 * u[0])],
    )

    terminal = ca.Function(
        "test_terminal",
        [x, p],
        [ca.vertcat(2.0 * x[0], x[1])],
    )

    model = Model(
        nx=2,
        nu=1,
        np=1,
        rhs=rhs,
        stage=stage,
        terminal=terminal,
    )

    bounds = Bounds(
        x_lower=[-3.0, -4.0],
        x_upper=[3.0, 4.0],
        u_lower=[-2.0],
        u_upper=[2.0],
    )

    options = Options(
        N=5,
        dt=0.05,
        integrator_substeps=2,
    )

    return Transcription(model, bounds, options)


def make_test_point(problem):
    measured = np.array([0.4, -0.2])
    parameters = np.full((1, 6), 0.1)

    theta, P = problem.parameters(measured, parameters)

    U = np.array([[0.1, 0.2, -0.1, 0.05, 0.0]])
    X = problem.rollout(measured, U, P)

    w = problem.pack(X, U)
    return w, theta, X, U


def test_variable_dimensions():
    problem = make_problem()

    # nx*(N+1) + nu*N = 2*6 + 1*5
    assert problem.nw == 17
    assert problem.n_state_variables == 12
    assert len(problem.lbg) == 12


def test_pack_unpack():
    problem = make_problem()
    w, _, X, U = make_test_point(problem)

    X_recovered, U_recovered = problem.unpack(w)

    np.testing.assert_allclose(X_recovered, X)
    np.testing.assert_allclose(U_recovered, U)


def test_rollout_satisfies_dynamics():
    problem = make_problem()
    w, theta, _, _ = make_test_point(problem)

    _, g = problem.evaluate(w, theta)

    # All constraints are dynamics equalities in this example.
    np.testing.assert_allclose(
        np.array(g).ravel(),
        np.zeros(len(problem.lbg)),
        atol=1e-11,
    )


def test_objective_value():
    problem = make_problem()
    w, theta, X, U = make_test_point(problem)

    objective, _ = problem.evaluate(w, theta)

    # Stage residual: [x1, x2, 0.5*u]
    stage_cost = 0.5 * sum(
        X[0, k] ** 2
        + X[1, k] ** 2
        + (0.5 * U[0, k]) ** 2
        for k in range(5)
    )

    # Terminal residual: [2*x1, x2]
    terminal_cost = 0.5 * (
        (2 * X[0, 5]) ** 2 + X[1, 5] ** 2
    )

    np.testing.assert_allclose(
        float(objective),
        stage_cost + terminal_cost,
        rtol=1e-10,
    )


def test_objective_gradient():
    problem = make_problem()
    w, theta, _, _ = make_test_point(problem)

    _, gradient, _, _, _ = problem.linearize(w, theta)
    gradient = np.array(gradient).ravel()

    eps = 1e-6
    gradient_fd = np.zeros_like(w)

    for i in range(len(w)):
        perturb = np.zeros_like(w)
        perturb[i] = eps

        f_plus, _ = problem.evaluate(w + perturb, theta)
        f_minus, _ = problem.evaluate(w - perturb, theta)

        gradient_fd[i] = (
            float(f_plus) - float(f_minus)
        ) / (2 * eps)

    np.testing.assert_allclose(
        gradient,
        gradient_fd,
        rtol=1e-5,
        atol=1e-7,
    )


def test_constraint_jacobian():
    problem = make_problem()
    w, theta, _, _ = make_test_point(problem)

    _, _, A, _, _ = problem.linearize(w, theta)
    A = A.full()

    eps = 1e-6
    A_fd = np.zeros_like(A)

    for i in range(len(w)):
        perturb = np.zeros_like(w)
        perturb[i] = eps

        _, g_plus = problem.evaluate(w + perturb, theta)
        _, g_minus = problem.evaluate(w - perturb, theta)

        A_fd[:, i] = (
            np.array(g_plus).ravel()
            - np.array(g_minus).ravel()
        ) / (2 * eps)

    np.testing.assert_allclose(
        A,
        A_fd,
        rtol=1e-5,
        atol=1e-7,
    )


def test_gauss_newton_hessian():
    problem = make_problem()
    w, theta, _, _ = make_test_point(problem)

    H, _, _, _, _ = problem.linearize(w, theta)

    # linearize returns only the upper-triangular Hessian.
    H_upper = H.full()
    H_full = (
        H_upper + H_upper.T
        - np.diag(np.diag(H_upper))
    )

    x = ca.SX.sym("x", problem.nw)
    t = ca.SX.sym("t", theta.size)

    # Reconstruct the residual used by this example.
    X = ca.reshape(x[:12], 2, 6)
    U = ca.reshape(x[12:], 1, 5)

    residuals = []
    for k in range(5):
        residuals.append(
            ca.vertcat(X[0, k], X[1, k], 0.5 * U[0, k])
        )
    residuals.append(
        ca.vertcat(2 * X[0, 5], X[1, 5])
    )

    r = ca.vertcat(*residuals)
    J = ca.jacobian(r, x)

    expected_function = ca.Function(
        "expected_hessian",
        [x],
        [J.T @ J + 1e-8 * ca.SX.eye(problem.nw)],
    )

    expected = expected_function(w).full()

    np.testing.assert_allclose(
        H_full,
        expected,
        atol=1e-10,
    )


def test_bounds():
    problem = make_problem()

    assert len(problem.lbx) == 17
    assert len(problem.ubx) == 17

    # First state bounds.
    np.testing.assert_array_equal(
        problem.lbx[:2], [-3.0, -4.0]
    )

    # Control bounds.
    np.testing.assert_array_equal(
        problem.lbx[12:], [-2.0] * 5
    )
    np.testing.assert_array_equal(
        problem.ubx[12:], [2.0] * 5
    )


def test_parameter_validation():
    problem = make_problem()

    with pytest.raises(ValueError):
        problem.parameters(
            measured=[0.0, 0.0],
            parameters=np.zeros((2, 6)),
        )

    with pytest.raises(ValueError):
        problem.parameters(
            measured=[5.0, 0.0],
            parameters=np.zeros((1, 6)),
        )
