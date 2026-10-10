
import casadi as ca
import numpy as np

from generalmpc.integrators import rk4


def make_linear_system():
    x = ca.SX.sym("x", 1)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    rhs = ca.Function(
        "linear_rhs",
        [x, u, p],
        [-2.0 * x + 1.5 * u],
    )
    return rhs


def test_rk4_against_exact_solution():
    rhs = make_linear_system()
    dt = 0.1

    step, _ = rk4(rhs, 1, 1, 0, dt, 2)

    x0 = 1.2
    u0 = 0.4

    numerical = float(step([x0], [u0], []))

    a, b = -2.0, 1.5
    exact = (
        np.exp(a * dt) * x0
        + b / a * (np.exp(a * dt) - 1.0) * u0
    )

    np.testing.assert_allclose(
        numerical, exact, rtol=1e-6, atol=1e-8
    )


def test_rk4_equilibrium():
    rhs = make_linear_system()

    step, _ = rk4(rhs, 1, 1, 0, 0.1, 2)

    # Equilibrium: -2*x + 1.5*u = 0
    result = float(step([0.75], [1.0], []))

    np.testing.assert_allclose(result, 0.75, atol=1e-12)


def test_rk4_state_jacobian():
    rhs = make_linear_system()
    dt = 0.1

    step, step_jac = rk4(rhs, 1, 1, 0, dt, 2)

    x0 = np.array([0.8])
    u0 = np.array([0.3])
    eps = 1e-6

    A_ad, _ = step_jac(x0, u0, [])

    A_fd = (
        np.array(step(x0 + eps, u0, []))
        - np.array(step(x0 - eps, u0, []))
    ) / (2 * eps)

    np.testing.assert_allclose(
        np.array(A_ad), A_fd, rtol=1e-6, atol=1e-8
    )


def test_rk4_control_jacobian():
    rhs = make_linear_system()

    step, step_jac = rk4(rhs, 1, 1, 0, 0.1, 2)

    x0 = np.array([0.8])
    u0 = np.array([0.3])
    eps = 1e-6

    _, B_ad = step_jac(x0, u0, [])

    B_fd = (
        np.array(step(x0, u0 + eps, []))
        - np.array(step(x0, u0 - eps, []))
    ) / (2 * eps)

    np.testing.assert_allclose(
        np.array(B_ad), B_fd, rtol=1e-6, atol=1e-8
    )
