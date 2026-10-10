import casadi as ca
import numpy as np
import pytest

from generalmpc import Controller, Model, Bounds, Options


def make_controller(mode="sqp", input_limit=2.0):
    """Scalar MPC with an analytically known solution."""
    x = ca.SX.sym("x", 1)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    rhs = ca.Function(
        "controller_rhs",
        [x, u, p],
        [u],
    )

    stage = ca.Function(
        "controller_stage",
        [x, u, p],
        [ca.sqrt(0.1) * u],
    )

    terminal = ca.Function(
        "controller_terminal",
        [x, p],
        [x],
    )

    model = Model(
        nx=1,
        nu=1,
        np=0,
        rhs=rhs,
        stage=stage,
        terminal=terminal,
    )

    bounds = Bounds(
        u_lower=[-input_limit],
        u_upper=[input_limit],
    )

    options = Options(
        N=1,
        dt=1.0,
        mode=mode,
        max_sqp_iterations=20,
        tolerance=1e-5,
        qp_tolerance=1e-9,
    )

    return Controller(model, bounds, options)


def test_sqp_analytical_solution():
    controller = make_controller()

    result = controller.solve(measured=np.array([1.0]))

    assert result.status == "converged"

    np.testing.assert_allclose(
        result.control,
        [-10.0 / 11.0],
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result.X[:, -1],
        [1.0 / 11.0],
        atol=1e-5,
    )

    assert result.constraint_violation < 1e-5
    assert result.stationarity < 1e-5
    assert result.complementarity < 1e-5


def test_sqp_input_constraint():
    """
    With u >= -0.5, the unconstrained optimum
    u=-10/11 is infeasible.

    The constrained optimum must be u=-0.5.
    """
    controller = make_controller(input_limit=0.5)

    result = controller.solve(measured=np.array([1.0]))

    assert result.status == "converged"

    np.testing.assert_allclose(
        result.control,
        [-0.5],
        atol=1e-5,
    )

    np.testing.assert_allclose(
        result.X[:, -1],
        [0.5],
        atol=1e-5,
    )

    assert result.constraint_violation < 1e-5


def test_rti_single_iteration():
    controller = make_controller(mode="rti")

    result = controller.solve(measured=np.array([1.0]))

    assert result.status == "rti_step"
    assert result.iterations == 1

    assert np.isfinite(result.objective)
    assert np.isfinite(result.control).all()
    assert np.isfinite(result.X).all()
    assert np.isfinite(result.U).all()


def test_invalid_initial_controls():
    controller = make_controller()

    with pytest.raises(ValueError):
        controller.solve(
            measured=np.array([1.0]),
            initial_controls=np.zeros((2, 2)),
        )


def test_warm_start_and_reset():
    controller = make_controller()

    first = controller.solve(measured=np.array([1.0]))

    assert controller.previous_U is not None

    second = controller.solve(measured=np.array([0.5]))

    assert second.status == "converged"
    assert np.isfinite(second.control).all()

    controller.reset()

    assert controller.previous_U is None
    assert controller.qp.solver is None
