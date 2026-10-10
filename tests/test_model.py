
import casadi as ca
import numpy as np
import pytest

from generalmpc.model import Model, Bounds, bound_vector


def make_valid_model():
    """Create a simple 2-state, 1-input system."""
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    rhs = ca.Function(
        "rhs",
        [x, u, p],
        [ca.vertcat(x[1], -x[0] + u[0])],
    )

    stage = ca.Function(
        "stage",
        [x, u, p],
        [ca.vertcat(x, u)],
    )

    terminal = ca.Function(
        "terminal",
        [x, p],
        [x],
    )

    return Model(
        nx=2,
        nu=1,
        np=0,
        rhs=rhs,
        stage=stage,
        terminal=terminal,
    )


def test_valid_model():
    model = make_valid_model()

    assert model.nx == 2
    assert model.nu == 1
    assert model.np == 0
    assert model.path is None
    assert model.terminal_path is None


@pytest.mark.parametrize(
    "nx, nu, np_",
    [
        (0, 1, 0),
        (2, 0, 0),
        (2, 1, -1),
    ],
)
def test_invalid_model_dimensions(nx, nu, np_):
    valid = make_valid_model()

    with pytest.raises(ValueError):
        Model(
            nx=nx,
            nu=nu,
            np=np_,
            rhs=valid.rhs,
            stage=valid.stage,
            terminal=valid.terminal,
        )


def test_wrong_rhs_input_dimensions():
    valid = make_valid_model()

    x = ca.SX.sym("x", 3)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    wrong_rhs = ca.Function(
        "wrong_rhs",
        [x, u, p],
        [x],
    )

    with pytest.raises(ValueError):
        Model(
            nx=2,
            nu=1,
            np=0,
            rhs=wrong_rhs,
            stage=valid.stage,
            terminal=valid.terminal,
        )


def test_wrong_rhs_output_dimensions():
    valid = make_valid_model()

    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    wrong_rhs = ca.Function(
        "wrong_rhs_output",
        [x, u, p],
        [x[0]],
    )

    with pytest.raises(ValueError):
        Model(
            nx=2,
            nu=1,
            np=0,
            rhs=wrong_rhs,
            stage=valid.stage,
            terminal=valid.terminal,
        )


def test_optional_path_constraints():
    valid = make_valid_model()

    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    path = ca.Function(
        "path",
        [x, u, p],
        [ca.vertcat(x[0] ** 2, u[0])],
    )

    model = Model(
        nx=2,
        nu=1,
        np=0,
        rhs=valid.rhs,
        stage=valid.stage,
        terminal=valid.terminal,
        path=path,
    )

    assert model.path.numel_out(0) == 2


def test_bound_vector_scalar():
    result = bound_vector(2.0, 3, -np.inf)

    np.testing.assert_array_equal(
        result, [2.0, 2.0, 2.0]
    )


def test_bound_vector_array():
    result = bound_vector([-1.0, 0.0, 1.0], 3, -np.inf)

    np.testing.assert_array_equal(
        result, [-1.0, 0.0, 1.0]
    )


def test_bound_vector_default():
    result = bound_vector(None, 3, -np.inf)

    assert np.all(np.isneginf(result))


def test_bound_vector_rejects_nan():
    with pytest.raises(ValueError):
        bound_vector([0.0, np.nan], 2, -np.inf)


def test_bound_vector_rejects_wrong_shape():
    with pytest.raises(ValueError):
        bound_vector([1.0, 2.0, 3.0], 2, -np.inf)
