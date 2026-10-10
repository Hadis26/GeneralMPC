
import numpy as np
import pytest
from types import SimpleNamespace

from generalmpc.globalization import (
    violations,
    complementarity,
    line_search,
)


def test_violations():
    values = np.array([-2.0, 0.5, 3.0])
    lower = np.array([-1.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 2.0])

    np.testing.assert_allclose(
        violations(values, lower, upper),
        [1.0, 0.0, 1.0],
    )


def test_violations_with_infinite_bounds():
    values = np.array([5.0, -3.0])
    lower = np.array([-np.inf, -2.0])
    upper = np.array([4.0, np.inf])

    np.testing.assert_allclose(
        violations(values, lower, upper),
        [1.0, 1.0],
    )


def test_complementarity_inactive_constraints():
    values = np.array([0.0, 0.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    dual = np.zeros(2)

    assert complementarity(
        values, lower, upper, dual
    ) == 0.0


def test_complementarity_active_bounds():
    # Upper bound active: positive OSQP multiplier.
    # Lower bound active: negative OSQP multiplier.
    values = np.array([1.0, -1.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    dual = np.array([2.0, -3.0])

    assert complementarity(
        values, lower, upper, dual
    ) == 0.0


def test_complementarity_nonzero():
    values = np.array([0.5])
    lower = np.array([0.0])
    upper = np.array([1.0])
    dual = np.array([2.0])

    # Positive multiplier on upper bound,
    # but the upper bound is inactive.
    # Complementarity error = 2*(1-0.5) = 1.
    assert complementarity(
        values, lower, upper, dual
    ) == pytest.approx(1.0)


def test_line_search_accepts_full_step():
    # Minimize f(w) = 0.5*w^2, starting at w=2.
    # Newton step d=-2 reaches the optimum.
    def evaluate(w, theta):
        return 0.5 * w[0] ** 2, np.array([])

    options = SimpleNamespace(
        max_line_search_iterations=20
    )

    w = np.array([2.0])
    d = np.array([-2.0])
    gradient = np.array([2.0])

    candidate, alpha = line_search(
        evaluate=evaluate,
        w=w,
        theta=None,
        d=d,
        objective=2.0,
        gradient=gradient,
        g=np.array([]),
        lbg=np.array([]),
        ubg=np.array([]),
        lbx=np.array([-np.inf]),
        ubx=np.array([np.inf]),
        dual=np.array([]),
        options=options,
    )

    assert alpha == pytest.approx(1.0)
    np.testing.assert_allclose(candidate, [0.0])


def test_line_search_backtracking():
    # A full step increases the objective,
    # but a smaller step reduces it.
    def evaluate(w, theta):
        return 0.5 * w[0] ** 2, np.array([])

    options = SimpleNamespace(
        max_line_search_iterations=20
    )

    w = np.array([1.0])
    d = np.array([-3.0])
    gradient = np.array([1.0])

    candidate, alpha = line_search(
        evaluate=evaluate,
        w=w,
        theta=None,
        d=d,
        objective=0.5,
        gradient=gradient,
        g=np.array([]),
        lbg=np.array([]),
        ubg=np.array([]),
        lbx=np.array([-np.inf]),
        ubx=np.array([np.inf]),
        dual=np.array([]),
        options=options,
    )

    assert 0.0 < alpha < 1.0
    assert float(candidate[0] ** 2) < 1.0


def test_line_search_rejects_non_descent():
    def evaluate(w, theta):
        return 0.5 * w[0] ** 2, np.array([])

    options = SimpleNamespace(
        max_line_search_iterations=20
    )

    with pytest.raises(
        RuntimeError, match="not a descent direction"
    ):
        line_search(
            evaluate=evaluate,
            w=np.array([1.0]),
            theta=None,
            d=np.array([1.0]),
            objective=0.5,
            gradient=np.array([1.0]),
            g=np.array([]),
            lbg=np.array([]),
            ubg=np.array([]),
            lbx=np.array([-np.inf]),
            ubx=np.array([np.inf]),
            dual=np.array([]),
            options=options,
        )
