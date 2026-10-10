import shutil

import numpy as np
import pytest

from generalmpc.codegen import (
    export_kernels,
    compile_kernels,
    load_kernels,
)

# Reuse the already-tested linear MPC fixture.
from test_controller import make_controller


@pytest.mark.skipif(
    shutil.which("cc") is None,
    reason="C compiler not available",
)
def test_generated_kernels_match_original(tmp_path):
    original = make_controller()
    generated = make_controller()

    # Generate and compile the application-specific C library.
    source = export_kernels(generated, tmp_path)
    assert source.exists()

    library = compile_kernels(source)
    assert library.exists()

    load_kernels(generated, library)

    # Compare numerical evaluations.
    measured = np.array([1.0])

    theta, P = original.problem.parameters(measured)

    U = np.zeros((1, original.options.N))
    X = original.problem.rollout(measured, U, P)
    w = original.problem.pack(X, U)

    for attribute in ("evaluate", "linearize"):
        f_python = getattr(original.problem, attribute)
        f_generated = getattr(generated.problem, attribute)

        expected = f_python(w, theta)
        actual = f_generated(w, theta)

        for reference, result in zip(expected, actual):
            np.testing.assert_allclose(
                np.asarray(result),
                np.asarray(reference),
                rtol=1e-10,
                atol=1e-10,
            )

    # Compare complete MPC solutions.
    result_original = original.solve(measured)
    result_generated = generated.solve(measured)

    assert result_original.status == "converged"
    assert result_generated.status == "converged"

    np.testing.assert_allclose(
        result_generated.control,
        result_original.control,
        atol=1e-8,
    )
