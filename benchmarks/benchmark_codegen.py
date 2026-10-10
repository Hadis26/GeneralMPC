import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from cartpole_stabilization import make_controller

from generalmpc.codegen import (
    export_kernels,
    compile_kernels,
    load_kernels,
)


def benchmark(controller, steps=100):
    """Time closed-loop MPC solves with a given backend."""
    dt = controller.options.dt

    x = np.array([0.0, 0.2, 0.0, 0.0])

    times = []
    iterations = []
    violations = []
    controls = []
    states = [x.copy()]

    for k in range(steps):
        result = controller.solve(x)

        if result.status != "converged":
            raise RuntimeError(
                f"Solver failed at step {k}: {result.status}"
            )

        times.append(result.solve_time)
        iterations.append(result.iterations)
        violations.append(result.constraint_violation)
        controls.append(result.control.copy())

        x = np.asarray(
            controller.problem.step(
                x,
                result.control,
                [],
            )
        ).ravel()

        states.append(x.copy())

    return {
        "times": np.asarray(times),
        "iterations": np.asarray(iterations),
        "violations": np.asarray(violations),
        "controls": np.asarray(controls),
        "states": np.asarray(states),
    }


def summarize(result):
    times = result["times"] * 1000

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": float(np.max(times)),
        "mean_iterations": float(
            np.mean(result["iterations"])
        ),
        "max_violation": float(
            np.max(result["violations"])
        ),
        "final_state_norm": float(
            np.linalg.norm(result["states"][-1])
        ),
    }


def main():
    output = ROOT / "results" / "benchmarks"
    output.mkdir(parents=True, exist_ok=True)

    # Baseline: regular CasADi functions.
    print("Benchmarking standard CasADi...")
    controller_python = make_controller(mode="sqp")
    baseline = benchmark(controller_python)

    # Generate application-specific C kernels.
    print("Generating and compiling C kernels...")
    controller_c = make_controller(mode="sqp")

    source = export_kernels(
        controller_c,
        output / "generated",
    )

    library = compile_kernels(source)
    load_kernels(controller_c, library)

    print("Benchmarking generated C kernels...")
    compiled = benchmark(controller_c)

    summary = {
        "casadi": summarize(baseline),
        "generated_c": summarize(compiled),
    }

    # Validate numerical equivalence.
    max_control_difference = float(
        np.max(np.abs(
            baseline["controls"] - compiled["controls"]
        ))
    )

    max_state_difference = float(
        np.max(np.abs(
            baseline["states"] - compiled["states"]
        ))
    )

    summary["comparison"] = {
        "max_control_difference": max_control_difference,
        "max_state_difference": max_state_difference,
        "mean_speedup": (
            summary["casadi"]["mean_ms"]
            / summary["generated_c"]["mean_ms"]
        ),
    }

    (output / "codegen_benchmark.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nResults:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
