
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from cartpole_stabilization import simulate


def main():
    output = ROOT / "results" / "benchmarks"
    output.mkdir(parents=True, exist_ok=True)

    results = {}

    for mode in ("sqp", "rti"):
        print(f"Running {mode.upper()}...")

        dt, states, controls, times, iterations, violations = (
            simulate(mode=mode)
        )

        metrics = {
            "mean_time_ms": float(np.mean(times) * 1000),
            "median_time_ms": float(np.median(times) * 1000),
            "p95_time_ms": float(np.percentile(times, 95) * 1000),
            "max_time_ms": float(np.max(times) * 1000),
            "mean_iterations": float(np.mean(iterations)),
            "max_violation": float(np.max(violations)),
            "deadline_fraction": float(np.mean(times <= dt)),
            "final_state_norm": float(
                np.linalg.norm(states[:, -1])
            ),
            "control_effort": float(
                dt * np.sum(controls**2)
            ),
        }

        results[mode] = {
            "metrics": metrics,
            "states": states,
            "controls": controls,
            "times": times,
        }

        print(json.dumps(metrics, indent=2))

    # Persist numerical metrics.
    summary = {
        mode: values["metrics"]
        for mode, values in results.items()
    }

    (output / "sqp_vs_rti.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Compare solve times.
    fig, ax = plt.subplots(figsize=(9, 5))

    for mode, values in results.items():
        ax.plot(
            np.arange(len(values["times"])),
            values["times"] * 1000,
            label=mode.upper(),
        )

    ax.axhline(
        50.0,
        linestyle="--",
        label="50 ms deadline",
    )

    ax.set_xlabel("Control step")
    ax.set_ylabel("Solve time [ms]")
    ax.set_title("Cart-Pole MPC: SQP vs. RTI")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        output / "sqp_vs_rti_timing.png",
        dpi=200,
    )

    # Compare pendulum angles.
    fig, ax = plt.subplots(figsize=(9, 5))

    for mode, values in results.items():
        t = np.arange(values["states"].shape[1]) * 0.05
        ax.plot(
            t,
            values["states"][1],
            label=mode.upper(),
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pendulum angle [rad]")
    ax.set_title("Cart-Pole Stabilization: SQP vs. RTI")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        output / "sqp_vs_rti_tracking.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":
    main()
