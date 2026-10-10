from pathlib import Path
import json

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from generalmpc import Controller, Model, Bounds, Options


def make_controller(mode="sqp"):
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    M = 1.0
    m = 0.2
    l = 0.5
    g = 9.81

    pos, theta, vel, omega = ca.vertsplit(x)
    force = u[0]

    s = ca.sin(theta)
    c = ca.cos(theta)
    denominator = M + m * s**2

    acceleration = (
        force + m*l*omega**2*s - m*g*s*c
    ) / denominator

    angular_acceleration = (
        (M + m)*g*s
        - force*c
        - m*l*omega**2*s*c
    ) / (l*denominator)

    rhs = ca.Function(
        "cartpole_rhs",
        [x, u, p],
        [ca.vertcat(
            vel,
            omega,
            acceleration,
            angular_acceleration,
        )],
    )

    Q = ca.diag(ca.DM([5.0, 100.0, 1.0, 5.0]))
    R = 0.1
    QN = ca.diag(ca.DM([20.0, 200.0, 5.0, 20.0]))

    stage = ca.Function(
        "cartpole_stage",
        [x, u, p],
        [ca.vertcat(
            ca.mtimes(ca.sqrt(Q), x),
            ca.sqrt(R) * u,
        )],
    )

    terminal = ca.Function(
        "cartpole_terminal",
        [x, p],
        [ca.mtimes(ca.sqrt(QN), x)],
    )

    model = Model(
        nx=4,
        nu=1,
        np=0,
        rhs=rhs,
        stage=stage,
        terminal=terminal,
    )

    bounds = Bounds(
        x_lower=[-2.0, -np.inf, -np.inf, -np.inf],
        x_upper=[2.0, np.inf, np.inf, np.inf],
        u_lower=[-10.0],
        u_upper=[10.0],
    )

    options = Options(
        N=30,
        dt=0.05,
        integrator_substeps=2,
        mode=mode,
        max_sqp_iterations=50,
        tolerance=1e-5,
        qp_tolerance=1e-8,
    )

    return Controller(model, bounds, options)

def simulate(mode="sqp"):
    controller = make_controller(mode=mode)

    dt = controller.options.dt
    n_steps = 100

    states = np.zeros((4, n_steps + 1))
    controls = np.zeros(n_steps)
    solve_times = np.zeros(n_steps)
    iterations = np.zeros(n_steps, dtype=int)
    violations = np.zeros(n_steps)

    states[:, 0] = [0.0, 0.2, 0.0, 0.0]

    for k in range(n_steps):
        result = controller.solve(states[:, k])

        expected_status = (
            "converged" if mode == "sqp" else "rti_step"
        )

        if result.status != expected_status:
            raise RuntimeError(
                f"Controller failed at step {k}: "
                f"{result.status}"
            )

        controls[k] = result.control[0]
        solve_times[k] = result.solve_time
        iterations[k] = result.iterations
        violations[k] = result.constraint_violation

        states[:, k + 1] = np.asarray(
            controller.problem.step(
                states[:, k],
                result.control,
                [],
            )
        ).ravel()

    return dt, states, controls, solve_times, iterations, violations


def save_results(dt, states, controls, times, iterations, violations):
    output = Path("results/cartpole")
    output.mkdir(parents=True, exist_ok=True)

    np.savez(
        output / "simulation_data.npz",
        dt=dt,
        states=states,
        controls=controls,
        solve_times=times,
        iterations=iterations,
        violations=violations,
    )

    metrics = {
        "final_state": states[:, -1].tolist(),
        "max_force_N": float(np.max(np.abs(controls))),
        "max_cart_position_m": float(np.max(np.abs(states[0]))),
        "mean_solve_time_ms": float(np.mean(times) * 1000),
        "median_solve_time_ms": float(np.median(times) * 1000),
        "max_solve_time_ms": float(np.max(times) * 1000),
        "p95_solve_time_ms": float(np.percentile(times, 95) * 1000),
        "mean_sqp_iterations": float(np.mean(iterations)),
        "max_constraint_violation": float(np.max(violations)),
        "fraction_under_deadline": float(np.mean(times <= dt)),
    }

    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    return output, metrics


def plot_results(dt, states, controls, times, output):
    t_states = np.arange(states.shape[1]) * dt
    t_controls = np.arange(controls.size) * dt

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(t_states, states[1])
    axes[0].axhline(0, linestyle="--")
    axes[0].set_ylabel("Pendulum angle [rad]")
    axes[0].grid(True)

    axes[1].plot(t_states, states[0])
    axes[1].axhline(0, linestyle="--")
    axes[1].axhline(2, linestyle=":")
    axes[1].axhline(-2, linestyle=":")
    axes[1].set_ylabel("Cart position [m]")
    axes[1].grid(True)

    axes[2].step(t_controls, controls, where="post")
    axes[2].axhline(10, linestyle="--")
    axes[2].axhline(-10, linestyle="--")
    axes[2].set_ylabel("Force [N]")
    axes[2].grid(True)

    axes[3].plot(t_controls, times * 1000)
    axes[3].axhline(
        dt * 1000,
        linestyle="--",
        label="Control period",
    )
    axes[3].set_ylabel("Solve time [ms]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()
    axes[3].grid(True)

    fig.suptitle("Nonlinear MPC — Cart-Pole Stabilization")
    fig.tight_layout()

    fig.savefig(
        output / "closed_loop.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    dt, states, controls, times, iterations, violations = simulate()

    output, metrics = save_results(
        dt, states, controls, times, iterations, violations
    )

    print("\nCart-pole closed-loop results")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    plot_results(dt, states, controls, times, output)
