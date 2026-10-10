from pathlib import Path
import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from generalmpc import Controller, Model, Bounds, Options


def make_controller():
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    rhs = ca.Function(
        "pendulum_rhs",
        [x, u, p],
        [ca.vertcat(
            x[1],
            -9.81 * ca.sin(x[0])
            - 0.1 * x[1]
            + u[0],
        )],
    )

    stage = ca.Function(
        "pendulum_stage",
        [x, u, p],
        [ca.vertcat(
            ca.sqrt(10.0) * x[0],
            x[1],
            ca.sqrt(0.1) * u[0],
        )],
    )

    terminal = ca.Function(
        "pendulum_terminal",
        [x, p],
        [ca.vertcat(
            ca.sqrt(20.0) * x[0],
            ca.sqrt(2.0) * x[1],
        )],
    )

    model = Model(
        nx=2,
        nu=1,
        np=0,
        rhs=rhs,
        stage=stage,
        terminal=terminal,
    )

    bounds = Bounds(
        u_lower=[-5.0],
        u_upper=[5.0],
    )

    options = Options(
        N=20,
        dt=0.05,
        integrator_substeps=2,
        mode="sqp",
        max_sqp_iterations=50,
        tolerance=1e-5,
        qp_tolerance=1e-9,
    )

    return Controller(model, bounds, options)


def simulate():
    controller = make_controller()
    dt = controller.options.dt
    n_steps = 100

    states = np.zeros((2, n_steps + 1))
    controls = np.zeros(n_steps)
    solve_times = np.zeros(n_steps)
    iterations = np.zeros(n_steps, dtype=int)

    states[:, 0] = [0.5, 0.0]

    for k in range(n_steps):
        result = controller.solve(states[:, k])

        if result.status != "converged":
            raise RuntimeError(
                f"MPC failed at step {k}: {result.status}"
            )

        controls[k] = result.control[0]
        solve_times[k] = result.solve_time
        iterations[k] = result.iterations

        # Apply the first optimal control using
        # the discretized plant dynamics.
        states[:, k + 1] = np.asarray(
            controller.problem.step(
                states[:, k],
                result.control,
                [],
            )
        ).ravel()

    return dt, states, controls, solve_times, iterations


def plot_results(dt, states, controls, solve_times):
    output = Path("results/pendulum")
    output.mkdir(parents=True, exist_ok=True)

    time_states = np.arange(states.shape[1]) * dt
    time_controls = np.arange(controls.size) * dt

    fig, axes = plt.subplots(
        4, 1, figsize=(9, 11), sharex=True
    )

    axes[0].plot(time_states, states[0])
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Angle [rad]")
    axes[0].grid(True)

    axes[1].plot(time_states, states[1])
    axes[1].set_ylabel("Velocity [rad/s]")
    axes[1].grid(True)

    axes[2].step(
        time_controls,
        controls,
        where="post",
    )
    axes[2].axhline(5, linestyle="--")
    axes[2].axhline(-5, linestyle="--")
    axes[2].set_ylabel("Torque [Nm]")
    axes[2].grid(True)

    axes[3].plot(
        time_controls,
        solve_times * 1000,
    )
    axes[3].axhline(
        dt * 1000,
        linestyle="--",
        label="Control period",
    )
    axes[3].set_ylabel("Solve time [ms]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()
    axes[3].grid(True)

    fig.suptitle(
        "Nonlinear MPC — Pendulum Stabilization"
    )
    fig.tight_layout()

    fig.savefig(
        output / "closed_loop.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    dt, states, controls, times, iterations = simulate()

    output = Path("results/pendulum")
    output.mkdir(parents=True, exist_ok=True)

    np.savez(
        output / "simulation_data.npz",
        dt=dt,
        states=states,
        controls=controls,
        solve_times=times,
        iterations=iterations,
    )

    print("\nClosed-loop results")
    print(f"Final angle:      {states[0,-1]:.6f} rad")
    print(f"Final velocity:   {states[1,-1]:.6f} rad/s")
    print(f"Max torque:       {np.max(np.abs(controls)):.4f} Nm")
    print(f"Mean solve time:  {np.mean(times)*1000:.3f} ms")
    print(f"Median solve time:{np.median(times)*1000:.3f} ms")
    print(f"Max solve time:   {np.max(times)*1000:.3f} ms")
    print(f"Mean SQP iters:   {np.mean(iterations):.2f}")

    plot_results(dt, states, controls, times)
