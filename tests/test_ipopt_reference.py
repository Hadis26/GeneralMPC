
import casadi as ca
import numpy as np

from generalmpc import Controller, Model, Bounds, Options


def make_controller():
    x = ca.SX.sym("x", 2)
    u = ca.SX.sym("u", 1)
    p = ca.SX.sym("p", 0)

    rhs = ca.Function(
        "pendulum_rhs",
        [x, u, p],
        [
            ca.vertcat(
                x[1],
                -9.81 * ca.sin(x[0])
                - 0.1 * x[1]
                + u[0],
            )
        ],
    )

    stage = ca.Function(
        "pendulum_stage",
        [x, u, p],
        [
            ca.vertcat(
                ca.sqrt(10.0) * x[0],
                x[1],
                ca.sqrt(0.1) * u[0],
            )
        ],
    )

    terminal = ca.Function(
        "pendulum_terminal",
        [x, p],
        [
            ca.vertcat(
                ca.sqrt(20.0) * x[0],
                ca.sqrt(2.0) * x[1],
            )
        ],
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


def test_sqp_matches_ipopt():
    controller = make_controller()
    problem = controller.problem

    measured = np.array([0.5, 0.0])

    # 1. Solve using our SQP implementation.
    sqp = controller.solve(measured)

    assert sqp.status == "converged"
    assert sqp.constraint_violation < 1e-5

    # 2. Construct the same NLP for IPOPT.
    theta, _ = problem.parameters(measured)

    nlp_solver = ca.nlpsol(
        "reference_ipopt",
        "ipopt",
        problem.nlp,
        {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.tol": 1e-9,
            "ipopt.max_iter": 1000,
        },
    )

    # Start IPOPT from the SQP solution so both
    # solvers are compared near the same local optimum.
    initial_guess = problem.pack(sqp.X, sqp.U)

    reference = nlp_solver(
        x0=initial_guess,
        p=theta,
        lbx=problem.lbx,
        ubx=problem.ubx,
        lbg=problem.lbg,
        ubg=problem.ubg,
    )

    assert nlp_solver.stats()["success"]

    w_ipopt = np.array(reference["x"]).ravel()
    X_ipopt, U_ipopt = problem.unpack(w_ipopt)

    objective_sqp = sqp.objective
    objective_ipopt = float(reference["f"])

    objective_difference = abs(
        objective_sqp - objective_ipopt
    )

    control_difference = np.max(
        np.abs(sqp.U - U_ipopt)
    )

    print("\nSQP vs. IPOPT")
    print(f"SQP objective:       {objective_sqp:.10f}")
    print(f"IPOPT objective:     {objective_ipopt:.10f}")
    print(f"Objective difference:{objective_difference:.3e}")
    print(f"Control difference:  {control_difference:.3e}")
    print(f"SQP iterations:      {sqp.iterations}")
    print(f"SQP solve time:      {sqp.solve_time:.4f} s")

    assert objective_difference < 1e-4
    assert control_difference < 1e-3

    np.testing.assert_allclose(
        sqp.X,
        X_ipopt,
        atol=1e-3,
        rtol=1e-3,
    )
