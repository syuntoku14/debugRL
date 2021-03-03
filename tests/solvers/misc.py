def run_solver_tb(solver, env):
    solver.solve_options["record_performance_interval"] = 1
    solver.run(num_steps=10)
    action_values = solver.env.compute_action_values(solver.tb_policy)
    visitation = env.compute_visitation(solver.tb_policy)
    assert solver.step == 10
    assert solver.tb_values.shape \
        == solver.tb_policy.shape \
        == action_values.shape \
        == (env.dS, env.dA)
    assert visitation.shape \
        == (env.horizon, env.dS, env.dA)


def run_solver_gym(solver, env):
    solver.solve_options["record_performance_interval"] = 1
    solver.solve_options["num_episodes_gym_record"] = 1
    solver.run(num_steps=10)
    assert solver.step == 10
