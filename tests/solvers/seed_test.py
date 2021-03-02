
def test_seed_tabular():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    env.reset()
    solver = SamplingViSolver(env)
    fsolver = SamplingFittedViSolver(env)
    solver.initialize({"seed": 0, "record_performance_interval": 1})
    fsolver.initialize({"seed": 0, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val1 = solver.values
    fval1 = fsolver.values

    solver.initialize({"seed": 0, "record_performance_interval": 1})
    fsolver.initialize({"seed": 0, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val2 = solver.values
    fval2 = fsolver.values

    assert np.all(val1 == val2)
    assert np.all(fval1 == fval2)

    solver.initialize({"seed": 1, "record_performance_interval": 1})
    fsolver.initialize({"seed": 1, "record_performance_interval": 1})
    solver.run(num_steps=10)
    fsolver.run(num_steps=10)
    val3 = solver.values
    fval3 = fsolver.values

    assert not np.all(val1 == val3)
    assert not np.all(fval1 == fval3)
