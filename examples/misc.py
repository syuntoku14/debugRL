from shinrl import solvers

DISCRETE_SOLVERS = {
    # Value iteration
    "OVI": solvers.vi.discrete.OracleViSolver,
    "EFVI": solvers.vi.discrete.ExactFittedViSolver,
    "SVI": solvers.vi.discrete.SamplingViSolver,
    "SFVI": solvers.vi.discrete.SamplingFittedViSolver,
    # Conservative value iteration
    "OCVI": solvers.vi.discrete.OracleCviSolver,
    "EFCVI": solvers.vi.discrete.ExactFittedCviSolver,
    "SCVI": solvers.vi.discrete.SamplingCviSolver,
    "SFCVI": solvers.vi.discrete.SamplingFittedCviSolver,
    # Policy gradient
    "EPG": solvers.pg.discrete.ExactPgSolver,
    "SPG": solvers.pg.discrete.SamplingPgSolver,
    "IPG": solvers.ipg.discrete.IpgSolver,
    "PPO": solvers.ppo.discrete.PpoSolver,
    # SAC
    "SAC": solvers.sac.discrete.SacSolver,
}

CONTINUOUS_SOLVERS = {
    "SAC": solvers.sac.continuous.SacSolver,
    "PPO": solvers.ppo.continuous.PpoSolver,
}


def to_numeric(arg):
    try:
        float(arg)
    except ValueError:
        return arg
    else:
        return int(float(arg)) if float(arg).is_integer() else float(arg)
