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
    "EPG": solvers.onpg.discrete.ExactPgSolver,
    "SPG": solvers.onpg.discrete.SamplingPgSolver,
    "PPO": solvers.onpg.discrete.PpoSolver,
    "IPG": solvers.ipg.discrete.IpgSolver,
    # SAC
    "SAC": solvers.sac.discrete.SacSolver,
}

CONTINUOUS_SOLVERS = {
    "EPG": solvers.onpg.continuous.ExactPgSolver,
    "SPG": solvers.onpg.continuous.SamplingPgSolver,
    "PPO": solvers.onpg.continuous.PpoSolver,
    "SAC": solvers.sac.continuous.SacSolver,
}


def to_numeric(arg):
    try:
        float(arg)
    except ValueError:
        return arg
    else:
        return int(float(arg)) if float(arg).is_integer() else float(arg)
