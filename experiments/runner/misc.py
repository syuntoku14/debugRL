from shinrl import solvers

MINATAR_SOLVERS = {
    "DQN": solvers.minatar.DqnSolver,
    "M-DQN": solvers.minatar.MDqnSolver,
    "SAC": solvers.minatar.SacSolver,
}

MUJOCO_SOLVERS = {"SAC": solvers.mujoco.SacSolver}

TABULAR_DISCRETE_SOLVERS = {
    # Value iteration
    "OVI": solvers.tabular.discrete.vi.OracleViSolver,
    "EFVI": solvers.tabular.discrete.vi.ExactFittedViSolver,
    "SVI": solvers.tabular.discrete.vi.SamplingViSolver,
    "SFVI": solvers.tabular.discrete.vi.SamplingFittedViSolver,
    # Policy iteration
    "OPI": solvers.tabular.discrete.pi.OraclePiSolver,
    "OCPI": solvers.tabular.discrete.pi.OracleCpiSolver,
    # Conservative value iteration
    "OCVI": solvers.tabular.discrete.vi.OracleCviSolver,
    "EFCVI": solvers.tabular.discrete.vi.ExactFittedCviSolver,
    "SCVI": solvers.tabular.discrete.vi.SamplingCviSolver,
    "SFCVI": solvers.tabular.discrete.vi.SamplingFittedCviSolver,
    # Munchausen value iteration
    "OMVI": solvers.tabular.discrete.vi.OracleMviSolver,
    "EFMVI": solvers.tabular.discrete.vi.ExactFittedMviSolver,
    "SMVI": solvers.tabular.discrete.vi.SamplingMviSolver,
    "SFMVI": solvers.tabular.discrete.vi.SamplingFittedMviSolver,
    # Policy gradient
    "EPG": solvers.tabular.discrete.pg.ExactPgSolver,
    "SPG": solvers.tabular.discrete.pg.SamplingPgSolver,
    "PPO": solvers.tabular.discrete.pg.PpoSolver,
}

TABULAR_CONTINUOUS_SOLVERS = {
    "EPG": solvers.tabular.continuous.pg.ExactPgSolver,
    "SPG": solvers.tabular.continuous.pg.SamplingPgSolver,
    "PPO": solvers.tabular.continuous.pg.PpoSolver,
}


def to_numeric(arg):
    try:
        float(arg)
    except ValueError:
        if arg == "False":
            return False
        elif arg == "True":
            return True
        else:
            return arg
    else:
        return int(float(arg)) if float(arg).is_integer() else float(arg)


def make_valid_options(args, SOLVER):
    options = {}
    for key, val in vars(args).items():
        if key in SOLVER.default_options:
            options[key] = to_numeric(val)
    return options


def prepare(
    parser,
    defaults={
        "--epochs": 30,
        "--evaluation_interval": 5000,
        "--steps_per_epoch": 100000,
    },
):
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    for key, default in defaults.items():
        parser.add_argument(key, type=type(default), default=default)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--clearml", action="store_true")

    # For arbitrary options, e.g. --device cpu --seed 0
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split("=")[0])
    args = parser.parse_args()
    project_name = args.env if args.exp_name is None else args.exp_name
    task_name = args.solver if args.task_name is None else args.task_name

    if args.clearml:
        from clearml import Task

        task = Task.init(
            project_name=project_name, task_name=task_name, reuse_last_task_id=False
        )
        logger = task.get_logger()
    else:
        task, logger = None, None

    return args, project_name, task_name, task, logger
