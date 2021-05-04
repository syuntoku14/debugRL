from shinrl import solvers
from clearml import Task

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
    # Munchausen value iteration
    "OMVI": solvers.vi.discrete.OracleMviSolver,
    "EFMVI": solvers.vi.discrete.ExactFittedMviSolver,
    "SMVI": solvers.vi.discrete.SamplingMviSolver,
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


def prepare(
    parser,
    defaults={
        "--epochs": 30,
        "--evaluation_interval": 5000,
        "--steps_per_epoch": 100000}
    ):
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--task_name', type=str, default=None)
    for key, default in defaults.items():
        parser.add_argument(key, type=type(default), default=default)
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--clearml', action="store_true")

    # For arbitrary options, e.g. --device cpu --seed 0
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0])
    args = parser.parse_args()

    options = {}
    for key, val in vars(args).items():
        if key not in ["solver", "env", "exp_name", "task_name",
                       "epochs", "steps_per_epoch",
                       "save", "clearml"]:
            options[key] = to_numeric(val)

    project_name = args.env if args.exp_name is None else args.exp_name
    task_name = args.solver if args.task_name is None else args.task_name

    if args.clearml:
        task = Task.init(project_name=project_name,
                         task_name=task_name, reuse_last_task_id=False)
        logger = task.get_logger()
    else:
        task, logger = None, None

    return args, options, project_name, task_name, task, logger