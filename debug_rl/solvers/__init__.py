from .base import Solver

from .oracle_vi.oracle_vi import (
    OracleViSolver,
    OracleCviSolver)

from .exact_fvi.exact_fvi import (
    ExactFittedViSolver,
    ExactFittedCviSolver)

from .sampling_vi.sampling_vi import (
    SamplingViSolver,
    SamplingCviSolver
)

from .sampling_fvi.sampling_fvi import (
    SamplingFittedViSolver,
    SamplingFittedCviSolver)

from .exact_pg.exact_pg import ExactPgSolver

from .sac.sac import SacSolver
from .sac_continuous.sac_continuous import SacContinuousSolver
from .ppo.ppo import PpoSolver
