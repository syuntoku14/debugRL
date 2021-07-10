from shinrl.solvers.minatar.vi import DQNSolver, MDQNSolver  # NOQA
from shinrl.solvers.minatar.sac import SacSolver

from .minatar_env import make_minatar  # NOQA

__all__ = ["DQNSolver", "MDQNSolver", "SacSolver", "make_minatar"]
