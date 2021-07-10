from shinrl.solvers.minatar.vi import DqnSolver, MDqnSolver  # NOQA
from shinrl.solvers.minatar.sac import SacSolver

from .minatar_env import make_minatar  # NOQA

__all__ = ["DqnSolver", "MDqnSolver", "SacSolver", "make_minatar"]
