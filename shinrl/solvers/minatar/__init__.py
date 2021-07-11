from shinrl.solvers.minatar.sac import SacSolver
from shinrl.solvers.minatar.vi import DqnSolver, MDqnSolver  # NOQA

from .minatar_env import make_minatar  # NOQA

__all__ = ["DqnSolver", "MDqnSolver", "SacSolver", "make_minatar"]
