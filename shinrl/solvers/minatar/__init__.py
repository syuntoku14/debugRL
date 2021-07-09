from shinrl.solvers.minatar.vi import DQNSolver, MDQNSolver  # NOQA
from .minatar_env import make_minatar  # NOQA


__all__ = ["DQNSolver", "MDQNSolver", "make_minatar"]