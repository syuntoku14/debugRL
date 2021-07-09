from shinrl.solvers.atari.vi import DQNSolver, MDQNSolver  # NOQA
from .atari_wrappers import make_atari  # NOQA


__all__ = ["DQNSolver", "MDQNSolver", "make_atari"]