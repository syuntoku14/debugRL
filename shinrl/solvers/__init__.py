from .base import BaseSolver  # isort:skip
from shinrl.solvers import minatar  # NOQA
from shinrl.solvers import mujoco  # NOQA
from shinrl.solvers import tabular  # NOQA

__all__ = ["Solver", "minatar", "mujoco", "tabular"]
