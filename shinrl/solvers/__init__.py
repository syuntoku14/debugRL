from .base import BaseSolver  # isort:skip
from shinrl.solvers import minatar  # NOQA
from shinrl.solvers import mujoco  # NOQA
from shinrl.solvers import onpg  # NOQA
from shinrl.solvers import pi  # NOQA
from shinrl.solvers import sac  # NOQA
from shinrl.solvers import vi  # NOQA

__all__ = ["Solver", "vi", "onpg", "sac", "pi", "minatar", "mujoco"]
