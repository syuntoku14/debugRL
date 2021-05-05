import numpy as np

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    # CVI settings
    "er_coef": 0.2,
    "kl_coef": 0.1,
    "max_operator": "mellow_max",
    "noise_scale": 0.0,
}


class Solver(BaseSolver):
    def initialize(self, options={}):
        assert self.is_tabular
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.record_array("Values", np.zeros((self.dS, self.dA)))
        self.record_array("Policy", np.ones((self.dS, self.dA)) / self.dA)
        self.max_operator = getattr(utils, self.solve_options["max_operator"])
