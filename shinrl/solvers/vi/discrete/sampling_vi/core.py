from copy import deepcopy

import numpy as np

from shinrl import utils
from shinrl.solvers import BaseSolver

OPTIONS = {
    "num_samples": 4,
    # CVI settings
    "er_coef": 0.2,
    "kl_coef": 0.1,
    "max_operator": "mellow_max",
    # Q settings
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 200,
    # Tabular settings
    "lr": 0.1,  # learning rate of tabular methods
}


class Solver(BaseSolver):
    default_options = deepcopy(BaseSolver.default_options)
    default_options.update(OPTIONS)

    def initialize(self, options={}):
        assert self.is_tabular
        super().initialize(options)
        self.record_array("Values", np.zeros((self.dS, self.dA)))
        self.record_array("Policy", np.ones((self.dS, self.dA)) / self.dA)
        self.max_operator = getattr(utils, self.solve_options["max_operator"])
