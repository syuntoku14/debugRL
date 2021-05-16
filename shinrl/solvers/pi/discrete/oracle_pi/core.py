from copy import deepcopy

import numpy as np

from shinrl.solvers import BaseSolver

OPTIONS = {"constant_mix_rate": -1, "use_spi": False, "noise_scale": 0.0}


class Solver(BaseSolver):
    default_options = deepcopy(BaseSolver.default_options)
    default_options.update(OPTIONS)

    def initialize(self, options={}):
        assert self.is_tabular
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        values = np.zeros((self.dS, self.dA))
        policy = np.ones((self.dS, self.dA)) / self.dA
        self.record_array("Values", values)
        self.record_array("Policy", policy)
        self.mix_rate = 1.0
