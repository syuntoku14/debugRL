import numpy as np

from shinrl.solvers import Solver
from shinrl.utils import boltzmann_softmax, mellow_max

OPTIONS = {
    # CVI settings
    "er_coef": 0.2,
    "kl_coef": 0.1,
    "max_operator": "mellow_max",
    "noise_scale": 0.0,
}


class Solver(Solver):
    def initialize(self, options={}):
        assert self.is_tabular
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.record_array("Values", np.zeros((self.dS, self.dA)))
        self.record_array("Policy", np.ones((self.dS, self.dA)) / self.dA)

        # set max_operator
        if self.solve_options["max_operator"] == "boltzmann_softmax":
            self.max_operator = boltzmann_softmax
        elif self.solve_options["max_operator"] == "mellow_max":
            self.max_operator = mellow_max
        else:
            raise ValueError("Invalid max_operator")
