import numpy as np
from shinrl.solvers import Solver
from shinrl.utils import (
    boltzmann_softmax,
    mellow_max,
)


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


class Solver(Solver):
    def initialize(self, options={}):
        assert self.is_tabular
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.record_array("Values", np.zeros((self.dS, self.dA)))
        self.record_array("Policy", np.ones((self.dS, self.dA))/self.dA)

        # set max_operator
        if self.solve_options["max_operator"] == "boltzmann_softmax":
            self.max_operator = boltzmann_softmax
        elif self.solve_options["max_operator"] == "mellow_max":
            self.max_operator = mellow_max
        else:
            raise ValueError("Invalid max_operator")
