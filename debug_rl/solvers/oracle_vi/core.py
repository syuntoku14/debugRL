import numpy as np
from debug_rl.solvers import Solver
from debug_rl.utils import (
    boltzmann_softmax,
    mellow_max,
) 


OPTIONS = {
    # CVI settings
    "er_coef": 0.2,
    "kl_coef": 0.1,
    "max_operator": "mellow_max",
}


class Solver(Solver):
    def initialize(self, options={}):
        self.solve_options.update(OPTIONS)
        super().initialize(options)
        self.record_array("Values", np.zeros((self.dS, self.dA)))

        # set max_operator
        if self.solve_options["max_operator"] == "boltzmann_softmax":
            self.max_operator = boltzmann_softmax
        elif self.solve_options["max_operator"] == "mellow_max":
            self.max_operator = mellow_max
        else:
            raise ValueError("Invalid max_operator")

    def record_performance(self, values):
        if self.step % self.solve_options["record_performance_interval"] == 0:
            values = np.asarray(values)
            policy = self.compute_policy(values)
            expected_return = self.env.compute_expected_return(policy)
            self.record_array("Policy", policy)
            self.record_array("Values", values)
            self.record_scalar("Return", expected_return, tag="Policy")
