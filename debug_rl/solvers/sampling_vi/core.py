import numpy as np
from debug_rl.solvers import Solver
from debug_rl.utils import (
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

        # set max_operator
        if self.solve_options["max_operator"] == "boltzmann_softmax":
            self.max_operator = boltzmann_softmax
        elif self.solve_options["max_operator"] == "mellow_max":
            self.max_operator = mellow_max
        else:
            raise ValueError("Invalid max_operator")

    def record_performance(self, values, eval_policy):
        if self.step % self.solve_options["record_performance_interval"] == 0:
            expected_return = self.env.compute_expected_return(eval_policy)
            self.record_scalar("Return", expected_return, tag="Policy")
            self.record_array("Policy", eval_policy)
            self.record_array("Values", values)
