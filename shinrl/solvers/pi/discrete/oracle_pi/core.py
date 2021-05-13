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
        self.set_tb_values(values)
        self.set_tb_policy(policy)
        self.mix_rate = 1.0

    def set_tb_policy(self, policy):
        if "Policy" not in self.history.keys():
            self.record_array("PrevPolicy", np.asarray(policy))
        else:
            self.record_array("PrevPolicy", self.tb_policy)
        self.record_array("Policy", np.asarray(policy))

    def set_tb_values(self, values):
        values = np.asarray(values)
        noise_scale = self.solve_options["noise_scale"]
        values = values + np.random.randn(*values.shape) * noise_scale
        self.record_array("Values", values)

    @property
    def tb_prev_policy(self):
        if len(self.history["PrevPolicy"]["array"]) == 0:
            raise ValueError('"PrevPolicy" has not been recorded yet. Check history.')
        return self.history["PrevPolicy"]["array"][-1]
