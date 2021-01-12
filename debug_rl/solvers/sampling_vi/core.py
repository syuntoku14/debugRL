import pprint
from debug_rl.solvers import Solver
from debug_rl.utils import (
    boltzmann_softmax,
    mellow_max,
)


OPTIONS = {
    "num_samples": 4,
    # CVI settings
    "alpha": 0.9,
    "beta": 1.0,
    "max_operator": "mellow_max",
    # Q settings
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 200,
    # Tabular settings
    "lr": 0.1,  # learning rate of tabular methods
    "num_trains": 10000,  # for sampling methods
}


class Solver(Solver):
    def set_options(self, options={}):
        self.solve_options.update(OPTIONS)
        super().set_options(options)

        # set max_operator
        if self.solve_options["max_operator"] == "boltzmann_softmax":
            self.max_operator = boltzmann_softmax
        elif self.solve_options["max_operator"] == "mellow_max":
            self.max_operator = mellow_max
        else:
            raise ValueError("Invalid max_operator")

        print("{} solve_options:".format(type(self).__name__))
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.solve_options)

    def record_performance(self, k, values, eval_policy, force=False):
        if k % self.solve_options["record_performance_interval"] == 0 or force:
            expected_return = self.env.compute_expected_return(eval_policy)
            self.record_scalar("Return mean", expected_return, x=k)
            self.record_array("policy", eval_policy, x=k)
            self.record_array("values", values, x=k)
