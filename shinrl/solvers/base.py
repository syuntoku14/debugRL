import pprint
import warnings
import pickle
import numpy as np
import torch
import gym
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from scipy import sparse
from shinrl.envs import TabularEnv
from copy import deepcopy


DEFAULT_OPTIONS = {
    # general settings
    "seed": 0,
    "discount": 0.99,
    "gym_evaluation_episodes": 10,
    "evaluation_interval": 100,
    "log_interval": 100,
    "record_all_array": False
}


class Solver(ABC):
    """
    Args:
        env (gym.Env)
        solve_options (dict): Hyperparameters for the algorithm, e.g. discount_factor.
        logger (Trains.Logger): logger for ClearML
    """

    def __init__(self, env, solve_options={}, logger=None):
        # solver options
        self.solve_options = DEFAULT_OPTIONS
        self.logger = logger

        # env parameters
        self.is_tabular = isinstance(env, TabularEnv)
        if self.is_tabular:
            self.dS, self.dA, self.horizon = env.dS, env.dA, env.horizon
        self.env = env
        self.initialize(solve_options)

    def initialize(self, options={}):
        """
        Initialize the solver with the given options. 
        Reset the env and history.
        """
        options = deepcopy(options)
        for key in options.keys():
            if key not in self.solve_options:
                raise KeyError("{} is an invalid option.".format(key))
        self.solve_options.update(options)

        # set seed
        np.random.seed(self.solve_options["seed"])
        torch.manual_seed(self.solve_options["seed"])
        if isinstance(self.env, gym.wrappers.Monitor):
            # With Monitor, you cannot call reset() unless the episode is over.
            if self.env.stats_recorder.steps is None:
                self.env.obs = self.env.reset()
            else:
                done = False
                while not done:
                    _, _, done, _ = self.env.step(
                        self.env.action_space.sample())
                self.env.obs = self.env.reset()
        else:
            self.env.obs = self.env.reset()

        # initialize history
        self.history = defaultdict(lambda: {"x": [], "y": []})
        self.step = 0
        self.print_options()

    @abstractmethod
    def run(self, num_steps=1000):
        """ Run the algorithm with the given environment.

        Args:
            num_steps (int): Number of iterations.
        """

    @property
    def step(self):
        return self.history["step"]

    @step.setter
    def step(self, step):
        self.history["step"] = step

    @property
    def tb_values(self):
        if len(self.history["Values"]["array"]) == 0:
            raise ValueError(
                "\"values\" has not been recorded yet. Check history.")
        return self.history["Values"]["array"][-1]

    @property
    def tb_policy(self):
        if len(self.history["Policy"]["array"]) == 0:
            raise ValueError(
                "\"policy\" has not been recorded yet. Check history.")
        return self.history["Policy"]["array"][-1]

    def record_scalar(self, title, y, tag=None):
        """
        Record a scalar y to self.history. 
        Report to clearML if self.logger is set.
        """
        if len(self.history[title]["x"]) > 0:
            prev_step = self.history[title]["x"][-1]
            if self.step - prev_step < self.solve_options["log_interval"]:
                return None
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().item()
        if self.logger is not None:
            _tag = title if tag is None else tag
            self.logger.report_scalar(title, _tag, y, self.step)
        if tag is not None:
            title += tag
        self.history[title]["x"].append(self.step)
        self.history[title]["y"].append(y)

    def record_array(self, title, array):
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        assert isinstance(array, np.ndarray), \
            "array must be a torch.tensor or a numpy.ndarray"

        if self.solve_options["record_all_array"]:
            self.history[title]["x"].append(self.step)
            self.history[title]["array"].append(array.astype(np.float32))
        else:
            self.history[title]["x"] = [self.step, ]
            self.history[title]["array"] = [array.astype(np.float32), ]

    def save(self, dir_name, data={}):
        def list_to_array(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    list_to_array(v)
                elif isinstance(v, list):
                    d[k] = np.array(v)
            return d
        history = list_to_array(dict(deepcopy(self.history)))
        data.update({
            "env_name": str(self.env),
            "history": history,
            "options": self.solve_options,
            "step": self.step
        })
        torch.save(data, os.path.join(dir_name, "data.tar"))
        return data

    def load(self, dir_name, options={}, device="cpu"):
        def array_to_list(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    array_to_list(v)
                elif isinstance(v, np.ndarray) and not k == "array":
                    d[k] = v.tolist()
            return d
        data = torch.load(os.path.join(dir_name, "data.tar"), map_location=device)
        data["history"] = array_to_list(data["history"])
        data["options"].update(options)
        self.initialize(data["options"])
        self.history.update(data["history"])
        self.step = data["step"]
        return data

    def print_options(self):
        print("{} solve_options:".format(type(self).__name__))
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.solve_options)
        if not self.solve_options["record_all_array"]:
            warnings.warn(
                "The option \"record_all_array\" is False and the record_array function only record the lastly recorded array", UserWarning)
