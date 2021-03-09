import pprint
import warnings
import pickle
import numpy as np
import torch
import gym
from abc import ABC, abstractmethod
from collections import defaultdict
from scipy import sparse
from shinrl.envs import TabularEnv
from copy import deepcopy


DEFAULT_OPTIONS = {
    # general settings
    "seed": 0,
    "discount": 0.99,
    "num_episodes_gym_record": 10,
    "record_performance_interval": 100,
    "record_all_array": False
}


class Solver(ABC):
    def __init__(self, env, solve_options={}, logger=None):
        """
        Solver for a finite horizon MDP.
        Args:
            env (TabularEnv)
            solve_options (dict): Parameters for the algorithm, e.g. discount_factor.
            logger (Trains.Logger): logger for Trains
        """
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
        Update self.solve_options with options.
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
                    _, _, done, _ = self.env.step(self.env.action_space.sample())
                self.env.obs = self.env.reset()
        else:
            self.env.obs = self.env.reset()

        # initialize history
        self.history = defaultdict(lambda: {"x": [], "y": []})
        self.step = 0
        self.print_options()

    @abstractmethod
    def run(self, num_steps=1000):
        """
        Run algorithm to solve a MDP.
            - num_trains: Number of iterations
            - restart: Reset the solver and run again if True
        """

    @property
    def step(self):
        return self.history["step"]

    @step.setter
    def step(self, step):
        self.history["step"] = step

    @property
    def tb_values(self):
        if len(self.history["Values"]["y"]) == 0:
            assert ValueError(
                "\"values\" has not been recorded yet. Check history.")
        return self.history["Values"]["y"][-1]

    @property
    def tb_policy(self):
        if len(self.history["Policy"]["y"]) == 0:
            assert ValueError(
                "\"policy\" has not been recorded yet. Check history.")
        return self.history["Policy"]["y"][-1]

    def record_scalar(self, title, y, tag=None):
        """
        Record a scalar y to self.history. 
        Report to trains also if self.logger is set.
        """
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
            self.history[title]["y"].append(array.astype(np.float32))
        else:
            self.history[title]["x"] = [self.step, ]
            self.history[title]["y"] = [array.astype(np.float32), ]

    def save(self, path, data={}):
        data.update({
            "env_name": str(self.env),
            "history": dict(self.history),
            "options": self.solve_options,
            "step": self.step
        })
        torch.save(data, path)

    def load(self, path, device="cpu"):
        data = torch.load(path, map_location=device)
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
