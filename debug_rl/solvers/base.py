import warnings
import pickle
import numpy as np
import torch
import gym
from torch.nn import functional as F
from abc import ABC, abstractmethod
from collections import defaultdict
from scipy import sparse
from debug_rl.envs.base import TabularEnv
from debug_rl.utils import (
    boltzmann_softmax, mellow_max, make_replay_buffer)
from copy import deepcopy


DEFAULT_OPTIONS = {
    # general settings
    "seed": 0,
    "discount": 0.99,
    "record_performance_interval": 100,
    "record_all_array": False
}


def check_tabular_env(func):
    def decorator(self, *args, **kwargs):
        assert self.is_tabular_env, "self.env is not TabularEnv"
        return func(self, *args, **kwargs)
    return decorator


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
        self.env = env
        self.is_tabular_env = isinstance(self.env, TabularEnv)

        if self.is_tabular_env:
            self.dS = env.dS
            self.dA = env.dA
            self.transition_matrix = env.transition_matrix  # (SxA)xS
            self.reward_matrix = env.reward_matrix  # (SxA)x1
            self.trr_rew_sum = env.trr_rew_sum
            self.horizon = env.horizon
            self.all_obss = env.all_observations
            self.all_actions = env.all_actions if env.action_mode == "continuous" else None
        self.init_history()
        self.set_options(solve_options)

    def set_options(self, options={}):
        """
        Update self.solve_options with options.
        """
        for key in options.keys():
            if key not in self.solve_options:
                raise KeyError("{} is an invalid option.".format(key))
        self.solve_options.update(options)

        # set seed
        np.random.seed(self.solve_options["seed"])
        torch.manual_seed(self.solve_options["seed"])

    def init_history(self):
        self.history = defaultdict(lambda: {"x": [], "y": []})

    @abstractmethod
    def solve(self):
        """
        Iterate the backup function and compute value iteration.

        Returns:
            SxA matrix
        """

    @property
    def values(self):
        if len(self.history["values"]["y"]) == 0:
            assert ValueError("\"values\" has not been recorded yet. Check history.")
        return self.history["values"]["y"][-1]

    @property
    def policy(self):
        if len(self.history["policy"]["y"]) == 0:
            assert ValueError("\"policy\" has not been recorded yet. Check history.")
        return self.history["policy"]["y"][-1]

    def record_scalar(self, title, y, x=None, tag=None):
        """
        Record a scalar y to self.history. 
        Report to trains also if self.logger is set.
        """
        if x is None:
            x = len(self.history[title]["x"])
        if self.logger is not None:
            tag = title if tag is None else tag
            self.logger.report_scalar(title, tag, y, x)
        self.history[title]["x"].append(x)
        self.history[title]["y"].append(y)

    def record_array(self, title, array, x=None):
        if x is None:
            x = len(self.history[title]["x"])
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        assert isinstance(array, np.ndarray), \
            "array must be a torch.tensor or a numpy.ndarray"

        if self.solve_options["record_all_array"]:
            self.history[title]["x"].append(x)
            self.history[title]["y"].append(array.astype(np.float32))
        else:
            warnings.warn(
                "The option \"record_all_array\" is False and the record_array \
                function only record the lastly recorded array", UserWarning)
            self.history[title]["x"] = [x, ]
            self.history[title]["y"] = [array.astype(np.float32), ]

    def save(self, path):
        data = {
            "env_name": str(self.env),
            "history": dict(self.history),
            "options": self.solve_options
        }
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path)
        self.init_history()
        self.history.update(data["history"])
        self.set_options(data["options"])
