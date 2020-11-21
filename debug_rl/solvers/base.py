import warnings
import pickle
import numpy as np
import torch
import gym
from torch.nn import functional as F
from abc import ABC, abstractmethod
from collections import defaultdict
from scipy import sparse
from debug_rl.utils import (
    get_all_observations, get_all_actions, boltzmann_softmax,
    mellow_max, make_replay_buffer)
from copy import deepcopy


DEFAULT_OPTIONS = {
    # general settings
    "seed": 0,
    "discount": 0.99,
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
        self.env = env
        self.dS = env.num_states
        self.dA = env.num_actions
        self.transition_matrix = env.transition_matrix()  # (SxA)xS
        self.rew_matrix = env.reward_matrix()  # (SxA)x1
        self.horizon = env.horizon
        self.all_obss = get_all_observations(env)
        self.all_actions = get_all_actions(env) if env.action_mode == "continuous" else None
        # matrix for speeding up computation
        self.trr_rew_sum = np.sum(self.transition_matrix.multiply(
            self.rew_matrix), axis=-1).reshape(self.dS, self.dA)  # SxA

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
        Iterate the backup function and comptue value iteration.

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

    def compute_action_values(self, policy):
        """
        Compute action values using bellman operator.
        Returns:
            Q values: SxA matrix
        """

        values = np.zeros((self.dS, self.dA))  # SxA

        def backup(curr_q_val, policy):
            discount = self.solve_options["discount"]
            curr_v_val = np.sum(policy * curr_q_val, axis=-1)  # S
            prev_q = self.trr_rew_sum \
                + discount*(self.transition_matrix *
                            curr_v_val).reshape(self.dS, self.dA)
            prev_q = np.asarray(prev_q)
            return prev_q

        for _ in range(self.horizon):
            values = backup(values, policy)  # SxA

        return values

    def compute_visitation(self, policy, discount=1.0):
        """
        Compute state and action frequency on the given policy
        Args:
            policy: SxA matrix
            discount (float): discount factor.

        Returns:
            visitation: TxSxA matrix
        """
        sa_visit = np.zeros((self.horizon, self.dS, self.dA))  # SxAxS
        s_visit_t = np.zeros((self.dS, 1))  # Sx1
        for (state, prob) in self.env.initial_state_distribution.items():
            s_visit_t[state] = prob

        norm_factor = 0.0
        for t in range(self.horizon):
            cur_discount = (discount ** t)
            norm_factor += cur_discount
            sa_visit_t = s_visit_t * policy  # SxA
            sa_visit[t, :, :] = cur_discount * sa_visit_t
            # sum-out (SA)S
            new_s_visit_t = \
                sa_visit_t.reshape(self.dS*self.dA) * self.transition_matrix
            s_visit_t = np.expand_dims(new_s_visit_t, axis=1)
        visitation = sa_visit / norm_factor
        return visitation

    def compute_expected_return(self, policy):
        """
        Compute expected return of the given policy

        Args:
            policy: SxA matrix

        Returns:
            expected_return (float)
        """

        q_values = np.zeros((self.dS, self.dA))  # SxA
        for t in reversed(range(1, self.horizon+1)):
            # backup q values
            curr_vval = np.sum(policy * q_values, axis=-1)  # S
            prev_q = (self.transition_matrix *
                      curr_vval).reshape(self.dS, self.dA)  # SxA
            q_values = np.asarray(prev_q + self.trr_rew_sum)  # SxA

        init_vval = np.sum(policy*q_values, axis=-1)  # S
        init_probs = np.zeros(self.dS)  # S
        for (state, prob) in self.env.initial_state_distribution.items():
            init_probs[state] = prob
        expected_return = np.sum(init_probs * init_vval)
        return expected_return
