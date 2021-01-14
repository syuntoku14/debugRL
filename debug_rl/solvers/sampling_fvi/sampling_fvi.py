from scipy.special import rel_entr
from scipy import stats
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from .core import Solver
from debug_rl.envs.base import TabularEnv
from debug_rl.utils import (
    trajectory_to_tensor,
    squeeze_trajectory,
    collect_samples,
    compute_epsilon,
    eps_greedy_policy,
    softmax_policy
)


class SamplingFittedSolver(Solver):
    def solve(self):
        self.init_history()
        self.all_obss = torch.tensor(self.env.all_observations,
                                     dtype=torch.float32, device=self.device)
        values = self.value_network(self.all_obss).reshape(
            self.dS, self.dA).detach().cpu().numpy()

        self.env.reset()
        if self.solve_options["use_replay_buffer"]:
            # Collect random samples in advance
            if isinstance(self, SamplingFittedViSolver):
                policy = self.compute_policy(values, eps_greedy=1.0)
            else:
                policy = self.compute_policy(values)
            trajectory = collect_samples(
                self.env, policy, self.solve_options["minibatch_size"])
            self.buffer.add(**trajectory)

        # start training
        for k in tqdm(range(self.solve_options["num_trains"])):
            # ------ collect samples by the current policy ------
            if isinstance(self, SamplingFittedViSolver):
                eps_greedy = compute_epsilon(
                    k, self.solve_options["eps_start"],
                    self.solve_options["eps_end"],
                    self.solve_options["eps_decay"])
                self.record_scalar("epsilon", eps_greedy, x=k)
                policy = self.compute_policy(values, eps_greedy=eps_greedy)
                eval_policy = self.compute_policy(values, eps_greedy=0)
            else:
                policy = self.compute_policy(values)
                eval_policy = policy
            trajectory = collect_samples(
                self.env, policy, self.solve_options["num_samples"])

            if self.solve_options["use_replay_buffer"]:
                # ----- generate mini-batch from the replay_buffer -----
                self.buffer.add(**trajectory)
                trajectory = self.buffer.sample(self.solve_options["minibatch_size"])
                trajectory = squeeze_trajectory(trajectory)

            # ----- record performance -----
            self.record_performance(k, eval_policy)

            # ----- update values -----
            tensor_traj = trajectory_to_tensor(trajectory, self.device)
            target = self.backup(tensor_traj)
            loss = self.update_network(
                target, self.value_network, self.value_optimizer,
                obss=tensor_traj["obs"], actions=tensor_traj["act"])
            if self.solve_options["use_double_estimation"]:
                self.update_network(
                    target, self.value_network2, self.value_optimizer2,
                    obss=tensor_traj["obs"], actions=tensor_traj["act"])

            self.record_scalar("value loss", loss)

            values = self.value_network(self.all_obss).reshape(
                self.dS, self.dA).detach().cpu().numpy()

            # ----- update target network -----
            if self.solve_options["use_target_network"] and \
               (k+1) % self.solve_options["target_update_interval"] == 0:
                self.target_value_network.load_state_dict(
                    self.value_network.state_dict())
                self.target_value_network2.load_state_dict(
                    self.value_network2.state_dict())

        self.record_performance(k, eval_policy, force=True)


class SamplingFittedViSolver(SamplingFittedSolver):
    """
    This class is same as DQN if the following options are True
    * use_target_network
    * use_replay_buffer
    """

    def backup(self, tensor_traj):
        discount = self.solve_options["discount"]

        with torch.no_grad():
            next_obss = tensor_traj["next_obs"]
            rews = tensor_traj["rew"]
            dones = tensor_traj["done"]
            if self.solve_options["use_target_network"]:
                next_q_val = self.target_value_network(next_obss)  # SxA
            else:
                next_q_val = self.value_network(next_obss)  # SxA
            if self.solve_options["use_double_estimation"]:
                next_actions = torch.argmax(
                    self.value_network(next_obss), dim=-1)
                next_vval = next_q_val.gather(
                    1, next_actions.reshape(-1, 1)).squeeze()  # S
            else:
                next_vval, _ = torch.max(next_q_val, dim=-1)  # S
            new_q = rews + discount*next_vval*(~dones)
        return new_q.squeeze()  # S

    def compute_policy(self, q_values, eps_greedy=0.0):
        # return epsilon-greedy policy
        return eps_greedy_policy(q_values, eps_greedy=eps_greedy)


class SamplingFittedCviSolver(SamplingFittedSolver):
    def backup(self, tensor_traj):
        discount = self.solve_options["discount"]
        alpha = self.solve_options["alpha"]
        beta = self.solve_options["beta"]

        with torch.no_grad():
            obss = tensor_traj["obs"]
            actions = tensor_traj["act"]
            next_obss = tensor_traj["next_obs"]
            rews = tensor_traj["rew"]
            dones = tensor_traj["done"]
            if self.solve_options["use_target_network"]:
                curr_preference = self.target_value_network(obss)
                next_preference = self.target_value_network(next_obss)  # SxA
                curr_preference2 = self.target_value_network2(obss)
                next_preference2 = self.target_value_network2(next_obss)  # SxA
            else:
                curr_preference = self.value_network(obss)  # SxA
                next_preference = self.value_network(next_obss)  # SxA
                curr_preference2 = self.value_network2(obss)  # SxA
                next_preference2 = self.value_network2(next_obss)  # SxA

            curr_max = self.max_operator(curr_preference, beta)  # S
            next_max = self.max_operator(next_preference, beta)  # S
            if self.solve_options["use_double_estimation"]:
                curr_max2 = self.max_operator(curr_preference2, beta)
                curr_max = torch.min(curr_max, curr_max2)  # S
                next_max2 = self.max_operator(next_preference2, beta)
                next_max = torch.min(next_max, next_max2)  # S

            curr_preference = curr_preference.gather(
                1, actions.reshape(-1, 1)).squeeze()  # S

            new_preference = \
                alpha * (curr_preference.squeeze() - curr_max.squeeze()) \
                + rews + discount*next_max*(~dones)
        return new_preference.squeeze()  # S

    def compute_policy(self, preference):
        # return softmax policy
        beta = self.solve_options["beta"]
        return softmax_policy(preference, beta=beta)
