import io
import PIL.Image
import numpy as np
import gym
import time
import scipy.signal
from functools import update_wrapper
from scipy import special
import matplotlib.pyplot as plt
try:
    import torch
    from torchvision.transforms import ToTensor
    from torch import nn
    from torch.nn import functional as F
    from cpprb import (ReplayBuffer, PrioritizedReplayBuffer,
                       create_env_dict, create_before_add_func)
except ImportError:
    pass


class lazy_property(object):
    r"""
    Stolen from https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)  # type: ignore[arg-type]

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


# -----plot utils-----
def make_image():
    """Save created plot to buffer."""
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close()
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).detach().cpu().numpy()
    return image


# -----math utils-----
def boltzmann_softmax(values, beta):
    # See the appendix A.1 in CVI paper
    if isinstance(values, np.ndarray):
        probs = special.softmax(beta*values, axis=-1)
        return np.sum(probs * values, axis=-1)
    elif isinstance(values, torch.Tensor):
        probs = F.softmax(beta*values, dim=-1)
        return torch.sum(probs * values, dim=-1)


def mellow_max(values, beta):
    # See the CVI paper
    if isinstance(values, np.ndarray):
        x = special.logsumexp(beta*values, axis=-1)
        x += np.log(1/values.shape[-1])
        return x / beta
    elif isinstance(values, torch.Tensor):
        x = torch.logsumexp(beta*values, dim=-1)
        x += np.log(1/values.shape[-1])
        return x / beta


def sigmoid(x):
    return 1 / (1+np.exp(-x))


# -----policy utils-----
def eps_greedy_policy(q_values, eps_greedy=0.0):
    # return epsilon-greedy policy (*)xA
    policy_probs = np.zeros_like(q_values)
    policy_probs[
        np.arange(q_values.shape[0]),
        np.argmax(q_values, axis=-1)] = 1.0 - eps_greedy
    policy_probs += eps_greedy / (policy_probs.shape[-1])
    return policy_probs


def compute_epsilon(step, eps_start, eps_end, eps_decay):
    return eps_end + (eps_start - eps_end) * \
        np.exp(-1. * step / eps_decay)


def softmax_policy(preference, beta=1.0):
    # return softmax policy (*)xA
    policy = special.softmax(beta * preference, axis=-1).astype(np.float64)
    # to avoid numerical instability
    policy /= policy.sum(axis=-1, keepdims=True)
    return policy.astype(np.float32)


# -----trajectory utils-----
def squeeze_trajectory(trajectory):
    for key, value in trajectory.items():
        if trajectory[key].shape[-1] == 1:
            trajectory[key] = np.squeeze(value, axis=-1)
    return trajectory


def trajectory_to_tensor(trajectory, device="cpu", is_discrete=True):
    tensor_traj = {}
    for key, value in trajectory.items():
        if key == "done" or key == "timeout":
            dtype = torch.bool
        elif key in ["state", "next_state"] or (key == "act" and is_discrete):
            dtype = torch.long
        else:
            dtype = torch.float32
        tensor_traj[key] = torch.tensor(value, dtype=dtype, device=device)
    return tensor_traj


def collect_samples(env, policy, num_samples, num_episodes=None, render=False):
    """
    Args:
        env (debug_rl.envs.base.TabularEnv)
        policy (np.nd_array): SxA matrix
        num_samples (int): Number of samples to collect
        num_episodes (int): Number of episodes to collect. Prioritize this if not None.
        render (bool, optional)
    """
    states, next_states, obss, next_obss = [], [], [], []
    actions, rewards, dones, act_prob, timeouts = [], [], [], [], []
    done, done_count = False, 0

    # prioritize num_episodes if not None
    num_samples = -1 if num_episodes is not None else num_samples

    while True:
        if render:
            time.sleep(1/20)
            env.render()
        # add current info
        s = env.get_state()
        states.append(s)
        obss.append(env.observation(s))

        # do action
        probs = policy[s]
        if np.sum(probs) != 1:
            probs /= np.sum(probs)
        action = np.random.choice(np.arange(0, env.dA), p=probs)
        next_obs, rew, done, info = env.step(action)

        # add next info
        ns = env.get_state()
        next_states.append(ns)
        next_obss.append(next_obs)
        rewards.append(rew)
        dones.append(done)
        timeouts.append(not info["TimeLimit.truncated"])
        act_prob.append(probs[action])
        if env.action_mode == "continuous":
            action = env.to_continuous_action(action)
        actions.append(action)

        if done:
            env.reset()
            done_count += 1

        if len(obss) == num_samples or done_count == num_episodes:
            break

    trajectory = {
        "obs": np.array(obss),
        "act": np.array(actions),
        "next_obs": np.array(next_obss),
        "rew": np.array(rewards),
        "done": np.array(dones),
        "state": np.array(states),
        "next_state": np.array(next_states),
        "act_prob": np.array(act_prob),
        "timeout": np.array(timeouts)
    }
    return trajectory


def make_replay_buffer(env, size):
    env_dict = create_env_dict(env)
    env_dict.update({
        "obs": {'dtype': np.float32, 'shape': env_dict["obs"]["shape"]},
        "next_obs": {'dtype': np.float32, 'shape': env_dict["obs"]["shape"]},
        "state": {'dtype': np.int, 'shape': 1},
        "next_state": {'dtype': np.int, 'shape': 1},
        "act_prob": {'dtype': np.float32, 'shape': 1},
        "timeout": {'dtype': np.bool, 'shape': 1},
    })
    return ReplayBuffer(size, env_dict)
