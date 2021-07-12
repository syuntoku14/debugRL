import time
from collections import defaultdict
from functools import update_wrapper

import numpy as np
from cpprb import ReplayBuffer, create_env_dict
from scipy import special

try:
    import torch
    from torch.nn import functional as F
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

    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        update_wrapper(self, wrapped)  # type: ignore[arg-type]

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


# -----math utils-----
def boltzmann_softmax(values, beta):
    # See the appendix A.1 in CVI paper
    if isinstance(values, np.ndarray):
        probs = special.softmax(beta * values, axis=-1)
        return np.sum(probs * values, axis=-1)
    elif isinstance(values, torch.Tensor):
        probs = F.softmax(beta * values, dim=-1)
        return torch.sum(probs * values, dim=-1)


def mellow_max(values, beta):
    # See the CVI paper
    if isinstance(values, np.ndarray):
        x = special.logsumexp(beta * values, axis=-1)
        x += np.log(1 / values.shape[-1])
        return x / beta
    elif isinstance(values, torch.Tensor):
        x = torch.logsumexp(beta * values, dim=-1)
        x += np.log(1 / values.shape[-1])
        return x / beta


# -----policy utils-----
def eps_greedy_policy(q_values, eps_greedy=0.0):
    # return epsilon-greedy policy (*)xA
    policy_probs = np.zeros_like(q_values)
    policy_probs[np.arange(q_values.shape[0]), np.argmax(q_values, axis=-1)] = (
        1.0 - eps_greedy
    )
    policy_probs += eps_greedy / (policy_probs.shape[-1])
    return policy_probs


def compute_epsilon(step, decay_period, warmup_steps, eps_end):
    """Copied from dopamine(https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/dqn_agent.py#L43).
    Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
      Begin at 1. until warmup_steps steps have been taken; then
      Linearly decay epsilon from 1. to eps_end in decay_period steps; and then
      Use eps_end from there on.
    Args:
      decay_period: float, the period over which epsilon is decayed.
      step: int, the number of training steps completed so far.
      warmup_steps: int, the number of steps taken before epsilon is decayed.
      eps_end: float, the final value to which to decay the epsilon parameter.
    Returns:
      A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - eps_end) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - eps_end)
    return eps_end + bonus


def softmax_policy(preference, beta=1.0):
    # return softmax policy (*)xA
    policy = special.softmax(beta * preference, axis=-1).astype(np.float64)
    # to avoid numerical instability
    policy /= policy.sum(axis=-1, keepdims=True)
    return policy.astype(np.float32)


def get_tb_action(env, policy):
    # policy: dS x dA
    probs = policy[env.get_state()]
    if np.sum(probs) != 1:
        probs /= np.sum(probs)
    act = np.random.choice(np.arange(0, env.dA), p=probs)
    return act, np.log(probs[act])


# -----trajectory utils-----
def trajectory_to_tensor(trajectory, device="cpu"):
    tensor_traj = {}
    for key, value in trajectory.items():
        if trajectory[key].shape[-1] == 1:
            value = np.squeeze(value, axis=-1)
        if key == "done" or key == "timeout":
            dtype = torch.bool
        elif (
            key in ["state", "next_state"]
            or trajectory[key].dtype == np.int32
            or trajectory[key].dtype == np.int64
        ):
            dtype = torch.long
        else:
            dtype = torch.float32
        tensor_traj[key] = torch.tensor(value, dtype=dtype, device=device)
    return tensor_traj


def collect_samples(
    env,
    get_action,
    num_samples=None,
    num_episodes=None,
    render=False,
    buffer=None,
    **kwargs,
):
    """
    Args:
        env (gym.Env or shinrl.envs.TabularEnv)
        get_action (function): take env and return action and log_prob
        num_samples (int): Number of samples to collect
        num_episodes (int): Number of episodes to collect. Prioritize this if not None.
        render (bool, optional)
        buffer (cpprb.ReplayBuffer): For calling on_episode_end. Do buffer.add if not None.
    """
    is_tabular = hasattr(env, "transition_matrix")
    assert hasattr(
        env, "obs"
    ), 'env has no attribute "obs". Run env.obs = env.reset() before collect_samples.'
    samples = defaultdict(lambda: [])
    done, done_count, step_count = False, 0, 0

    # prioritize num_episodes if not None
    num_samples = -1 if num_episodes is not None else num_samples

    while True:
        if render:
            time.sleep(1 / 20)
            env.render()
        sample = {}
        # add current info
        if is_tabular:
            s = env.get_state()
            obs = env.observation(s)
        else:
            obs = env.obs
        sample["obs"] = obs

        # do one step
        action, log_prob = get_action(env, **kwargs)
        next_obs, rew, done, info = env.step(action)
        env.obs = next_obs
        if is_tabular and env.action_mode == "continuous":
            action = env.to_continuous_action(action)
        step_count += 1

        # add next info
        timeout = (
            info["TimeLimit.truncated"] if "TimeLimit.truncated" in info else False
        )
        sample.update(
            {
                "next_obs": next_obs,
                "rew": rew,
                "done": done,
                "log_prob": log_prob,
                "act": action,
                "timeout": timeout,
            }
        )
        if is_tabular:
            sample.update({"state": s, "next_state": env.get_state()})

        if buffer is not None:
            buffer.add(**sample)

        if done:
            env.obs = env.reset()
            done_count += 1
            if buffer is not None:
                buffer.on_episode_end()

        for key, val in sample.items():
            samples[key].append(val)
        if step_count == num_samples or done_count == num_episodes:
            break
    return {key: np.array(val) for key, val in samples.items()}


def make_replay_buffer(env, size):
    is_tabular = hasattr(env, "transition_matrix")
    env_dict = create_env_dict(env)
    del env_dict["next_obs"]
    env_dict.update(
        {
            "obs": {"dtype": np.float32, "shape": env_dict["obs"]["shape"]},
            "log_prob": {"dtype": np.float32, "shape": 1},
            "timeout": {"dtype": np.bool, "shape": 1},
        }
    )
    if is_tabular:
        env_dict.update({"state": {"dtype": np.int32, "shape": 1}})
        return ReplayBuffer(size, env_dict, next_of=("obs", "state"))
    return ReplayBuffer(size, env_dict, next_of=("obs",))
