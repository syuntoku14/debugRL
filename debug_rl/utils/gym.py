import numpy as np
import gym
import time
from scipy import special
try:
    from cpprb import (ReplayBuffer, PrioritizedReplayBuffer,
                       create_env_dict, create_before_add_func)
except ImportError:
    pass


# -----trajectory utils-----
def collect_samples(env, get_action, num_samples, render=False):
    """
    Args:
        env (debug_rl.envs.base.TabularEnv)
        get_action (function): take env and return action and log_prob
        num_samples (int): Number of samples to collect
        render (bool, optional)
    """
    obss, next_obss = [], []
    actions, rewards, dones, log_probs, timeouts = [], [], [], [], []
    done = False

    if not hasattr(env, 'obs'):
        env.obs = env.reset()

    while True:
        if render:
            time.sleep(1/20)
            env.render()
        # add current info
        obss.append(env.obs)

        # do action
        action, log_prob = get_action(env)
        next_obs, rew, done, info = env.step(action)

        # add next info
        next_obss.append(next_obs)
        rewards.append(rew)
        dones.append(done)
        if "TimeLimit.truncated" in info:
            timeouts.append(info["TimeLimit.truncated"])
        else:
            timeouts.append(False)
        log_probs.append(log_prob)
        actions.append(action)
        env.obs = next_obs

        if done:
            env.obs = env.reset()

        if len(obss) == num_samples:
            break

    trajectory = {
        "obs": np.array(obss),
        "act": np.array(actions),
        "next_obs": np.array(next_obss),
        "rew": np.array(rewards),
        "done": np.array(dones),
        "log_prob": np.array(log_probs),
        "timeout": np.array(timeouts)
    }
    return trajectory


def make_replay_buffer(env, size):
    env_dict = create_env_dict(env)
    env_dict.update({
        "obs": {'dtype': np.float32, 'shape': env_dict["obs"]["shape"]},
        "next_obs": {'dtype': np.float32, 'shape': env_dict["obs"]["shape"]},
        "log_prob": {'dtype': np.float32, 'shape': 1},
        "timeout": {'dtype': np.bool, 'shape': 1},
    })
    return ReplayBuffer(size, env_dict)
