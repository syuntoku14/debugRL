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
def collect_samples(env, policy, num_samples, render=False):
    """
    Args:
        env (debug_rl.envs.base.TabularEnv)
        policy (np.nd_array): SxA matrix
        num_samples (int): Number of samples to collect
        render (bool, optional)
    """
    states, next_states, obss, next_obss = [], [], [], []
    actions, rewards, dones, log_probs, timeouts = [], [], [], [], []
    done = False

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
        if "TimeLimit.truncated" in info:
            timeouts.append(info["TimeLimit.truncated"])
        else:
            timeouts.append(False)
        log_probs.append(np.log(probs[action]))
        if env.action_mode == "continuous":
            action = env.to_continuous_action(action)
        actions.append(action)

        if done:
            env.reset()

        if len(obss) == num_samples:
            break

    trajectory = {
        "obs": np.array(obss),
        "act": np.array(actions),
        "next_obs": np.array(next_obss),
        "rew": np.array(rewards),
        "done": np.array(dones),
        "state": np.array(states),
        "next_state": np.array(next_states),
        "log_prob": np.array(log_probs),
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
        "log_prob": {'dtype': np.float32, 'shape': 1},
        "timeout": {'dtype': np.bool, 'shape': 1},
    })
    return ReplayBuffer(size, env_dict)
