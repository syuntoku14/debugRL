import numpy as np
from functools import update_wrapper
from scipy import special
try:
    import torch
    from torchvision.transforms import ToTensor
    from torch import nn
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

    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)  # type: ignore[arg-type]

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


# -----trajectory utils-----
def trajectory_to_tensor(trajectory, device="cpu", is_discrete=True):
    tensor_traj = {}
    for key, value in trajectory.items():
        if trajectory[key].shape[-1] == 1:
            trajectory[key] = np.squeeze(value, axis=-1)
        if key == "done" or key == "timeout":
            dtype = torch.bool
        elif key in ["state", "next_state"] or (key == "act" and is_discrete):
            dtype = torch.long
        else:
            dtype = torch.float32
        tensor_traj[key] = torch.tensor(value, dtype=dtype, device=device)
    return tensor_traj


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
