import pytest
import numpy as np
import torch
from rlowan.envs.pendulum import Pendulum
from rlowan.utils import *


def test_policy():
    value = np.random.rand(3, 4)
    assert eps_greedy_policy(value).shape == value.shape
    assert softmax_policy(value).shape == value.shape


def test_max_operator():
    # check shape
    beta = 0.9
    sample = np.random.rand(2, 3, 4)
    assert boltzmann_softmax(sample, beta=beta).shape == sample.shape[:-1]
    assert mellow_max(sample, beta=beta).shape == sample.shape[:-1]
    sample = torch.randn((2, 3, 4))
    assert boltzmann_softmax(sample, beta=beta).shape == sample.shape[:-1]
    assert mellow_max(sample, beta=beta).shape == sample.shape[:-1]

    # value check
    sample = np.array([2, 3])
    res = boltzmann_softmax(sample, beta=beta)
    np.testing.assert_almost_equal(
        res, np.sum(np.exp(beta * sample) * sample, axis=-1)
        / np.sum(np.exp(beta * sample), axis=-1))

    res = mellow_max(sample, beta=beta)
    np.testing.assert_almost_equal(
        res, np.log(np.mean(np.exp(beta*sample), axis=-1)) / beta)
