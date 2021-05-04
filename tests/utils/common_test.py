import numpy as np
import pytest
import torch

from shinrl import utils
from shinrl.envs import Pendulum


def test_policy():
    value = np.random.rand(3, 4)
    assert utils.eps_greedy_policy(value).shape == value.shape
    assert utils.softmax_policy(value).shape == value.shape


def test_max_operator():
    # check shape
    beta = 0.9
    sample = np.random.rand(2, 3, 4)
    assert utils.boltzmann_softmax(sample, beta=beta).shape == sample.shape[:-1]
    assert utils.mellow_max(sample, beta=beta).shape == sample.shape[:-1]
    sample = torch.randn((2, 3, 4))
    assert utils.boltzmann_softmax(sample, beta=beta).shape == sample.shape[:-1]
    assert utils.mellow_max(sample, beta=beta).shape == sample.shape[:-1]

    # value check
    sample = np.array([2, 3])
    res = utils.boltzmann_softmax(sample, beta=beta)
    np.testing.assert_almost_equal(
        res,
        np.sum(np.exp(beta * sample) * sample, axis=-1)
        / np.sum(np.exp(beta * sample), axis=-1),
    )

    res = utils.mellow_max(sample, beta=beta)
    np.testing.assert_almost_equal(
        res, np.log(np.mean(np.exp(beta * sample), axis=-1)) / beta
    )
