import numpy as np
import pytest
from shinrl.envs import Pendulum


@pytest.fixture
def setUp():
    env = Pendulum(state_disc=32, dA=5, horizon=200)
    env.reset()
    policy = np.random.rand(env.dS, env.dA)
    policy /= policy.sum(axis=1, keepdims=True)
    base_policy = np.random.rand(env.dS, env.dA)
    base_policy /= base_policy.sum(axis=1, keepdims=True)
    yield env, policy, base_policy


def test_all_observations():
    env = Pendulum(state_disc=5, dA=3)
    env.reset()
    obss = env.all_observations
    assert obss.shape == (env.dS, env.observation_space.shape[0])


def test_all_actions():
    env = Pendulum(state_disc=5, dA=3, action_mode="continuous")
    env.reset()
    actions = env.all_actions
    assert actions.shape == (env.dA, 1)


def compute_visitation(env, policy):
    return env.compute_visitation(policy)


def compute_expected_return(env, policy):
    return env.compute_expected_return(policy)


def compute_action_values(env, policy):
    return env.compute_action_values(policy)


def compute_er_action_values(env, policy, base_policy):
    return env.compute_action_values(
        policy, base_policy=base_policy, er_coef=0.1, kl_coef=0.1
    )


def test_compute_visitation(setUp, benchmark):
    env, policy, _ = setUp
    benchmark.pedantic(compute_visitation, kwargs={"env": env, "policy": policy})


def test_compute_expected_return(setUp, benchmark):
    env, policy, _ = setUp
    benchmark.pedantic(compute_expected_return, kwargs={"env": env, "policy": policy})


def test_compute_action_values(setUp, benchmark):
    env, policy, _ = setUp
    benchmark.pedantic(compute_action_values, kwargs={"env": env, "policy": policy})


def test_er_compute_action_values(setUp, benchmark):
    env, policy, base_policy = setUp
    benchmark.pedantic(
        compute_er_action_values,
        kwargs={"env": env, "policy": policy, "base_policy": base_policy},
    )
