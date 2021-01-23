import numpy as np
from debug_rl.envs.base import TabularEnv
from .grid_env import GridEnv
from gym.spaces import Box
from gym import ObservationWrapper
from debug_rl.utils import lazy_property


def flat_to_one_hot(val, ndim):
    """

    >>> flat_to_one_hot(2, ndim=4)
    array([ 0.,  0.,  1.,  0.])
    >>> flat_to_one_hot(4, ndim=5)
    array([ 0.,  0.,  0.,  0.,  1.])
    >>> flat_to_one_hot(np.array([2, 4, 3]), ndim=5)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  1.,  0.]])
    """
    shape = np.array(val).shape
    v = np.zeros(shape + (ndim,))
    if len(shape) == 1:
        v[np.arange(shape[0]), val] = 1.0
    else:
        v[val] = 1.0
    return v


def one_hot_to_flat(val):
    """
    >>> one_hot_to_flat(np.array([0,0,0,0,1]))
    4
    >>> one_hot_to_flat(np.array([0,0,1,0]))
    2
    >>> one_hot_to_flat(np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0]]))
    array([2, 0, 1])
    """
    idxs = np.array(np.where(val == 1.0))[-1]
    if len(val.shape) == 1:
        return int(idxs)
    return idxs


class OneHotObsWrapper(ObservationWrapper):
    """
    A wrapper converting the discrete observation to one-hot observation.
    """

    def __init__(self, env):
        assert isinstance(env, GridEnv)
        super().__init__(env)
        self.gs = env.gs
        self.dS = env.dS
        self.dO = self.gs.width+self.gs.height
        self.observation_space = Box(0, 1, (self.dO, ))
        self.horizon = env.horizon

    def observation(self, obs):
        xy = self.gs.idx_to_xy(obs)
        x = flat_to_one_hot(xy[0], self.gs.width)
        y = flat_to_one_hot(xy[1], self.gs.height)
        obs = np.hstack([x, y])
        return obs

    @lazy_property
    def all_observations(self):
        """
        Returns:
            S x obs_dim matrix: Observations of the all states.
        """
        obss = []
        for s in range(self.dS):
            s = self.observation(s)
            obss.append(np.expand_dims(s, axis=0))
        obss = np.vstack(obss)  # S x obs_dim
        return obss

    def get_state(self):
        return self.env.get_state()

    @property
    def elapsed_steps(self):
        return self._elapsed_steps


class RandomObsWrapper(ObservationWrapper):
    """
    A wrapper converting the discrete observation to random-vector observation.
    """

    def __init__(self, env, obs_dim):
        assert isinstance(env, GridEnv)
        super().__init__(env)
        self.gs = env.gs
        self.dS = env.dS
        self.obs_dim = obs_dim
        self.obs_matrix = np.random.randn(len(self.gs), self.obs_dim)
        self.observation_space = Box(np.min(self.obs_matrix), np.max(self.obs_matrix),
                                     shape=(self.obs_dim, ), dtype=np.float32)
        self.horizon = env.horizon

    def observation(self, obs):
        return self.obs_matrix[obs]

    @lazy_property
    def all_observations(self):
        """
        Returns:
            S x obs_dim matrix: Observations of the all states.
        """
        obss = []
        for s in range(self.dS):
            s = self.observation(s)
            obss.append(np.expand_dims(s, axis=0))
        obss = np.vstack(obss)  # S x obs_dim
        return obss

    def get_state(self):
        return self.env.get_state()

    @property
    def elapsed_steps(self):
        return self._elapsed_steps
