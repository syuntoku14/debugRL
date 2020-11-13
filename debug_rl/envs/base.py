import numpy as np
import gym
from abc import abstractmethod
from scipy import sparse


class TabularEnv(gym.Env):
    def __init__(self,
                 num_states,
                 num_actions,
                 initial_state_distribution,
                 horizon=np.inf):
        self._state = -1
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.num_states = num_states
        self.num_actions = num_actions
        self.initial_state_distribution = initial_state_distribution
        self.horizon = horizon
        self._elapsed_steps = 0
        super().__init__()

    def get_state(self):
        return self._state

    def sample_int(self, transitions):
        next_states = np.array(list(transitions.keys()))
        ps = np.array(list(transitions.values()))
        return np.random.choice(next_states, p=ps)

    def step(self, action):
        """Simulates the environment by one timestep.

        Args:
          action: Action to take

        Returns:
          observation: Next observation
          reward: Reward incurred by agent
          done: A boolean indicating the end of an episode
          info: A debug info dictionary.
        """
        infos = {'state': self._state}
        transitions = self.transitions(self._state, action)
        next_state = self.sample_int(transitions)
        reward = self.reward(self._state, action)
        done = False
        self._state = next_state
        nobs = self.observation(self._state)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.horizon:
            done = True
        return nobs, reward, done, infos

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
          observation (object): The agent's initial observation.
        """

        self._elapsed_steps = 0
        initial_state = self.sample_int(self.initial_state_distribution)
        self._state = initial_state
        return self.observation(initial_state)

    def transition_matrix(self):
        """Constructs this environment's transition matrix.

        Returns:
          A (dS x dA) x dS array where the entry transition_matrix[sa, ns]
          corresponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        ds = self.num_states
        da = self.num_actions
        row = []
        col = []
        data = []
        for s in range(ds):
            for a in range(da):
                transitions = self.transitions(s, a)
                for next_s, prob in transitions.items():
                    row.append(da * s + a)
                    col.append(next_s)
                    data.append(prob)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        return sparse.csr_matrix((data, (row, col)), shape=(ds*da, ds))

    def reward_matrix(self):
        """Constructs this environment's reward matrix.

        Returns:
          A (dS x dA) x 1 scipy.sparse array where the entry reward_matrix[sa, 1]
          reward given to an agent when transitioning into state ns after taking
          action s from state s.
        """
        ds = self.num_states
        da = self.num_actions
        row = []
        col = []
        data = []
        for s in range(ds):
            for a in range(da):
                rew = self.reward(s, a)
                row.append(da*s + a)
                col.append(0)
                data.append(rew)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        return sparse.csr_matrix((data, (row, col)), shape=(ds*da, 1))

    def render(self):
        pass

    @abstractmethod
    def transitions(self, state, action):
        """Computes transition probabilities p(ns|s,a).

        Args:
          state:
          action:

        Returns:
          A python dict from {next state: probability}.
          (Omitted states have probability 0)
        """

    @abstractmethod
    def reward(self, state, action):
        """Return the reward

        Args:
          state:
          action:

        Returns:
          float: reward
        """

    @abstractmethod
    def observation(self, state):
        """Computes observation for a given state.

        Args:
          state:

        Returns:
          observation: Agent's observation of state, conforming with observation_space
        """

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def to_continuous_action(self, *args, **kwargs):
        return None
