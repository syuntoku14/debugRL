import numpy as np
import gym
from abc import abstractmethod
from scipy import sparse
from debug_rl.utils import lazy_property


class TabularEnv(gym.Env):
    def __init__(self,
                 dS,
                 dA,
                 initial_state_distribution,
                 horizon=np.inf):
        self._state = -1
        self.observation_space = gym.spaces.Discrete(dS)
        self.action_space = gym.spaces.Discrete(dA)
        self.dS = dS
        self.dA = dA
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
        transitions = self.transitions(self._state, action)
        next_state = self.sample_int(transitions)
        reward = self.reward(self._state, action)
        done = False
        self._state = next_state
        nobs = self.observation(self._state)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.horizon:
            done = True
        infos = {'state': self._state, 'TimeLimit.truncated': False}
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

    def to_continuous_action(self, *args, **kwargs):
        return None

    def render(self):
        pass

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

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

    @lazy_property
    def all_actions(self):
        """
        Returns:
            A x 1 vector
        """
        acts = []
        for a in range(self.dA):
            a = self.to_continuous_action(a)
            acts.append(np.expand_dims(a, axis=0))
        acts = np.vstack(acts)  # A x 1
        return acts

    @lazy_property
    def transition_matrix(self):
        """Constructs this environment's transition matrix.

        Returns:
          A (dS x dA) x dS array where the entry transition_matrix[sa, ns]
          corresponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        ds = self.dS
        da = self.dA
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

    @lazy_property
    def reward_matrix(self):
        """Constructs this environment's reward matrix.

        Returns:
          A dS x dA scipy.sparse array where the entry reward_matrix[s, a]
          reward given to an agent when transitioning into state ns after taking
          action a from state s.
        """
        ds = self.dS
        da = self.dA
        row = []
        col = []
        data = []
        for s in range(ds):
            for a in range(da):
                rew = self.reward(s, a)
                row.append(s)
                col.append(a)
                data.append(rew)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        return sparse.csr_matrix((data, (row, col)), shape=(ds, da))

    def compute_action_values(self, policy, discount=0.99, base_policy=None,
                              er_coef=None, kl_coef=None):
        """
        Compute action values of given policy.
        Args:
            policy, base_policy: SxA matrix
            er_coef: entropy regularization coefficient
            kl_coef: KL regularization coefficient
        Returns:
            Q values: SxA matrix
        """

        values = np.zeros((self.dS, self.dA))  # SxA
        reward_matrix = self.reward_matrix  # SxA

        if er_coef is not None:
            entropy = -np.sum(policy * np.log(policy+1e-8),
                              axis=-1, keepdims=True)  # Sx1
            reward_matrix = reward_matrix + er_coef*entropy  # SxA
        if base_policy is not None:
            kl = np.sum(policy * (np.log(policy+1e-8)-np.log(base_policy+1e-8)),
                        axis=-1, keepdims=True)  # Sx1
            reward_matrix = reward_matrix - kl_coef*kl  # SxA

        def backup(curr_q_val, policy):
            curr_v_val = np.sum(policy * curr_q_val, axis=-1)  # S
            prev_q = reward_matrix \
                + discount*(self.transition_matrix *
                            curr_v_val).reshape(self.dS, self.dA)
            prev_q = np.asarray(prev_q)
            return prev_q

        for _ in range(self.horizon):
            values = backup(values, policy)  # SxA
        return values

    def compute_visitation(self, policy, discount=1.0):
        """
        Compute state and action frequency on the given policy
        Args:
            policy: SxA matrix
            discount (float): discount factor.

        Returns:
            visitation: TxSxA matrix
        """
        sa_visit = np.zeros((self.horizon, self.dS, self.dA))  # SxAxS
        s_visit_t = np.zeros((self.dS, 1))  # Sx1
        for (state, prob) in self.initial_state_distribution.items():
            s_visit_t[state] = prob

        norm_factor = 0.0
        for t in range(self.horizon):
            cur_discount = (discount ** t)
            norm_factor += cur_discount
            sa_visit_t = s_visit_t * policy  # SxA
            sa_visit[t, :, :] = cur_discount * sa_visit_t
            # sum-out (SA)S
            new_s_visit_t = \
                sa_visit_t.reshape(self.dS*self.dA) * self.transition_matrix
            s_visit_t = np.expand_dims(new_s_visit_t, axis=1)
        visitation = sa_visit / norm_factor
        return visitation

    def compute_expected_return(self, policy):
        """
        Compute expected return of the given policy

        Args:
            policy: SxA matrix

        Returns:
            expected_return (float)
        """

        q_values = np.zeros((self.dS, self.dA))  # SxA
        for t in reversed(range(1, self.horizon+1)):
            # backup q values
            curr_vval = np.sum(policy * q_values, axis=-1)  # S
            prev_q = (self.transition_matrix *
                      curr_vval).reshape(self.dS, self.dA)  # SxA
            q_values = np.asarray(prev_q + self.reward_matrix)  # SxA

        init_vval = np.sum(policy*q_values, axis=-1)  # S
        init_probs = np.zeros(self.dS)  # S
        for (state, prob) in self.initial_state_distribution.items():
            init_probs[state] = prob
        expected_return = np.sum(init_probs * init_vval)
        return expected_return
