import numpy as np
import gym
import cv2
from debug_rl.envs.base import TabularEnv


class MountainCar(TabularEnv):
    """ 
    Dynamics and reward are based on OpenAI gym's implementation of MountainCar-v0
    obs_mode:
        - tuple: Return tuple of (pos, vel)
        - image: Return simple image of mountaincar
    """

    def __init__(self, posdisc=32, veldisc=32, init_dist=None,
                 horizon=200, num_actions=3,
                 action_mode="discrete", obs_mode="tuple"):
        # env parameters
        self.pos_disc = posdisc
        self.vel_disc = veldisc
        self.num_actions = num_actions
        self.max_vel = 0.07
        self.min_vel = -self.max_vel
        self.max_pos = 0.6
        self.min_pos = -1.2
        self.goal_pos = 0.5
        self.force_mag = 1.0
        self.force_list = np.linspace(-self.force_mag, self.force_mag, num=self.num_actions)

        self.action_step = (self.force_mag*2) / self.num_actions
        self.state_step = (self.max_pos-self.min_pos) / (self.pos_disc-1)
        self.vel_step = (self.max_vel-self.min_vel)/(self.vel_disc-1)

        # for rendering
        gym.make("MountainCar-v0")
        self.render_env = gym.envs.classic_control.MountainCarEnv()
        self.render_env.reset()

        # observation
        if obs_mode != "tuple" and obs_mode != "image":
            raise ValueError("Invalid obs_mode: {}".format(obs_mode))
        self.obs_mode = obs_mode
        self.base_image = np.zeros((28, 28))

        # set initial state probability
        ini_poss = np.arange(-0.6, -0.4, self.state_step)
        idxs = []
        for ini_pos in ini_poss:
            idxs.append(self.to_state_id(ini_pos, 0.0))
        idxs = set(idxs)
        random_init = {idx: 1/len(idxs) for idx in idxs}
        init_dist = random_init if init_dist is None else init_dist
        super().__init__(posdisc*veldisc, self.num_actions, init_dist, horizon=horizon)

        self.action_mode = action_mode
        if action_mode == "discrete":
            self.action_space = gym.spaces.Discrete(num_actions)
        elif action_mode == "continuous":
            self.action_space = gym.spaces.Box(
                low=np.array((-self.force_mag, )), high=np.array((self.force_mag, )))
        else:
            raise ValueError

        if obs_mode == "tuple":
            self.observation_space = gym.spaces.Box(
                low=np.array([self.min_pos, -self.max_vel]),
                high=np.array([self.max_pos, self.max_vel]),
                dtype=np.float32)
        elif obs_mode == "image":
            self.observation_space = gym.spaces.Box(
                low=np.expand_dims(np.zeros((28, 28)), axis=0),
                high=np.expand_dims(np.ones((28, 28)), axis=0),
                dtype=np.float32)

    def discretize_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high-1e-5)
        return int(np.floor((action - self.action_space.low)/self.action_step))

    def to_continuous_action(self, action):
        return action * self.action_step + self.action_space.low

    def transitions(self, state, action):
        force = self.force_list[action]
        transitions = {}
        pos, vel = self.pos_vel_from_state_id(state)
        for _ in range(5):  # one step is not enough when state is discretized
            vel += force*0.001 + np.cos(3*pos)*(-0.0025)
            vel = np.clip(vel, self.min_vel, self.max_vel)
            pos += vel
            pos = np.clip(pos, self.min_pos, self.max_pos)
        if (pos == self.min_pos and vel < 0):
            vel = 0
        next_state = self.to_state_id(pos, vel)
        transitions[next_state] = 1.0
        return transitions

    def reward(self, state, action):
        if self.action_mode == "continuous":
            action = self.discretize_action(action)
        pos, vel = self.pos_vel_from_state_id(state)
        if pos >= self.goal_pos:
            return 0.0
        return -1.0

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .75

    def observation(self, state):
        pos, vel = self.pos_vel_from_state_id(state)
        if self.obs_mode == "tuple":
            return np.array([pos, vel], dtype=np.float32)
        elif self.obs_mode == "image":
            image = self.base_image.copy()
            pos2pxl = 28 / (self.max_pos - self.min_pos) 
            x = int((pos - self.min_pos) * pos2pxl)
            y = int(self._height(pos - self.min_pos) * pos2pxl)
            image = cv2.rectangle(image, (x, y), (x+1, y+1), 0.8, thickness=2)

            vx = int((pos-vel*5.0 - self.min_pos) * pos2pxl)
            vy = int(self._height(pos-vel*5.0 - self.min_pos) * pos2pxl)
            image = cv2.rectangle(image, (vx, vy), (vx+1, vy+1), 0.2, thickness=2)
            return np.expand_dims(image, axis=0)  # 1x28x28

    def pos_vel_from_state_id(self, state):
        pos_idx = state % self.pos_disc
        vel_idx = state // self.pos_disc
        pos = self.min_pos + self.state_step * pos_idx
        vel = self.min_vel + self.vel_step * vel_idx
        return pos, vel

    def to_state_id(self, pos, vel):
        # round
        pos_idx = int(np.floor((pos-self.min_pos)/self.state_step))
        vel_idx = int(np.floor((vel-self.min_vel)/self.vel_step))
        return pos_idx + self.pos_disc * vel_idx

    def render(self):
        state = self.get_state()
        pos, vel = self.pos_vel_from_state_id(state)
        self.render_env.state = (pos, vel)
        self.render_env.render()

    def close(self):
        self.render_env.close()
