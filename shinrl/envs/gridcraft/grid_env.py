import sys
import numpy as np
from .grid_spec import REWARD, REWARD2, REWARD3, REWARD4, WALL, LAVA, TILES, START, RENDER_DICT
from shinrl.envs import TabularEnv
from .plotter import plot_values


ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_DICT = {
    ACT_NOOP: [0, 0],
    ACT_UP: [0, -1],
    ACT_LEFT: [-1, 0],
    ACT_RIGHT: [+1, 0],
    ACT_DOWN: [0, +1]
}
ACT_TO_STR = {
    ACT_NOOP: 'NOOP',
    ACT_UP: 'UP',
    ACT_LEFT: 'LEFT',
    ACT_RIGHT: 'RIGHT',
    ACT_DOWN: 'DOWN'
}


class TransitionModel(object):
    def __init__(self, gridspec, eps=0.2):
        self.gs = gridspec
        self.eps = eps

    def get_aprobs(self, s, a):
        # TODO: could probably output a matrix over all states...
        legal_moves = self._get_legal_moves(s)
        p = np.zeros(len(ACT_DICT))
        p[legal_moves] = self.eps / (len(legal_moves))
        if a in legal_moves:
            p[a] += 1.0-self.eps
        else:
            # p = np.array([1.0,0,0,0,0])  # NOOP
            p[ACT_NOOP] += 1.0-self.eps
        return p

    def _get_legal_moves(self, s):
        xy = np.array(self.gs.idx_to_xy(s))
        moves = [move for move in ACT_DICT if not self.gs.out_of_bounds(xy+ACT_DICT[move])
                 and self.gs[xy+ACT_DICT[move]] != WALL]
        return moves


class RewardFunction(object):
    def __init__(self, default=0):
        rew_map = {
            REWARD: 1.0,
            REWARD2: 2.0,
            REWARD3: 4.0,
            REWARD4: 8.0,
            LAVA: -1.0,
        }
        self.default = default
        self.rew_map = rew_map

    def __call__(self, gridspec, s, a):
        val = gridspec[gridspec.idx_to_xy(s)]
        if val in self.rew_map:
            return self.rew_map[val]
        return self.default


class GridEnv(TabularEnv):
    def __init__(self, gridspec,
                 horizon=20,
                 tiles=TILES,
                 trans_eps=0.0,
                 default_rew=0):
        self.gs = gridspec
        self.model = TransitionModel(gridspec, eps=trans_eps)
        self.rew_fn = RewardFunction(default=default_rew)
        self.possible_tiles = tiles
        self.action_mode = "discrete"

        # compute initial_state_distribution
        start_idxs = np.array(np.where(self.gs.spec == START)).T
        num_starts = start_idxs.shape[0]
        initial_distribution = {}
        for i in range(num_starts):
            initial_distribution[self.gs.xy_to_idx(
                start_idxs[i])] = 1.0/num_starts

        super().__init__(
            dS=len(gridspec),
            dA=5,
            initial_state_distribution=initial_distribution,
            horizon=horizon
        )

    def transitions(self, s, a):
        tile_type = self.gs[self.gs.idx_to_xy(s)]
        if tile_type == LAVA:  # Lava gets you stuck
            return {s: 1.0}

        aprobs = self.model.get_aprobs(s, a)
        t_dict = {}
        for sa in range(5):
            if aprobs[sa] > 0:
                next_s = self.gs.idx_to_xy(s) + ACT_DICT[sa]
                next_s_idx = self.gs.xy_to_idx(next_s)
                t_dict[next_s_idx] = t_dict.get(next_s_idx, 0.0) + aprobs[sa]
        return t_dict

    def reward(self, s, a):
        return self.rew_fn(self.gs, s, a)

    def render(self, ostream=sys.stdout):
        state = self.__state
        ostream.write('-'*(self.gs.width+2)+'\n')
        for h in range(self.gs.height):
            ostream.write('|')
            for w in range(self.gs.width):
                if self.gs.xy_to_idx((w, h)) == state:
                    ostream.write('*')
                else:
                    val = self.gs[w, h]
                    ostream.write(RENDER_DICT[val])
            ostream.write('|\n')
        ostream.write('-' * (self.gs.width + 2)+'\n')

    def observation(self, s):
        return s

    def plot_values(
            self, values, title=None, ax=None, **kwargs):
        plot_values(self.gs, values, title=title, ax=ax)
