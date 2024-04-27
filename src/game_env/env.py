import gymnasium as gym
from gymnasium.spaces import MultiBinary, Tuple, Dict, Discrete, MultiDiscrete


class RegicideEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = Tuple(MultiBinary(7), MultiBinary(7))
        self.observation_space = Dict(
            {
                "curr_enemy": Discrete(12),
                "enemies_left": Discrete(11),
                "enemy_health": Discrete(21, start=0),
                "enemy_attack": Discrete(21, start=0),
                "num_discard": Discrete(53, start=0),
                "num_tavern": Discrete(53, start=0),
                "ally_number_cards": Discrete(8, start=0),
                "player_hand": MultiDiscrete([53] * 7, start=[0] * 7),
            }
        )

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
