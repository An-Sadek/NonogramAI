import math

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    """
    Thay vi doc ca ma tran, thi co the thiet lap cho no di chuyen.
    Dung yen de no co the suy nghi ra hanh dong khac.
    Di chuyen den cac o khac va dat 0 tuong ung voi x, 1 tuong ung voi o vuong.
    """
    stay = 0 
    up = 1
    down = 2
    left = 3
    right = 4
    place_0 = 5 # Dat dau x
    place_1 = 6 # Dat o vuong
    

class NonogramEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, seed=42):
        self.size = size
        self.window_size = 512

        self.self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                
                # Goi y se co gia tri tu 1 -> size (do xac suat 0.5), nhung de chac chan thi quan sat bat dau tu 0
                "row_clues": spaces.Box(0, size, shape=(size, math.ceil(size/2))),
                "col_clues": spaces.Box(0, size, shape=(size, math.ceil(size/2)))
            }
        )

        # Cac gia tri anh huong den tac nhan
        self._agent_location = np.array([-1, -1], dtype=int)
        self.action_space = spaces.Discrete(7)
        self.row_clues = spaces.Box(0, size, shape=(size, math.ceil(size/2)))
        self.col_clues = spaces.Box(0, size, shape=(size, math.ceil(size/2)))

        # Cong tru toa do theo huong di chuyen
        self._action_to_direction = {
            Actions.stay.value: np.array([0, 0]),
            Actions.up.value: np.array{[0, 1]},
            Actions.down.value: np.array([0, -1]),
            Actions.right.value: np.array([0, 1]),
            Actions.left.value: np.array([0, -1])
        }

        # Neu render_mode co "human" thi moi su dung window va clock
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

        