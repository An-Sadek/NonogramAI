import math

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


def create_grid(size: int, seed) -> np.ndarray:
    return np.random.randint(0, 2, size=(size, size))

def extract_clues(matrix: np.ndarray):
    """
    Trong truong hop muon lay goi y col thi chu viec chuyen vi ma tran
    """
    row_clues = []
    size = matrix.shape[0]

    for row in matrix:
        clue = []
        count = 0  
        
        for col in range(size):
            if row[col] == 1:
                count += 1 
            else:
                if count > 0:  
                    clue.append(count)
                    count = 0  
        
        if count > 0: 
            clue.append(count)

        # Them 0 vao cho cung kich thuoc
        while len(clue) < math.ceil(size / 2):
            clue.append(0)
        
        row_clues.append(clue)

    return np.array(row_clues, dtype=int)

class Actions(Enum):
    """
    Thay vi doc ca ma tran, thi co the thiet lap cho no di chuyen.
    Dung yen de no co the suy nghi ra hanh dong khac.
    Di chuyen den cac o khac va dat 0 tuong ung voi x, 1 tuong ung voi o vuong.
    """
    stay = 0
    goto_x = 1
    goto_y = 2
    place_0 = 3
    place_1 = 4

class NonogramEnv(gym.Env):
    metadata = {"render_mode": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, lives=3, render_mode = None, size=5, seed=0):
        self.size = size
        self.window_size = 512

        # So lan thu
        self.MAX_LIFE = lives
        self.curr_lives = lives

        # Truong hop nhieu gio y nhat chi co n/2
        max_clue = math.ceil(size / 2) 
            
        self.observation_space = spaces.Dict(
            {
                "game_grid": spaces.Box(-1, 2, shape=(size, size), dtype=int),
                "row_clues": spaces.Box(0, size + 1, shape=(size, max_clue), dtype=int),
                "col_clues": spaces.Box(0, size + 1, shape=(size, max_clue), dtype=int)
            }
        )

        self.game_grid = np.zeros(shape=(size, size), dtype=int) - 1
        self.result = create_grid(size, seed)
        self.row_clues = extract_clues(self.result)
        self.col_clues = extract_clues(self.result.T)        

        # So luong hanh dong
        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {
            "game_grid": self.game_grid, 
            "row_clues": self.row_clues, 
            "col_clues": self.col_clues
        }

    def _get_info(self):
        return {
            "life_left": self.curr_lives,
            "completion": np.sum((self.game_grid == self.result)) / (self.size*self.size),
            "correct": self.game_grid == self.result
        }

    def reset(self, seed=None, options=None):
        """
        Reset lai trang thai ban dau
        """
        self.lives = self.MAX_LIFE
        
        self.game_grid = np.zeros(shape=(self.size, self.size), dtype=int) - 1
        self.result = create_grid(self.size, seed=seed)
        self.row_clues = extract_clues(self.result)
        self.col_clues = extract_clues(self.result.T)  

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

if __name__ == "__main__":
    print("Test ham ngoai")
    seed = 0
    mtx = create_grid(5, seed)
    
    print(mtx)
    r_clues = extract_clues(mtx)
    c_clues = extract_clues(mtx.T)
    print(r_clues)
    print(c_clues)

    print("\n\nTest moi truong")
    env = NonogramEnv()
    print("\nMa tran khoi tao")
    print(env.game_grid)
    
    print("\nKet qua thuc te")
    print(env.result)

    print("\nGoi y hang")
    print(env.row_clues)

    print("\nGoi y cot")
    print(env.col_clues)

    print("\nTest reset")
    print(env.reset())
