import math
import time

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


def create_grid(size: int, seed=0) -> np.ndarray:
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
    goto_x = 0
    goto_y = 1
    place_0 = 2
    place_1 = 3


class NonogramEnv(gym.Env):
    """
    De co the thuong cho con bot lam it sai hon, va lam nhanh hon thi can chuan hoa mang va thoi gian
    Cach de chuan hoa la ve [0, 1] * n voi n la he_so_phan_thuong

    Gia su chi co 5 mang thi khi hoan thanh:
        Neu con 5 mang, thi so diem se la 5/5 * n.
        Neu con 4 mang, 4/5*n

    Doi voi moc thoi gian cung vay
    Gia su cho 3'
        Neu con 180s, so diem se la 180/180 * n
        Neu con 45s, so diem se la 45/180 * n
    """
    metadata = {"render_mode": ["human", "rgn_array"], "render_fps": 4}

    def __init__(self, lives: int=3, time_limit: int=None, render_mode = None, size=5, reward_mult = 100):

        self.size = size
        self.window_size=512

        # Truong hop nhieu goi y nhat la ceil(n/2)
        # Nhung kha nang thap vi xs ham pp deu la 0.5, tru khi ma tran nho qua
        MAX_CLUES = math.ceil(size/2)

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "lives_left": spaces.Discrete(lives, start=1),
                
                "game_grid": spaces.Box(-1, 2, shape=(size, size), dtype=int),
                "row_clues": spaces.Box(0, size + 1, shape=(size, MAX_CLUES), dtype=int),
                "col_clues": spaces.Box(0, size + 1, shape=(size, MAX_CLUES), dtype=int)
            }
        )

        # Khoi tao vi tri cua agent
        self._agent_location = np.array([0, 0])

        # So lan thu
        assert lives > 0, "So lan thu toi thieu la 1"
        self.MAX_LIVES = lives # Hang so khong dung den de reset
        self.lives_left = lives 

        # Gioi han thoi gian
        assert time_limit is None or time_limit > 0, "Thoi gian gioi han la int hoac None"
        self.start_time = time.time()
        self.TIME_LIMIT = time_limit
        self.time_left = time_limit

        # So luong hanh dong cua bot
        self.action_space = spaces.Discrete(4)

        # Tao moi truong quan sat
        self.game_grid = np.zeros(shape=(size, size), dtype=int) - 1
        self.result = create_grid(size)        
        assert self.game_grid.shape == self.result.shape, "Ma tran khong deu nhau"
        
        self.row_clues = extract_clues(self.result)
        self.col_clues = extract_clues(self.result.T)

        # He so nhan phan thuong
        self.reward_mult = reward_mult

        # Render mode = human thi tao window cho nguoi ta coi
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "lives_left": self.lives_left, # Cho agent biet so mang con lai de xem goi y neu bi qua

            "game_grid": self.game_grid,
            "row_clues": self.row_clues,
            "col_clues": self.col_clues
        }
    
    def _get_info(self):
        return {
            "agent": self._agent_location,
            "lives_left": self.lives_left,
            "time_left": self.time_left,
            "completion": np.sum((self.game_grid == self.result)) / (self.size**2),

            "game_grid": self.game_grid,
            "result": self.result
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset vi tri
        self._agent_location = np.array([0, 0], dtype=int)

        # Reset lai so mang
        self.lives_left = self.MAX_LIVES

        # Reset lai thoi gian
        self.start_time = time.time()
        self.time_left = self.TIME_LIMIT

        # Reset lai ma tran game va goi y
        self.game_grid = np.zeros(shape=(self.size, self.size), dtype=int) - 1
        self.result = create_grid(self.size, seed)
        self.row_clues = extract_clues(self.result)
        self.col_clues = extract_clues(self.result.T)

        # Lay obs space va info
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    
        

if __name__ == "__main__":
    env = NonogramEnv(time_limit=60)

    print("\n\ninit")
    print("\n_get_obs")
    print(env._get_obs())

    print("\n_get_info")
    print(env._get_info())

    print("\n\nreset")
    env.reset(1)
    print("\n_get_obs")
    print(env._get_obs())

    print("\n_get_info")
    print(env._get_info())
    
        

