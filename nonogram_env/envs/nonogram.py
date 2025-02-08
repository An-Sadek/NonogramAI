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
    goto_x = 0
    goto_y = 1
    place_0 = 2
    place_1 = 3

class NonogramEnv(gym.Env):
    pass