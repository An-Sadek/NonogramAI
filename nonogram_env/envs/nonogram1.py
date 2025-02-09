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
    goto_x = 0
    goto_y = 1
    place_0 = 2
    place_1 = 3

