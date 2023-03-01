import torch
import time
import random
import numpy as np
from collections import deque
from game import CGOL
# from model import Linear_QNet, QTrainer
# from helper import plot\

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # self.model = Linear_QNet(11, 256, 3)
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass
    
    def train_short_memory(self):
        pass

    def get_action(self):
        pass


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = CGOL()


    states, game_iteration, num_unique = game.play_step()
    print(states, game_iteration, num_unique)
    cells = np.random.randint(0, 2, (60, 80))
    game.reset(cells)

        


if __name__ == '__main__':
    train()