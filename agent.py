import torch
import time
import random
import numpy as np
from collections import deque
from game import CGOL
from game import WIDTH, LENGTH
from ff_model import Linear_Net, Trainer
from cnn_model import CNN_Net, CNN_Trainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Controls the model's interaction with the game
class Agent:

    def __init__(self, model_type) -> None:
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        if (model_type == 'CNN'):
            self.model = CNN_Net((WIDTH * LENGTH)).to(device)
            self.trainer = CNN_Trainer(self.model, lr=LR)
        else:
            self.model = Linear_Net(LENGTH * WIDTH, 800, 1024, LENGTH * WIDTH).to(device)
            self.trainer = Trainer(self.model, lr=LR)

    def remember(self, initial, target):
        self.memory.append((initial, target))
         # popleft if MAX_MEMORY is reached
    
    def train_short_memory(self, prediction, target):
        self.trainer.train_step(prediction, target)

    def train_long_memory(self, loop):
        if (len(self.memory) > BATCH_SIZE):
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        self.trainer.train_step_batch(sample)

def train_ff(epochs):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    game = CGOL()
    agent = Agent('FF')
    
    for loop in range(epochs):
        print(f'epoch {loop}/{epochs}')

        running = True
        # create inital game state
        initial_state = np.random.randint(0, 2, (WIDTH, LENGTH))
        initial_Tensor = torch.tensor(initial_state.flatten(), dtype=torch.float, device=device)

        while running:
            # get move
            prediction = agent.model(initial_Tensor)
            # print(prediction)

            # get actual state
            target, num_states, game_running = game.play_step(initial_state)
            running = game_running
            target_Tensor = torch.tensor(target.flatten(), dtype=torch.float, device=device)
            agent.remember(initial_Tensor, target_Tensor)

            # train short memory
            agent.train_short_memory(prediction, target_Tensor)

            initial_state = target
            initial_Tensor = target_Tensor

            # download game images
            if ((loop + 1) % (epochs / 10) == 0):
                game.download_diff(torch.reshape(prediction, (WIDTH, LENGTH)), loop, game.num_states)
            
        if (loop > 0):
            agent.train_long_memory(loop)
    
    agent.model.save('ff_model')

def train_cnn(epochs):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    game = CGOL()
    agent = Agent('CNN')
    
    for loop in range(epochs):
        print(f'epoch {loop}/{epochs}')
        running = True

        # create inital game state
        initial_state = np.random.randint(0, 2, (WIDTH, LENGTH))
        initial_Tensor = torch.tensor(initial_state, dtype=torch.float, device=device) # [WIDTH, LENGTH]

        while running:
            # -> [0, 0, WIDTH, LENGTH]
            initial_Tensor = torch.unsqueeze(initial_Tensor, dim=0)
            initial_Tensor = torch.unsqueeze(initial_Tensor, dim=0)

            # get move
            prediction = agent.model(initial_Tensor)

            # get actual state
            target, num_states, game_running = game.play_step(initial_state)
            running = game_running
            target_Tensor = torch.tensor(target, dtype=torch.float, device=device)
            target_flat = torch.tensor(target.flatten(), dtype=torch.float, device=device)
            agent.remember(initial_Tensor, target_flat)

            # train short memory
            agent.train_short_memory(prediction, target_flat)

            initial_state = target
            initial_Tensor = target_Tensor

            # download game images
            if ((loop + 1) % (epochs / 10) == 0):
                game.download_diff(torch.reshape(prediction, (WIDTH, LENGTH)), loop, game.num_states)
            
        if (loop > 0):
            agent.train_long_memory(loop)
    
    agent.model.save('cnn_model')

def get_game_stats():
    game = CGOL()
    times = 1000
    for x in range (times):
        if (x % (times / 10) == 0):
            print(f'{x}/{times}')
        reward, done = game.play_step(np.random.randint(0, 2, (WIDTH, LENGTH)))
    game.get_avg_reached()


if __name__ == '__main__':
    # train_cnn(5000) # model_cnn_3
    train_ff(5000) # 95% accuracy
