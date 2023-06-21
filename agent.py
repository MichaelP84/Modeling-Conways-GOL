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

class Agent:

    def __init__(self, model_type) -> None:
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        if (model_type == 'CNN'):
            self.model = CNN_Net((WIDTH * LENGTH)).to(device)
            self.trainer = CNN_Trainer(self.model, lr=LR)
        else:
            self.model = Linear_Net(LENGTH * WIDTH, 500, 666, LENGTH * WIDTH).to(device)
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
        initial_Tensor = torch.tensor(initial_state.flatten(), dtype=torch.int, device=device)

        while running:
            # get move
            prediction = agent.model(initial_Tensor)
            # print(prediction)

            # get actual state
            target, num_states, game_running = game.play_step(initial_state)
            running = game_running
            target_Tensor = torch.tensor(target.flatten(), dtype=torch.float)
            agent.remember(initial_Tensor, target_Tensor)

            # train short memory
            agent.train_short_memory(prediction, target_Tensor)

            initial_state = target
            initial_Tensor = target_Tensor

            # download game images
            if (loop % (epochs / 10) == 0):
                game.show_diff(torch.reshape(prediction, (WIDTH, LENGTH)), loop, game.num_states)
            
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
                game.show_diff(torch.reshape(prediction, (WIDTH, LENGTH)), loop, game.num_states)
            
        if (loop > 0):
            agent.train_long_memory(loop)
    
    agent.model.save('cnn_model')

def test_model(epochs):
    model = load_model('C:\\Users\\Michael\\Desktop\\CGOL\\cnn_model\\model.pth')
    game = CGOL()
    accuracy = []
    
    for loop in range(epochs):
        print(f'epoch {loop}/{epochs}')
        running = True

        # create inital game state
        initial_state = np.random.randint(0, 2, (WIDTH, LENGTH))
        initial_Tensor = torch.tensor(initial_state, dtype=torch.float, device=device)

        while running:
            # [x , y] -> [0, 0, x, y]
            initial_Tensor = torch.unsqueeze(initial_Tensor, dim=0)
            initial_Tensor = torch.unsqueeze(initial_Tensor, dim=0)

            # get move
            prediction = model(initial_Tensor)

            # get actual state
            target, num_states, game_running = game.play_step(initial_state)
            target_Tensor = torch.tensor(target, dtype=torch.float, device=device)
            running = game_running

            acc = get_accuracy(torch.tensor(target.flatten(), dtype=torch.float, device=device), prediction)
            accuracy.append(acc)

            initial_state = target
            initial_Tensor = target_Tensor

            # # download game images
            # if ((loop + 1) % (epochs / 10) == 0):
            #     game.show_diff(torch.reshape(prediction, (WIDTH, LENGTH)), loop, game.num_states)
    
    print(sum(accuracy) / len(accuracy))
    
def base_line_model(epochs):
    accuracy = []
    game = CGOL()
    for loop in range(epochs):
        print(f'epoch {loop}/{epochs}')

        running = True
        # create inital game state
        initial_state = np.random.randint(0, 2, (WIDTH, LENGTH))
        initial_Tensor = torch.tensor(initial_state.flatten(), dtype=torch.float, device=device)

        while running:
            # get next games state
            target, num_states, game_running = game.play_step(initial_state)
            target_Tensor = torch.tensor(target.flatten(), dtype=torch.float, device=device)

            running = game_running
            previous_state = initial_Tensor

            # calculate accuracy
            acc = get_accuracy(target_Tensor, previous_state)
            accuracy.append(acc)

            # reassign states
            initial_state = target
            initial_Tensor = target_Tensor
    
    print(sum(accuracy) / len(accuracy))

def get_stats():
    game = CGOL()
    times = 1000
    for x in range (times):
        if (x % (times / 10) == 0):
            print(f'{x}/{times}')
        reward, done = game.play_step(np.random.randint(0, 2, (WIDTH, LENGTH)))
    
    game.get_avg_reached()

def get_accuracy(target, predicted):
    total_error = 0
    cutoff = 0.2
    # print(f'actual: {target}')
    # print(f'prediction: {predicted}')
    for t, p in zip(target, predicted):
        t_val = t.item()
        p_val = p.item()
        if (t_val == 0 and p_val < cutoff):
            total_error += 1
        elif (t_val == 1 and p_val > cutoff):
            total_error += 1
    accuracy = total_error / (WIDTH * LENGTH)
    # print(accuracy)
    return accuracy

def load_model(path):
    model = CNN_Net((WIDTH * LENGTH)).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == '__main__':
    # train_cnn(10000)
    # base_line_model(100) # 87% accuracy
    test_model(100)
