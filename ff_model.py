import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_Net(nn.Module):
    def __init__(self, input_size, small_hidden_size, hidden_size, output_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, small_hidden_size)
        self.hidden2 = nn.Linear(small_hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.hidden4 = nn.Linear(hidden_size, small_hidden_size)
        self.output = nn.Linear(small_hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.hidden2(x)
        x = F.relu(self.hidden3(x))
        x = self.hidden4(x)
        x = F.relu(self.output(x))
        return x

    def save(self, folder, file_name='model.pth'):
        model_folder_path = f'./{folder}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Trainer:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, prediction, target):
        # zero gradiesnt
        self.optimizer.zero_grad()
        # calculate loss
        loss = self.criterion(prediction, target)
        # calculate gradients
        loss.backward()

        self.optimizer.step()
    
    def train_step_batch(self, sample):
        total_loss = 0
        for (initial, target) in sample:
            # zero gradiesnt
            self.optimizer.zero_grad()
            # calculate loss
            prediction = self.model(initial)
            loss = self.criterion(prediction, target)
            total_loss += loss
            # calculate gradients
            loss.backward()

            self.optimizer.step()
        print(f'batch loss = {total_loss}, batch size = {len(sample)}')

