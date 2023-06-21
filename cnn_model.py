import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN_Net(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.CNN_1 = nn.Sequential(
            
            nn.Conv2d(1, 4, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(4,8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )

        self.CNN_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16 ,32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.CNN_3 = nn.Sequential(   
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.Linear = nn.Sequential(
            nn.Linear(800,800),
            nn.ReLU(),
            nn.Linear(800,1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        x = self.CNN_1(x)
        x = x.flatten()
        # x = self.CNN_2(x)
        # print(x.shape)
        # x = self.CNN_3(x)
        # print(x.shape)
        return self.Linear(x)

    def save(self, folder, file_name='model.pth'):
        model_folder_path = f'./{folder}'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class CNN_Trainer:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, prediction, target):
        # zero gradiesnt
        self.optimizer.zero_grad()
        prediction = torch.flatten(prediction)
        # print(f'training {prediction.shape, target.shape}')
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
            # calculate prediction
            prediction = self.model(initial)
            # calculate loss
            # print(f'training {prediction.shape, target.shape}')

            loss = self.criterion(prediction, target)
            total_loss += loss
            # calculate gradients
            loss.backward()

            # self.optimizer.step()
        print(f'batch loss = {total_loss}, batch size = {len(sample)}')

