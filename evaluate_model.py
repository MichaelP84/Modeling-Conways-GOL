from game import CGOL, WIDTH, LENGTH
import torch
import numpy as np
from cnn_model import CNN_Net
from ff_model import Linear_Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def load_model_CNN(path):
    model = CNN_Net((WIDTH * LENGTH)).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def load_model_FF(path):
    model = Linear_Net(LENGTH * WIDTH, 800, 1024, LENGTH * WIDTH).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

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

def test_model_cnn(epochs, model):
    model = load_model_CNN(f'C:\\Users\\Michael\\Desktop\\CGOL\\cnn_model\\{model}.pth')
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

def test_model_ff(epochs, model):
    model = load_model_FF(f'C:\\Users\\Michael\\Desktop\\CGOL\\ff_model\\{model}.pth')
    game = CGOL()
    accuracy = []
    
    for loop in range(epochs):
        print(f'epoch {loop}/{epochs}')
        running = True

        # create inital game state
        initial_state = np.random.randint(0, 2, (WIDTH, LENGTH))
        initial_Tensor = torch.tensor(initial_state.flatten(), dtype=torch.float, device=device)

        while running:

            # get move
            prediction = model(initial_Tensor)

            # get actual state
            target, num_states, game_running = game.play_step(initial_state)
            target_Tensor = torch.tensor(target.flatten(), dtype=torch.float, device=device)
            running = game_running

            acc = get_accuracy(target_Tensor, prediction)
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

def main():
    # base_line_model(100) # 87% accuracy
    # test_model_cnn(100, 'model_cnn_3') # 99.88% accuracy
    # test_model_ff(100, 'model_ff_2') # 99.81% accuracy
    pass


if __name__ == '__main__':
    main()