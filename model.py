# ---------------------------------------------- #
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
# ---------------------------------------------- #
# Setting up our neural network
# We will define;
#   - Size of the input layer
#   - Size of the hidden layer
#   - Size of the output layer

class neural_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # This initalises the neural network in PyTorch
        super().__init__()
        # Parse the input information throught the input layer and hidden layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Parse the data through to the output layer
        self.linear2 = nn.Linear(hidden_size, output_size)
# ---------------------------------------------- #

    def forward(self, x):
        # This allows our data to pass through the neural network

        # x is defined as our tensor

        # We use the relu() function to set up the rectified linear function as our activation fuction
        x = F.relu(self.linear1(x))
        # Then we parse and output the tensor
        x = self.linear2(x)
        return x
# ---------------------------------------------- #

    def save(self, file_name='model.pth'):
        # We use this function to save our model and store it in its own folder
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # This is used to save the model
        torch.save(self.state_dict(), file_name)
# ---------------------------------------------- #

class trainer:
    # We will use this class to help train the model
    # This will help us format our arrays/tensors to train the model
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # This sets the optimizer of our model 
        # In this case we chose the Adam optimzer
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # The criterion to evaluate the model is the Mean Squared Error loss function
        self.criterion = nn.MSELoss()
# ---------------------------------------------- #

    def train_step(self, state, action, reward, next_state, game_over):
        # This allows us to store all the relevant variables as tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # Tensor / Array shape handling
        
        # We want change the shape of the tensor if it only contains 1 dimension
        if len(state.shape) == 1:
            # This appends a dimension to the array
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            # Create a tuple out of the game over variable
            game_over = (game_over, )

        prediction = self.model(state)

        target = prediction.clone()
        for idx in range(len(game_over)):
            result = reward[idx]
            if not game_over[idx]:
                result = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = result

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
# ---------------------------------------------- #