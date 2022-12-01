# ---------------------------------------------- #
# Imports
import torch
import random
import numpy as np
from collections import deque
from demo_game import SnakeGameAI, Direction, Point
from model import neural_net, trainer
from helper import plot
import matplotlib.pyplot as plt
# ---------------------------------------------- #
# Parameters for our model

# This sets the memory limit for he deque in our model
memory_szie = 100_000
# This sets the batch size of the information we will be parsing into our model
batch = 10000
# This sets the learning rate of our model
LR = 0.001
# ---------------------------------------------- #
# Plotting
# Initialising the plots for the analytics of the game
fig, (ax1,ax2) = plt.subplots(2,1)
fig.set_figheight(10)
fig.set_figwidth(10)
# ---------------------------------------------- #
# Brain Class
class Brain:

    def __init__(self):
        # Tracks the number of games the AI plays
        self.game_number = 0
        # The epsilon parameter controls the "randomness" of the model
        self.epsilon = 0
        # The gamma parameter refers to the models discount rate
        self.gamma = 0.9

        # We use a deque so that we can pop elements from the left-hand side of our list
        self.memory = deque(maxlen=memory_szie)

        # The model
        # Input Layer
        # Hidden Layer
        # Output Layer
        self.model = neural_net(11, 256, 3)

        # Training
        self.trainer = trainer(self.model, lr=LR, gamma=self.gamma)

# ---------------------------------------------- #

    def get_state(self, game):
        # This functions grabs the state in which the Snake game is currently in to analyse for the next move

        # We begin by grabbing the head of the snake
        head = game.snake[0]
        # Analysing points next to the head of the snake

        # Left of the snake head
        left_point = Point(head.x - 20, head.y)
        # Right of the snake head
        right_point = Point(head.x + 20, head.y)
        # Above of the snake head
        up_point = Point(head.x, head.y - 20)
        # Below of the snake head
        down_point = Point(head.x, head.y + 20)
        
        # Next we check the current direction of the snake
        # We analyse this by using boolean statements 
        left_direction = game.direction == Direction.LEFT
        right_direction = game.direction == Direction.RIGHT
        up_direction = game.direction == Direction.UP
        down_direction = game.direction == Direction.DOWN

        # We analyse the state the snake is in and its "danger" (whether it is colse to hitting a boundary)
        # This analysis is carried out by creating an array and formed by boolean statements
        state = [
            # Checks for any danger in the straight line axis of the snake
            (right_direction and game.is_collision(right_point)) or 
            (left_direction and game.is_collision(left_point)) or 
            (up_direction and game.is_collision(up_point)) or 
            (down_direction and game.is_collision(down_point)),

            # Checks for any danger to the right of the snake
            (up_direction and game.is_collision(right_point)) or 
            (down_direction and game.is_collision(left_point)) or 
            (left_direction and game.is_collision(up_point)) or 
            (right_direction and game.is_collision(down_point)),

            # Checks for any danger to the left of the snake
            (down_direction and game.is_collision(right_point)) or 
            (up_direction and game.is_collision(left_point)) or 
            (right_direction and game.is_collision(up_point)) or 
            (left_direction and game.is_collision(down_point)),
            
            # Directional moves
            # In this case only one direction would be true
            left_direction,
            right_direction,
            up_direction,
            down_direction,
            
            # Locating the position of the food
            # Checking if the food is to the left of the snake 
            game.food.x < game.head.x,
            # Checking if the food is to the right of the snake
            game.food.x > game.head.x, 
            # Checking if the food is to the above of the snake
            game.food.y < game.head.y,
            # Checking if the food is to the below of the snake
            game.food.y > game.head.y
            ]


        # We return this array and convert the boolean outputs to either 0 or 1 using dtype = int
        return np.array(state, dtype=int)
# ---------------------------------------------- #

    def remember_memory(self, state, action, reward, next_state, game_over):
        # This functions remembers the latest state and returns its reward, the next action, and determines whether the game is over or not

        # If the memory exceeds the max memory we allow we will "pop left" (a feature of deque)
        self.memory.append((state, action, reward, next_state, game_over))
        # The memory is stored as tuples with the relevant attributes
# ---------------------------------------------- #

    def train_long(self):
        # We can pass a whole batch of data through our model 
        if len(self.memory) > batch:
            # If the size of the model's memory is greater than the size of the batch we will take a random sample of memory equal to the size of the batch
            mini_sample = random.sample(self.memory, batch)
            # This will create a list of tuples with all the relevant attributes
        else:
            # This just takes the entire memory that has already been stored
            mini_sample = self.memory

        # Now we need to zip up all the:
        # states
        # actions
        # rewards
        # next states (next game state for the snake)
        # game overs (whether the game is over or not)

        # We use the asterisk to zip each attribute of the tuple

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)
# ---------------------------------------------- #

    def train_short(self, state, action, reward, next_state, game_over):
        # Stores the memory of each step within the game
        self.trainer.train_step(state, action, reward, next_state, game_over)
# ---------------------------------------------- #

    def get_action(self, state):
        # This will tell the snake which action to take
        # We need a way for the model to begin exploring and then switch to exploitation
        #           Exploration: allows the model to explore potential moves
        #           Explotation: uses bias to find the optimal moves

        # Exploration
        # As the number of games increases the value of epsilon decreases and once it becomes negative the moves of the snake will no longer be random
        self.epsilon = 80 - self.game_number
        final_move = [0,0,0]
        # This changes an element within our random move parameter to allow the snake to make its move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            # Changing a random element in the final move parameter to 0 or 1
            final_move[move] = 1

        # Exploitation
        else:
            # Collects the state of the game and creates a prediction for the snake to follow
            # We pass this data using a tensorflow (3-dimensioanl array)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)

            # We will then take the maximun lement in our predicted ouput and set that value equal to 1 which will correspond
            # to the movement of our snake
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
# ---------------------------------------------- #

def train():
    # Variables to plot later
    # Tracking scores
    plot_scores = []
    # Tracking mean scores
    plot_mean_scores = []
    # Tracking total score
    total_score = 0
    # Tracking the record score
    record = 0
    # Tracking collisions with right wall
    right_wall_total = 0
    # Tracking collisions with left wall
    left_wall_total = 0
    # Tracking collisions with floor
    floor_total = 0
    # Tracking collisions with ceiling
    ceiling_total = 0
    # Tracking collisions with the snake itself
    snake_collison_total = 0

    # Calling the brain and the game
    brain = Brain()
    game = SnakeGameAI()

    # Training loop
    while True:
        # Acquire the old state
        state_old = brain.get_state(game)

        # Get snake's move
        final_move = brain.get_action(state_old)

        # Pefrom the move and provide the new state
        reward, game_over, score,right_wall,left_wall,ceiling,floor,snake_collison = game.play_step(final_move)
        state_new = brain.get_state(game)

        # Train the short memory
        brain.train_short(state_old, final_move, reward, state_new, game_over)

        # Remember everythhing and store in the memory
        brain.remember_memory(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # Train the long memory
            # Known as experience replay

            # Plot the results

            # This resets the game so that we can run the game again
            game.reset()
            # Keeps track of the number of games played
            brain.game_number += 1

            # Store the results in the long memory
            brain.train_long()
# ---------------------------------------------- #

            # Append to the total of collision statistics
            # Right wall
            if right_wall != None:
                right_wall_total += 1
            # Left Wall
            if left_wall != None:
                left_wall_total += 1
            # Ceiling
            if ceiling != None:
                ceiling_total += 1
            # Floor
            if floor != None:
                floor_total += 1
            # Itself
            if snake_collison != None:
                snake_collison_total += 1
# ---------------------------------------------- #

            # Saving the new record if the old record is exceeded
            if score > record:
                record = score
                brain.model.save()
            
            # Printing the number of games, current score and record score 
            print(f'This is game number: {brain.game_number} with score: {score}\nOur current record is: {record}')
            
            # Plotting the features
            plot_scores.append(score)
            total_score += score
            # Averaging out the scores
            mean_score = total_score / brain.game_number
            plot_mean_scores.append(mean_score)
            # Plotting all the parameters including the scores and collision statistics
            plot(plot_scores, plot_mean_scores,ax1,ax2,fig,right_wall_total,left_wall_total,floor_total,ceiling_total,snake_collison_total)            
# ---------------------------------------------- #

if __name__ == '__main__':
    # Beginning the training loop
    train()