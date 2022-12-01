#---------------------------------------------- # 
# In this file we will be adapting our Dnake game so that it can be controlled by an AI
# Within our play_motion file we will need to output a reward so the AI can train itself
# Equally we will add a reset function so the AI can continue playing the game
# ----------------------------------------------# 
# Imports
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
# ----------------------------------------------# 
# Intialising the pygame modules
pygame.init()
# ----------------------------------------------#
# Constants
# Importing the font to show our score
font = pygame.font.Font('font.ttf', 15)

# The colours we will be suing as part of our game
# Stored as RGB values as pygame requires
white = (255, 255, 255)
red = (200,0,0)
blue = (0, 0, 255)
black = (0,0,0)
green = (119,255,119)
orange = (255,122,0)
yellow = (255,255,0)

# Size of each block of the snake
block_size = 20
# This game speed attribute realtes to the speed of the snake / how fast the game will run
game_speed = 1000
# ----------------------------------------------#
# Classes
# This ENum class is used to prevent any mistakes when calling for the direction of the snake
# Capital Letters are used to indicate that these are constant values (not to be changed)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
# This object is a named tuple
# This tuple makes is easier to access the elements within the tuple by simply using a full stop
# This tuple is used to keep track of the x and y coordinates of the snake at all times
Point = namedtuple('Point', 'x, y')
# ----------------------------------------------#
class SnakeGameAI:

    def __init__(self, w=640, h=480):
        # Display Settings
        # These will be the dimensions of our game window
        self.w = w
        self.h = h

        # Intialising the game window with "set_mode"
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')

        # Adding a clock ot our game to keep the game in-sync
        self.clock = pygame.time.Clock()
        # Resets the game when the snake dies
        self.reset()
# ----------------------------------------------#

    def reset(self):
        # Start of the game
        self.direction = Direction.RIGHT

        # This initalising the snake
        # The named tuple is used to easily collect the x and y position of the snake
        self.head = Point(self.w/2, self.h/2)

        # Creates the body of the snake
        self.snake = [self.head,
                      Point(self.head.x-block_size, self.head.y),
                      Point(self.head.x-(2*block_size), self.head.y)]

        # Tracking the score of the game
        self.score = 0
        # Tracking the existence of food in the game
        self.food = None
        self._place_food()

        # Tracking the frame iteration
        self.frame_iteration = 0
# ----------------------------------------------#

    def _place_food(self):
        # In this function we will be randomnly placing food throughout the game 
        x = random.randint(0, (self.w-block_size )//block_size )*block_size
        y = random.randint(0, (self.h-block_size )//block_size )*block_size

        # Positions the food in a random location in the window
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
# ----------------------------------------------#

    def play_step(self, action):
        # In this function we will analyses eahc step taken within the game 

        # Collecting variables to evaluate which collision the snake makes to lose the game
        self.right_wall = None
        self.left_wall = None
        self.ceiling = None
        self.floor = None
        self.snake_collison = None

        # Each time a move is made we add to the frame iteration
        self.frame_iteration += 1
        # In this function we will updating the snake to move with the user inputted movement
        
        # Checking for a user's inputs 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Update the head
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # Implementing our reward
        # Eating the food: 10 points
        # Losing: -10 points
        # Otherwise: 0 points
        reward = 0
        # Game over parameter to check if the game is over or not
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score,self.right_wall,self.left_wall,self.ceiling,self.floor,self.snake_collison

        # Checks the collison between the head of the snake and the food
        # If there is a valid collision the score will go up by one
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # Call the update_window function to update the game window with a user's inputs
        self._update_ui()
        # Given the clock in our game a time to tick over
        self.clock.tick(game_speed)


        # Returned parameters:
        # Reward
        # Whether the game is over or not
        # Score
        return reward, game_over, self.score,self.right_wall,self.left_wall,self.ceiling,self.floor,self.snake_collison
# ----------------------------------------------#

    def is_collision(self, pt=None):
        # In this variation of the game we use a point parameter to analyse any "danger" that the snake may face
        # For example, if the snake if against a boundary

        # If the point is set to None we keep our point of reference as the head of the snake
        if pt is None:
            pt = self.head

        # This fucnton will check if the snake "dies", either it hits itself or hits a boundary
        # This if statement checks if the snake hits a boundary
        if pt.x > self.w - block_size:
            self.right_wall = 1
            return True

        if pt.x < 0:
            self.left_wall = 1
            return True

        if pt.y > self.h - block_size:
            self.floor = 1
            return True

        if pt.y < 0:
            self.ceiling = 1
            return True

        # This checks if the snake hits iteself
        # We check if the position of the ehad is equal to the position of any of the snake's body parts
        # Iterate from 1 as the snake head is already part of the snake 
        if pt in self.snake[1:]:
            self.snake_collison = 1
            return True

        # Whilst the game is in motion, if there is no collision we can continue
        return False
# ----------------------------------------------#

    def _update_ui(self):
        # This function uses outputs from the play_motion fucntion and updates the game window accordingly
        self.display.fill(black)

        # This plots the head and the body of the snake 
        for pt in self.snake:
            pygame.draw.rect(self.display, black, pygame.Rect(pt.x, pt.y, block_size, block_size))
            pygame.draw.rect(self.display, white, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # This plots the food for the snake   
        pygame.draw.rect(self.display, red, pygame.Rect(self.food.x, self.food.y, block_size, block_size))

        # This plots the text to show the score in the game
        text = font.render("Score: " + str(self.score), True, white)
        # Blit = prints onto the game window
        self.display.blit(text, [8, 8])

        # Adding the coloured boundaries to the game window

        # Ceiling
        pygame.draw.rect(self.display, green, pygame.Rect(0, 0, self.w, 5))
        # Left Wall
        pygame.draw.rect(self.display, orange, pygame.Rect(0, 0, 5, self.h))
        # Right Wall
        pygame.draw.rect(self.display, blue, pygame.Rect(self.w-5, 0, 5, self.h))
        # Floor Boundary
        pygame.draw.rect(self.display, yellow, pygame.Rect(0, self.h-5, self.w, 5))

        # This function is used to update the entire game window
        pygame.display.flip()
# ----------------------------------------------#

    def _move(self, action):
        # In this fucntion we detremine the "action" of the snake, which is determined by a label encoded array
        # STRAIGHT = [1,0,0]
        # RIGHT = [0,1,0]
        # LEFT = [0,0,1] 

        # In order to combat flipped directions when the snake goes up and down we will produce a clockwise movement system

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Now we must check the direction of the snake and match the arrays above to identify the snake's next move

        # Straight
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        
        # Right Turn
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        
        # Left Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += block_size
            self.right_count = 1
        elif self.direction == Direction.LEFT:
            x -= block_size
            self.left_count = 1
        elif self.direction == Direction.DOWN:
            y += block_size
            self.down_count = 1
        elif self.direction == Direction.UP:
            y -= block_size
            self.up_count = 1

        self.head = Point(x, y)