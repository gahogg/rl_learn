import sys, pygame
from collections.abc import Iterable
from pygame import draw
from pygame.math import Vector2
from numpy import random

pygame.init()
BLACK = 0, 0, 0
RED = 255, 0, 0
BLUE = 0, 0, 255

class Agent:
    def __init__(self, states, color=RED, center=(0,0), radius=10, speed=(.05, .05)):
        self.color = list(color) # [r, g, b]
        self.center = list(center) # [x, y]
        self.radius = radius # > 1
        self.speed = speed # (speed_x, speed_y)
        self.states = states # {state_name : state}
    
    def draw(self, surface):
        col = self.color
        center = self.center
        r = self.radius
        draw.circle(surface=surface, color=col, center=center, radius=r)
        return self
    
    def move(self, delta_xy=[1, 1], speed=None):
        if speed is None:
            speed = self.speed
        elif not isinstance(speed, Iterable):
            speed = [speed, speed]

        delta_x, delta_y = delta_xy
        self.center[0] += delta_x * speed[0]
        self.center[1] += delta_y * speed[1]
    
    def move_towards_state(self, state_name, speed=None):
        state = self.states[state_name]
        desired_pos = state.center
        already_there = self.move_towards(desired_pos, speed=speed)
        return already_there

    def move_towards(self, center, speed=None):
        diff = Vector2((center[0]-self.center[0], center[1]-self.center[1]))
        print(diff)
        if abs(diff.x) < 1 and abs(diff.y) < 1:
            return True

        normalized_diff = diff.normalize()
        scaled_diff = normalized_diff
        self.move(delta_xy=(scaled_diff.x, scaled_diff.y), speed=speed)
        return False
    


class State:
    def __init__(self, color=BLUE, center=(0,0), radius=20, name="State"):
        self.color = list(color) # [r, g, b]
        self.center = list(center) # [x, y]
        self.radius = radius # > 1
        self.name = name
    
    def draw(self, surface):
        col = self.color
        center = self.center
        r = self.radius
        draw.circle(surface=surface, color=col, center=center, radius=r)
        return self
    
size = width, height = 400, 400
screen = pygame.display.set_mode(size)

state0 = State(center=(10,10), name="state0")
state1 = State(center=(width-10, 10), name="state1")
state2 = State(center=(10, height-10), name="state2")
state3 = State(center=(width-10, height-10), name="state3")
states = [state0, state1, state2, state3]
agent = Agent(color=RED, 
              center=(100, 100), 
              radius=30, 
              states={state.name : state for state in states},
              speed=(.3, .3))
desired_state_index = 0

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    
    screen.fill(BLACK)

    desired_state_name = states[desired_state_index].name
    already_there = agent.move_towards_state(desired_state_name)

    if already_there:
        desired_state_index = random.choice(len(states))
    
    agent.draw(screen)

    for state in states:
        state.draw(screen)
    pygame.display.flip()