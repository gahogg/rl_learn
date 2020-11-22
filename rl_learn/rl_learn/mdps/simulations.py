import sys, pygame
from collections.abc import Iterable
from pygame import draw
from pygame.math import Vector2
from numpy import random

pygame.init()

BLACK = 0, 0, 0
RED = 255, 0, 0
BLUE = 0, 0, 255

class AgentVisual:
    def __init__(self, states, color=RED, center=(0,0), radius=10, speed=(.05, .05)):
        self.color = list(color) # [r, g, b]
        self.center = list(center) # [x, y]
        self.radius = radius # > 1
        self.speed = speed # (speed_x, speed_y)
        self.states = states # {state_name : state}
        print(self.states)
    
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
        print(center)
        print(self.center)
        diff = Vector2((center[0]-self.center[0], center[1]-self.center[1]))
        print(diff)
        if abs(diff.x) < 1 and abs(diff.y) < 1:
            return True

        normalized_diff = diff.normalize()
        scaled_diff = normalized_diff
        self.move(delta_xy=(scaled_diff.x, scaled_diff.y), speed=speed)
        return False
    
class StateVisual:
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

class StatePair:
    def __init__(self, state, state_visual):
        self.state = state
        self.state_visual = state_visual
    
    @staticmethod
    def construct_state_dict(state_pairs):
        return {pair.state_visual.name : pair.state_visual for pair in state_pairs}

class AgentPair:
    def __init__(self, agent, agent_visual):
        self.agent = agent
        self.agent_visual = agent_visual

def show_simulation(mdp, agent, window_width=600, window_height=600, speed=.1):
    pygame.init()
    S = mdp.S
    A = mdp.A
    R = mdp.R
    possible_rewards = mdp.rewards

    state_pairs = [StatePair(i, StateVisual(name="State " + str(i), center=(random.choice(window_width), random.choice(window_height)))) for i in range(S)]
    agent_pair = AgentPair(agent, AgentVisual(states=StatePair.construct_state_dict(state_pairs), speed=(speed, speed)))

    screen = pygame.display.set_mode((window_width, window_height))

    s = 0
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
        
        screen.fill(BLACK)

        print(state_pairs[s].state_visual.name)
        already_there = agent_pair.agent_visual.move_towards_state(state_name=state_pairs[s].state_visual.name)

        if already_there:
            # Take an action in the current state
            r, s = mdp.interact(s, agent.get_action(s))
        
        agent_pair.agent_visual.draw(screen)

        for state_pair in state_pairs:
            state_pair.state_visual.draw(screen)

        pygame.display.flip()

