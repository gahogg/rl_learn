import sys, pygame
from collections.abc import Iterable
from pygame import draw
from pygame.math import Vector2
from numpy import random
from pygame import font

pygame.init()
font.init()

STATE_FONT = font.SysFont('Comic Sans MS', 30)
ACTION_FONT = font.SysFont('Comic Sans MS', 20)
REWARD_FONT = font.SysFont('Comic Sans MS', 20)
REWARD_AVG_FONT = font.SysFont('Comic Sans MS', 30)

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
    
    def draw(self, surface):
        col = self.color
        center = self.center
        r = self.radius
        draw.circle(surface=surface, color=col, center=center, radius=r)
        return self
    
    def draw_action_text(self, surface, action):
        label = ACTION_FONT.render("Took action " + str(action), False, (255, 0, 255))
        surface.blit(label, (self.center[0], self.center[1]+self.radius*2))
    
    def draw_reward_text(self, surface, reward_val):
        label = REWARD_FONT.render("Received reward " + str(reward_val), False, (255, 255, 0))
        surface.blit(label, (self.center[0], self.center[1]+self.radius*2))
    
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
        label = STATE_FONT.render(self.name, False, (0, 255, 255))
        surface.blit(label, (self.center[0], self.center[1]+self.radius*2))
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
    state_pairs = [StatePair(i, StateVisual(name="State " + str(i), center=(random.choice([100, 300]), random.choice([100, 300])))) for i in range(S)]
    agent_pair = AgentPair(agent, AgentVisual(states=StatePair.construct_state_dict(state_pairs), speed=(speed, speed), center=state_pairs[0].state_visual.center))
    screen = pygame.display.set_mode((window_width, window_height))

    s = 0
    action = agent.get_action(s)
    r, s = mdp.interact(s, action)
    reward_avg = 0
    n = 1
    heading_to_state = False
    middle = (window_width//2, window_height//2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return n, reward_avg
        
        screen.fill(BLACK)

        if heading_to_state:
            already_there = agent_pair.agent_visual.move_towards_state(state_name=state_pairs[s].state_visual.name)

            if already_there:
                action = agent.get_action(s)
                r, s = mdp.interact(s, action)
                heading_to_state = False
        else:
            already_there = agent_pair.agent_visual.move_towards(center=middle)

            if already_there:
                heading_to_state = True
                reward_avg += (1/n)*(r - reward_avg)
                n += 1
            
        agent_pair.agent_visual.draw(screen)

        if heading_to_state:
            agent_pair.agent_visual.draw_reward_text(screen, r)
        else:
            agent_pair.agent_visual.draw_action_text(screen, action)
        
        _draw_reward_average(screen, reward_avg)
        for state_pair in state_pairs:
            state_pair.state_visual.draw(screen)

        pygame.display.flip()

def _draw_reward_average(screen, reward_avg):
    label = REWARD_AVG_FONT.render("Reward average: " + str(round(reward_avg,2)), False, (100, 0, 255))
    screen.blit(label, (50, 550))
