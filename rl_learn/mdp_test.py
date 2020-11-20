from rl_learn.mdps.environments import MDP
import numpy as np

state_transitions = [[[.25, .75],[.5, .5]], [[.4, .6],[.2, .8]]]
reward_sas_triples = [[[[.2, .8, 0],[.5, 0, .5]], [[0, .75, .25],[0, .5, .5]]], [[[.5, 0, .5],[0, .5, .5]], [[.5, 0, .5],[0, .75, .25]]]]
rewards = [-1, 0, 1]

env_from_scratch = MDP.MDP_from_transitions_and_reward_sas_triples(state_transitions, reward_sas_triples, rewards)
env_from_scratch.save("/Users\admin\Documents\MyCode\rl_learn\rl_learn")
"""env_from_gui = MDP.MDP_from_gui()

env = env_from_gui

s = 0
for i in range(100):
    a = np.random.choice(1)
    prev_s = s
    r, s = env.interact(s, a)
    print(str(i+1) + "). We transitioned to state " + str(s) + " from state " + str(prev_s) + " taking action " + str(a) + " and received a reward of " + str(r))
"""