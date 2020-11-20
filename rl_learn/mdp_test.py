from rl_learn.mdps.environments import MDP
import numpy as np


env_from_gui = MDP.MDP_from_gui()
path = env_from_gui.save("/Users/admin/Documents/MyCode/rl_learn/rl_learn")
new_env = MDP.load(path)
env = new_env

s = 0
for i in range(100):
    a = np.random.choice(1)
    prev_s = s
    r, s = env.interact(s, a)
    print(str(i+1) + "). We transitioned to state " + str(s) + " from state " + str(prev_s) + " taking action " + str(a) + " and received a reward of " + str(r))
