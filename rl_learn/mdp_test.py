from rl_learn.mdps.environments import MDP
from rl_learn.mdps.agents import PolicyIterationAgent
import numpy as np
from os import path

#env_from_gui = MDP.MDP_from_gui()
pth = path.join("/Users/admin/Documents/MyCode/rl_learn/rl_learn/mdp.json")
new_env = MDP.load(pth)
env = new_env

trained_agent = PolicyIterationAgent(env, tolerance=.01)
trained_agent.train()

s = 0
rewards = [[], []]
S = env.S
A = env.A

for i in range(100):
    random_a = np.random.choice(1)
    trained_a = trained_agent.get_action(s)
    #prev_s = s
    random_r, random_s = env.interact(s, random_a)
    trained_r, trained_s = env.interact(s, trained_a)
    rewards[0].append(random_r)
    rewards[1].append(trained_r)
    #print(str(i+1) + "). We transitioned to state " + str(s) + " from state " + str(prev_s) + " taking action " + str(a) + " and received a reward of " + str(r))

print("Random agent average reward: " + str(np.average(rewards[0])))
print("Trained agent average reward: " + str(np.average(rewards[1])))
