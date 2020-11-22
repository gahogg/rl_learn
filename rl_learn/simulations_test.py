from rl_learn.mdps.agents import PolicyIterationAgent, RandomAgent
from rl_learn.mdps.environments import MDP
from os import path
from rl_learn.mdps.simulations import show_simulation


mdp_path = path.join("/Users/admin/Documents/MyCode/rl_learn/rl_learn/mdp.json")
env = MDP.load(mdp_path)

policy_iteration_agent = PolicyIterationAgent(env)
policy_iteration_agent.train()

n1, reward_avg1 = show_simulation(env, policy_iteration_agent, speed=.1)
print("The policy iteration agent took " + str(n1) + " actions and had a reward average of " + str(round(reward_avg1,2)) + ".")
random_agent = RandomAgent(env)
n2, reward_avg2 = show_simulation(env, random_agent, speed=.1)
print("The random agent took " + str(n2) + " actions and had a reward average of " + str(round(reward_avg2,2)) + ".")