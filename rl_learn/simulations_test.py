from rl_learn.mdps.agents import PolicyIterationAgent
from rl_learn.mdps.environments import MDP
from os import path
from rl_learn.mdps.simulations import show_simulation


mdp_path = path.join("/Users/admin/Documents/MyCode/rl_learn/rl_learn/mdp.json")
env = MDP.load(mdp_path)
agent = PolicyIterationAgent(env)
agent.train()
show_simulation(env, agent, speed=1)