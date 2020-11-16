# rl_learn
A package to accompany the book "Reinforcement Learning: A Python Introduction".

It's on PyPi! Just pip install rl-learn and it should be good to go!

Link to the PyPi page: https://pypi.org/project/rl-learn/

Example Code:

from rl_learn.bandits.environments import KArmedTestbed
from rl_learn.bandits.agents import GradientAgent, UCBAgent
from rl_learn.bandits.interaction import assess_and_plot

# Create Gradient and UCB agents, and a description for them
agents = GradientAgent(), UCBAgent()
legend = ["Gradient Agent", "UCB Agent"]

# Define a k-armed bandit problem for the agents to interact with
env = KArmedTestbed()

# Run the interactions and plot the average rewards per timestep
assess_and_plot(env=env,
                agents=agents,
                file_name="example",
                legend=legend)
