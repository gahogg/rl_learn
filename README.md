# rl_learn
rl-learn is a Python package that teaches fundamental reinforcement learning techniques in Python. It closely follows Sutton and Barto’s famous “Reinforcement Learning: An Introduction”.

It's on PyPi! Just pip install rl-learn and it should be good to go!

Link to the PyPi page: https://pypi.org/project/rl-learn/

Example Code:

```python
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
```
