# rl_learn
rl-learn is a Python package that teaches fundamental reinforcement learning techniques in Python. It closely follows Sutton and Barto’s famous “Reinforcement Learning: An Introduction”.

Link to the PyPi page: https://pypi.org/project/rl-learn/

1. (Optional, highly recommended) Create an isolated Python virtual environment for using the library using venv: https://docs.python.org/3/library/venv.html

2. After activating the virtual environment, "pip install rl-learn" should grab the updated package from PyPi.

3. View (and optionally run) the example code shown below. It should produce a pdf showing the performance of some bandit agents.

4. Browse through the code in the GitHub browser or example code below to see how to get different (already implemented agents) to interact with other environments. Note that when the comments specify "... defines an interface" that these are not concrete implementations - they instead describe an interface that must be implemented. Also note that Contextual agents/bandits can only interact with their contextual counterparts - the others should be free to "mix and match", though.

5. The required reading to understand everything should be Sutton and Barto's Reinforcement Learning: An Introduction, Edition 2, all of chapter 2.

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
