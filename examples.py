from rl_learn.bandits.environments import KArmedTestbedNonStationary
from rl_learn.bandits.agents import IncrementalAvgEpsGreedyAgent
from rl_learn.bandits.agents import IncrementalConstEpsGreedyAgent
from rl_learn.bandits.agents import UCBAgent
from rl_learn.bandits.interaction import assess_and_plot

ucb_agent = UCBAgent()
inc_avg_agent = IncrementalAvgEpsGreedyAgent()

tb_changing = KArmedTestbedNonStationary()

assess_and_plot(env=tb_changing, agents=[ucb_agent, inc_avg_agent], file_name="yo3", timesteps=1500, legend=["UCB, Constant Alpha = 0.1",
                        "Epsilon = 0.01, Incremental Sample Average"])
