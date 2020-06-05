"""
interaction

A module that handles the interactions between agents and bandits. It provides
graphing functions to evaluate the performance of different agents on a variety
of bandit environments.

Functions
---------
assess_and_plot(env, agents, file_name, num_runs=RUNS,
                timesteps=T, legend=None, contextual=False):
  Performs get_assessment on each agent in agents, and calls plot_assessment
  on the results. This will both plot the results and store the
  image "./images/<file_name>.pdf"
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from rl_learn.bandits import T, RUNS  # Default timesteps and number of runs


def assess_and_plot(env, agents, file_name, num_runs=RUNS,
                    timesteps=T, legend=None, contextual=False):
    """
	Computes the reward that each agent in agents receive for each timestep
	averaged across the number of runs specified by num_runs when interacting
	with the environment. For each run, agent is reinitialized with its original
	parameters, and the environment is randomized using its original parameters.
	It plots the results and stores the image "./images/<file_name>.pdf".

	Parameters
	----------
	env : Bandit or ContextualBandit
	  An environment for the agents to interact with

	agents : [BanditAgent] or [ContextualBanditAgent]
	  A list of agents to run through the environment

	file_name : Str
	  The desired file name of the pdf WITHOUT the .pdf extension.
	  E.g file_name = "file_name" will produce the plot "./images/file_name.pdf"

	num_runs : Int
	  An integer to specify the number of runs to put agent through the env

	timesteps : Int
	  An integer to specify the number of timesteps for each run through env

	legend : [Str]
	  A list of strings representing the name to attach to each agent reward
	  sequence plotted. Has to be in the same order as agents.

	contextual : Bool
	  True if using contextual bandits, else False
	"""
    rew_seqs = []

    for agent in agents:
        seq = _get_assessment(env, agent, num_runs=num_runs,
                              timesteps=timesteps,
                              contextual=contextual)
        rew_seqs.append(seq)

    _plot_assessment(env, rew_seqs, file_name, legend=legend,
                     num_runs=num_runs, contextual=contextual)


def _get_assessment(env, agent, num_runs=RUNS, timesteps=T, contextual=False):
    """
	Returns the reward that agent receives for each timestep averaged across
	the number of runs specified by num_runs when interacting with the
	environment. For each run, agent is reinitialized with its original
	parameters, and the environment is reset/randomized with its
	original parameters.

	Parameters
	----------
	env : Bandit or ContextualBandit
	  A bandit instance to interact with

	agent : BanditAgent or ContextualBanditAgent
	  An agent to interact with the environment

	num_runs : Int
	  An integer to specify the number of runs to put agent through env

	timesteps : Int
	  An integer to specify the number of timesteps for each run through env
	
	contextual : Bool
	  True if env and agent are contextual, else False
	  
	Returns
	-------
	NumPy Array of shape (timesteps, )
	  The rewards averaged across all num_runs agents for each timestep
	"""
    rewards = np.zeros(shape=(num_runs, timesteps))

    # Our loop is slightly different depending on if the bandit
    # is contextual (has multiple states) or not
    if not contextual:
        for i in range(num_runs):
            env = env.get_alike_bandit()
            agent = agent.get_alike_agent()

            for t in range(timesteps):
                action = agent.get_action()
                reward = env.interact(action)
                agent.perform_update(action, reward)
                rewards[i][t] = reward
    else:
        for i in range(num_runs):
            env = env.get_alike_bandit()
            agent = agent.get_alike_agent()
            state = env.get_starting_state()

            for t in range(timesteps):
                action = agent.get_action(state)
                reward, state = env.interact(action)
                agent.perform_update(state, action, reward)
                rewards[i][t] = reward

    # Average across each run
    avg_rewards = np.average(rewards, axis=0)

    return avg_rewards


def _plot_assessment(env, rew_seqs, file_name, legend=None,
                     num_runs=RUNS, contextual=False):
    """
	Plots the results of a call(s) to _get_assessment, and stores the image
	in a pdf with name "<file_name>.pdf" into a "./images/".

	Parameters
	----------
	env : Bandit or ContextualBandit
	  The bandit instance that was used in _get_assessment

	rew_seqs : [NumPy Array of shape(None, )]
	  A list of 1D NumPy Arrays, each being a result from a
	  get_assessment() call

	file_name : Str
	  The desired file name of the pdf WITHOUT the .pdf extension.
	  E.g file_name = "file_name" will produce the
	  plot "./images/file_name.pdf"

	legend : [Str]
	  A list of strings representing the name to attach to each reward_sequence
	  plotted. Has to be in the same order as reward_sequences.

	num_runs : Int
	  The number of runs through the env

	contextual : Bool
	  True if using contextual bandits, else False
	"""
    f = plt.figure()
    plt.xlabel("Timestep")
    plt.ylabel("Average reward")

    if contextual:
        title = "Average Reward Across {0} {1}-state Contextual Bandits".format(
            str(num_runs), str(env.n))
    else:
        title = "Average Reward Across {0} {1}-armed Testbeds".format(
            str(num_runs), str(env.k))

    plt.title(title)

    for reward_sequence in rew_seqs:
        plt.plot(reward_sequence)

    if legend:
        plt.legend(legend)

    if not os.path.exists('images'):
        os.makedirs('images')

    f.savefig("images/" + file_name + ".pdf", bbox_inches="tight")
    plt.show()
