"""
interaction

A module that handles the interactions
between Bandits and BanditAgents. It provides
graphing and metric functions to
evaluate the performance of different 
BanditAgents on various Bandit environments.

Functions
---------
get_assessment(env, agent, num_runs=RUNS, timesteps=T)
  Returns the reward that agent receives for each timestep averaged
  across the number of runs specified by num_runs when interacting 
  with the environment.

plot_assessment(env, reward_sequences, file_name, legend=None, num_runs=RUNS):
  Plots the results of a call(s) to get_assessment, and stores the image
  in a pdf with name "<file_name>.pdf" into a "./images/".

assess_and_plot(env, agents, file_name, num_runs=RUNS, timesteps=T, legend=None):
  Performs get_assessment on each agent in agents,
  and calls plot_assessment on the results. This will both plot the results 
  and store the image "./images/<file_name>.pdf"
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from rl_learn.bandits import T, RUNS # Default timesteps and number of runs


def get_assessment(env, agent, num_runs=RUNS, timesteps=T):
	"""
	Returns the reward that agent receives for each timestep averaged
	across the number of runs specified by num_runs when interacting 
	with the environment. For each run, agent is reinitialized with its 
	original parameters, and the environment is reset/randomized
	with its original parameters.

	Parameters
	----------
	env : Bandit
	  A Bandit instance to interact with

	agent : BanditAgent
	  An agent to interact with the environment

	num_runs : Int
	  An integer to specify the number of runs to
	  put agent through the env

	timesteps : Int
	  An integer to specify the number of timesteps
	  for each run through the env

	Returns
	-------
	NumPy Array of shape (timesteps, )
	  The rewards averaged across all num_runs BanditAgents
	  for each timestep
	"""
	rewards = np.zeros(shape=(num_runs, timesteps))

	for i in range(num_runs):
		## Reinitialize both the agent and environment
		env = env.get_alike_bandit()
		agent = agent.get_alike_agent()

		for t in range(timesteps):
			action = agent.get_action()
			reward = env.interact(action)
			agent.perform_update(action, reward)
			rewards[i][t] = reward

	## Average across each run
	avg_rewards = np.average(rewards, axis=0)

	return avg_rewards


def plot_assessment(env, reward_sequences, file_name, legend=None, num_runs=RUNS):
	"""
	Plots the results of a call(s) to get_assessment, and stores the image
	in a pdf with name "<file_name>.pdf" into a "./images/".

	Parameters
	----------
	env : Bandit
	  The environment that the interaction was with

	reward_sequences : [NumPy Array of shape(None, )]
	  A list of 1D NumPy Arrays, each being a result
	  from a get_assessment() call

	file_name : Str
	  The desired file name of the pdf WITHOUT
	  the .pdf extension. E.g file_name = "file_name" will
	  produce the plot "./images/file_name.pdf"

	legend : [Str]
	  A list of strings representing the name to attach to 
	  each reward_sequence plotted. Has to be in the same
	  order as reward_sequences.

	num_runs : Int
	  The number of runs through the env
	"""
	f = plt.figure()
	plt.xlabel("Timestep")
	plt.ylabel("Average reward")
	plt.title("Average reward across " + str(num_runs)  
	+ " " + str(env.k) + "-armed testbeds")

	for reward_sequence in reward_sequences:
		plt.plot(reward_sequence)

	if legend:
		plt.legend(legend)

	if not os.path.exists('images'):
		os.makedirs('images')

	f.savefig("images/" + file_name + ".pdf", bbox_inches="tight")
	plt.show()


def assess_and_plot(env, agents, file_name, num_runs=RUNS, timesteps=T, legend=None):
	"""
	Computes the reward that each agent in agents 
	receive for each timestep averaged across the number of runs specified 
	by num_runs when interacting with the environment. For each run, agent
	is reinitialized with its original parameters, and the environment is
	randomized using its original parameters. It plots the results and stores 
	the image "./images/<file_name>.pdf".

	Parameters
	----------
	env : Bandit
	  An environment for the agents to
	  interact with

	agents : [BanditAgent]
	  A list of BanditAgents to run through
	  the environment

	file_name : Str
	  The desired file name of the pdf WITHOUT
	  the .pdf extension. E.g file_name = "file_name" will
	  produce the plot "./images/file_name.pdf"

	num_runs : Int
	  An integer to specify the number of runs to
	  put agent through the env

	timesteps : Int
	  An integer to specify the number of timesteps
	  for each run through the env

	legend : [Str]
	  A list of strings representing the name to attach to 
	  each agent reward sequence plotted. Has to be in the same
	  order as agents.
	"""
	rew_seqs = []

	for agent in agents:
		seq = get_assessment(env, agent, num_runs=num_runs, timesteps=timesteps)
		rew_seqs.append(seq)

	plot_assessment(env, rew_seqs, file_name, legend=legend, num_runs=num_runs)
