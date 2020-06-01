"""
bandits

A package for designing and evaluating
agents that perform on bandit problems.

Modules
-------
environments
  Defines an interface that all bandit environments
  must provide, as well as provide some sample ones.
  They interact with agents in the agents module.

agents
  Defines an interface that all bandit agents must
  provide, as well as provide some sample ones. They
  interact with environments in the bandits module.

interaction
  A module that handles the interactions
  between Bandits and BanditAgents. It provides
  graphing and metric functions to
  evaluate the performance of different 
  BanditAgents on various Bandit environments.
"""

## Default number of bandit arms
K = 10

## Default timesteps for plotting and evaluating
T = 750

## Default number of runs to average across
RUNS = 300
