"""
bandits

A package for designing and evaluating agents that perform on bandit problems.


Modules
-------
environments
  Defines an interface that all bandit environments must provide, as well as
  provide some sample ones. They interact with agents in the agents module.

agents
  Defines an interface that all bandit agents must provide, as well as provide
  some sample ones. They interact with environments in the bandits module.

interaction
  A module that handles the interactions between agents and bandits. It provides
  graphing functions to evaluate the performance of different agents on a
  variety of bandit environments.
"""

# Default number of bandit arms
K = 10

# Default timesteps for plotting and evaluating
T = 1000

# Default number of runs to average across
RUNS = 600

# Default number of states in a contextual bandit problem
N = 3
