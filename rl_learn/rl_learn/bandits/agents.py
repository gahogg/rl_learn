"""
agents

A module that defines the interface that any bandit agent must provide, as
well as provide some sample ones.

Classes
-------
BanditAgent
  The interface that all subclasses of BanditAgent must implement so that
  they can interact with Bandits - the standard bandit environment that has
  no concept of state.

GreedyAgent(BanditAgent)
  A bandit agent that acts greedily with respect to its Q(a)s, which are
  computed via a sample average of the rewards received for taking each action.

ExploreGreedyAgent(GreedyAgent):
  A bandit agent that will act greedily only after trying each of its actions
  once. It uses the sample average method of estimating its Q-values.

EpsilonGreedyAgent(GreedyAgent):
  A bandit agent that acts epsilon-greedily with respect to its Q-values.
  With probability (1 - epsilon) it chooses the greedy action, and with small
  probability epsilon it acts randomly. It uses the sample average method of
  estimating its Q-values.

IncrementalAvgAgent(EpsilonGreedyAgent):
  A bandit agent that acts epsilon-greedily with respect to its Q-values.
  It uses the sample average method of estimating its Q-values, which is
  computed incrementally.

ConstStepAgent(EpsilonGreedyAgent):
  A bandit agent that acts epsilon-greedily with respect to its Q-values.
  It uses the constant step-size method of estimating its Q-values.

UCBAgent(BanditAgent):
  A class used to represent a bandit agent that uses the Upper Confidence Bound
  (UCB) action selection method. It uses the constant step-size method of
  updating its Q-values.

GradientAgent(BanditAgent)
  A class representing a bandit agent that instead of estimating action-values,
  learns preferences for actions and chooses them according to a softmax
  probability distribution.

ContextualBanditAgent
  A class that defines the interface for agents that act on contextual bandit
  problems - those with different states. Agents of this type interact
  with ContextualBandits.

ContextualEpsGreedyAgent(ContextualBanditAgent)
  A contextual bandit agent that uses the constant step-size update rule and
  acts epsilon-greedily.
"""

import numpy as np
from rl_learn.bandits import K  # Default k value

# Default number of states in contextual bandit problem
from rl_learn.bandits import N


class BanditAgent:
	"""
	A class used to represent an agent that acts on bandit problems - a
	particular type of environment where there is no concept of state.
	BanditAgents interact with environments of class Bandit. This class serves
	as an interface that all subclasses of BanditAgent must implement. The
	action space A is represented as {0, ..., k-1}, where k is the number of
	bandit arms.

	Methods
	-------
	BanditAgent(k=K)
	  Returns a new bandit agent that has k actions it can take, or bandit arms
	  to pull.

	get_action()
	  Returns the choice of action, or arm to pull.

	perform_update(action, reward)
	  Performs the changes desired by the agent given the action it took and
	  the reward it received so it can (hopefully) improve its decision making.
	
	get_alike_agent()
	  Returns a BanditAgent initialized with the same parameters

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment
	"""

	def __init__(self, k=K):
		"""
		Returns a new bandit agent that has k actions it can take, or bandit
		arms to pull.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment
		"""
		pass

	def get_action(self):
		"""
		Returns a choice of action, or arm to pull.

		Returns
		-------
		Int
		  An int in {0, ..., k-1} - the agent's choice of arm to pull
		"""
		return None

	def perform_update(self, action, reward):
		"""
		Performs the changes desired by the agent given the action it took and
		the reward it received so it can (hopefully) improve.

		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for taking the action
		"""
		pass

	def get_alike_agent(self):
		"""
		Returns a BanditAgent initialized with the same parameters.

		Returns
		-------
		BanditAgent
		  A reinitialized BanditAgent
		"""
		return None


class GreedyAgent(BanditAgent):
	"""
	A class representing a greedy bandit agent. It acts greedily with respect
	to its Q(a)s, which are computed via a sample average of the rewards
	received for taking each action.

	Methods
	-------
	GreedyAgent(k=K, q=0)
	  Returns a new Greedy agent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1}.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the reward received for taking
	  that action, using the sample average method.
	
	get_alike_agent()
	  Returns a new GreedyAgent initialized with the same parameters.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment
	"""

	def __init__(self, k=K, q=0):
		"""
		Returns a new Greedy agent with k arms. Its default Q-values are set
		to q, for each action a in {0, ..., k-1}.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment

		q : Real scalar
		  The value to initialize the Q(a)s with
		"""
		self.k = k
		self._q = q

		# We store the Q values as a list of 2-tuples where the
		# [0] element is the up-to-date Q value(a), the average of the
		# rewards received for taking action a, and the [2] element is
		# the list of numerical rewards received for taking action a.
		# We use the helper method _get_action_values to extract the
		# Q-values from this list.
		self._q_val_arr = [(q, []) for i in range(k)]

	def get_action(self):
		"""
		Returns the greedy action a* in {0, ..., k-1} with respect to the Q(a)s.
		
		Returns
		-------
		Int
		  An action a in {0, ..., k-1}
		"""
		q_values = self._get_action_values()

		return np.argmax(q_values)

	def perform_update(self, action, reward):
		"""
		Updates the estimate Q(action) based on the reward received for taking
		that action, using the sample average method.
		
		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for taking the action
		"""
		old_q, rewards = self._q_val_arr[action]
		rewards.append(reward)
		new_q = np.mean(rewards)
		self._q_val_arr[action] = (new_q, rewards)

	def get_alike_agent(self):
		"""
		Returns a new GreedyAgent initialized with the same parameters.

		Returns
		-------
		GreedyAgent
		  A reinitialized GreedyAgent
		"""
		cls = self.__class__

		return cls(self.k, self._q)

	def _get_action_values(self):
		"""
		Helper to return the Q(a)s as a list.
		"""
		return [a[0] for a in self._q_val_arr]


class ExploreGreedyAgent(GreedyAgent):
	"""
	A bandit agent that will act greedily only after trying each of its
	actions once. It uses the sample average method of estimating its Q-values.
	
	Methods
	-------
	ExploreGreedyAgent(k=K, q=0)
	  Returns a new ExploreGreedyAgent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1}, after trying each of
	  its actions once.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the reward received for taking
	  that action, using the sample average method.
	
	get_alike_agent()
	  Returns a new ExploreGreedyAgent initialized with the same parameters.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment
	"""

	def __init__(self, k=K, q=0):
		"""
		Returns a new ExploreGreedyAgent with k arms. Its default Q-values are
		set to q, for each action a in {0, ..., k-1}.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment

		q : Real scalar
		  The value to initialize the Q(a)s with
		"""
		super().__init__(k, q)
		self._action_index = 0

	def get_action(self):
		"""
		Returns the greedy action a* in {0, ..., k-1} after trying
		each action once.
		
		Returns
		-------
		Int
		  An action a in {0, ..., k-1}
		"""
		q_values = self._get_action_values()

		# Take the first action that hasn't been taken,
		# if one exists, otherwise the greedy action
		if self._action_index >= self.k:
			action = np.argmax(q_values)
		else:
			action = self._action_index

		self._action_index += 1

		return action


class EpsilonGreedyAgent(GreedyAgent):
	"""
	A bandit agent that acts epsilon-greedily with respect to its Q-values.
	With probability (1 - epsilon) it chooses the greedy action, and with small
	probability epsilon it acts randomly. It uses the sample average method of
	estimating its Q-values.

	Methods
	-------
	EpsilonGreedyAgent(k=K, q=0, epsilon=0.01)
	  Returns a new EpsilonGreedyAgent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1} with probability
	  (1 - epsilon) or a random action with probability epsilon.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the reward received for 
	  taking that action, using the sample average method.
	
	get_alike_agent()
	  Returns a new EpsilonGreedyAgent initialized with the same parameters.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment
	"""

	def __init__(self, k=K, q=0, epsilon=0.01):
		"""
		Returns a new EpsilonGreedyAgent with k arms. Its default Q-values are
		set to q, for each action a.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment

		q : Real scalar
		  The value to initialize the Q(a)s with

		epsilon : Real scalar in (0, 1)
		  The probability with which to act randomly
		"""
		super().__init__(k, q)
		self._epsilon = epsilon
		self._rng = np.random.default_rng()

	def get_action(self):
		"""
		Returns the greedy action a* in {0, ..., k-1} with probability
		(1 - epsilon) or a random action with small probability epsilon.

		Returns
		-------
		Int
		  An action a in {0, ..., k-1}
		"""
		num = self._rng.uniform(0, 1)

		if num > self._epsilon:
			q_values = self._get_action_values()
			action = np.argmax(q_values)
		else:
			action = self._rng.choice(self.k)

		return action

	def get_alike_agent(self):
		"""
		Returns a new EpsilonGreedyAgent initialized with the same parameters.
		
		Returns
		-------
		EpsilonGreedyAgent
		  A reinitialized EpsilonGreedyAgent
		"""
		cls = self.__class__

		return cls(self.k, self._q, self._epsilon)


class IncrementalAvgAgent(EpsilonGreedyAgent):
	"""
	A bandit agent that uses the sample average method of estimating its
	Q-values, which is computed incrementally. It acts epsilon-greedily with
	respect to its Q-values.

	Methods
	-------
	IncrementalAvgAgent(k=K, q=0, epsilon=0.01)
	  Returns a new IncrementalAvgEpsGreedyAgent agent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1} with probability
	  (1 - epsilon) or a random action with small probability epsilon.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the reward received for
	  taking that action using the sample average method, which is
	  computed incrementally.
	
	get_alike_agent()
	  Returns a new IncrementalAvgAgent initialized with the same parameters.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment
	"""

	def __init__(self, k=K, q=0, epsilon=0.01):
		"""
		Returns a new IncrementalAvgAgent agent with k arms.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment

		q : Real scalar
		  The value to initialize the Q(a)s with

		epsilon : Real scalar in (0, 1)
		  The probability with which to act randomly
		"""
		super().__init__(k, q, epsilon)

		# q_val_arr has the second element of each
		# tuple in the list as the number of times action
		# a was taken plus one (we use the notation Q1(a) = q
		# when action a has been taken 0 times, so we keep this
		# value consistent with the index for the Q-value). The
		# helper method _get_action_values as implemented by the parent
		# classes will still work with this design of _q_val_arr.
		self._q_val_arr = [(q, 1)] * k

	def perform_update(self, action, reward):
		"""
		Updates the estimate Q(action) based on the reward received for taking
		that action using the sample average method, which is
		computed incrementally.

		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for taking the action
		"""
		q_n_minus_one, n_minus_one = self._q_val_arr[action]
		q_n = q_n_minus_one + (1 / n_minus_one) * (reward - q_n_minus_one)
		n = n_minus_one + 1
		self._q_val_arr[action] = (q_n, n)


class ConstStepAgent(EpsilonGreedyAgent):
	"""
	A bandit agent that uses the incremental constant step-size update method
	of estimating its Q-values. It acts epsilon-greedily.

	Methods
	-------
	ConstStepAgent(k=K, q=0, epsilon=0.01, alpha=0.1)
	  Returns a new ConstStepAgent agent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1} with probability
	  (1 - epsilon) or a random action with small probability epsilon.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the reward received for
	  taking that action using the incremental constant step-size
	  update method.
	
	get_alike_agent()
	  Returns a new ConstStepAgent initialized with the same parameters.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment
	"""

	def __init__(self, k=K, q=0, epsilon=0.01, alpha=0.1):
		"""
		Returns a new ConstStepAgent with k arms.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment

		q : Real scalar
		  The value to initialize the Q(a)s with

		epsilon : Real scalar in (0, 1)
		  The probability with which to act randomly
		
		alpha : Real scalar > 0
		  The step-size constant
		"""
		super().__init__(k, q, epsilon)
		self._alpha = alpha

	def perform_update(self, action, reward):
		"""
		Updates the estimate Q(action) based on the reward received for taking
		that action, using the constant step-size update method.

		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for taking the action
		"""
		# To reuse code, we keep self._q_val_arr as [(Q(a), [Rewards(a)]],
		# but we don't ever store or use the [Rewards(a)].
		# Thus, we can change [Rewards(a)] to be some dummy value here.
		q_n_minus_one = self._get_action_values()[action]
		q_n = q_n_minus_one + (self._alpha * (reward - q_n_minus_one))
		self._q_val_arr[action] = (q_n, 0)

	def get_alike_agent(self):
		"""
		Returns a newly initialized ConstStepAgent using the
		same parameters.
		
		Returns
		-------
		ConstStepAgent
		  A reinitialized ConstStepAgent
		"""
		cls = self.__class__
		return cls(self.k, self._q, self._epsilon, self._alpha)


class UCBAgent(BanditAgent):
	"""
	A class used to represent a bandit agent that uses the Upper Confidence
	Bound (UCB) action selection method. It uses the constant step-size
	method of updating its Q-values.

	Methods
	-------
	UCBAgent(k=K, c=2, alpha=0.1)
	  Returns a new UCBAgent that has k actions it can take, or bandit
	  arms to pull.

	get_action()
	  Returns choice of action, or arm to pull using the UCB action
	  selection method.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the reward received for taking
	  that action, using the constant step-size update method.

	get_alike_agent()
	  Returns a UCBAgent initialized with the same parameters.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment
	"""

	def __init__(self, k=K, c=2, alpha=0.1):
		"""
		Returns a new UCBAgent that has k actions it can take, or bandit
		arms to pull.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment

		c : Real scalar > 0
		  The parameter in the UCB selection algorithm
		
		alpha : Real scalar > 0
		  The step-size constant
		"""
		self.k = k
		self._c = c
		self._alpha = alpha
		self._q_val_arr = [(0, 1)] * k  # [(Q(a), NumTaken(a)+1]
		self._t = 0

	def get_action(self):
		"""
		Selects the action according the UCB selection method. If any action
		has yet to be selected, it will choose that action.

		Returns
		-------
		Int in {0, ..., k-1}
		  The action chosen
		"""
		self._t += 1

		# Because our notation is that Q_1 is our first estimate
		# but no rewards have yet been received, action_counts[i] is
		# 1 higher than one might expect
		action_counts = np.array(self._get_action_counts())

		# Return the first action that hasn't been taken,
		# if one exists
		for i, n in enumerate(action_counts):
			if n == 1:
				return i

		q_vals = np.array(self._get_action_values())
		ucbs = q_vals + (self._c * (np.sqrt(np.log(self._t) / action_counts)))

		return np.argmax(ucbs)

	def perform_update(self, action, reward):
		"""
		Updates the estimate Q(action) based on the reward received for taking
		that action, using the constant step-size update method.

		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for taking the action
		"""
		q_n_minus_one, n_minus_one = self._q_val_arr[action]
		q_n = q_n_minus_one + (self._alpha * (reward - q_n_minus_one))
		n = n_minus_one + 1
		self._q_val_arr[action] = (q_n, n)

	def get_alike_agent(self):
		"""
		Returns a new UCBAgent initialized with the same parameters.

		Returns
		-------
		UCBAgent
		  A reinitialized UCBAgent
		"""
		cls = self.__class__

		return cls(self.k, self._c, self._alpha)

	def _get_action_values(self):
		"""
		A helper to return the Q-values as a list
		"""
		return [a[0] for a in self._q_val_arr]

	def _get_action_counts(self):
		"""
		A helper to return the number of times each action has been
		taken (plus one)
		"""
		return [a[1] for a in self._q_val_arr]


class GradientAgent(BanditAgent):
	"""
	An agent that chooses actions according to a softmax probability
	distribution. It learns action preferences using a stochastic gradient
	ascent update rule in an attempt to maximize its expected reward.
	It incrementally computes the average reward which is used as a baseline.

	Methods
	-------
	GradientAgent(k=K, h=0. alpha=0.1)
	  Returns a new GradientAgent that has k actions it can take, or bandit
	  arms to pull.

	get_action()
	  Returns the stochastic choice of action using a softmax probability
	  distribution calculated with its current action preferences.

	perform_update(action, reward)
	  Performs one step of stochastic gradient ascent to (hopefully) improve
	  its action preferences.

	get_alike_agent()
	  Returns a GradientAgent initialized with the same parameters.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment
	"""

	def __init__(self, k=K, h=0, alpha=0.1):
		"""
		Returns a new GradientAgent with k arms. Its default action preference
		values are all initialized to h.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment

		h : Real scalar
		  An initializing value for the action preferences

		alpha : Real scalar > 0
		  The step-size constant
		"""
		self.k = k
		self._h = h
		self._alpha = alpha
		self._average_tuple = (0, 0)  # (reward average, num actions taken)
		self._actions = list(range(k))
		self._preferences = np.full(k, h, dtype=np.float32)  # array([h, ,,, h]

		# Initialize softmax probabilities to zeros,
		# then make them correct using the helper
		self._softmax_probs = np.zeros(shape=k)
		self._sync_softmax_probs()

	def get_action(self):
		"""
		Returns the stochastic choice of action using a softmax probability
		distribution calculated with its current action preferences.

	    Returns
	    -------
	    Int in {0, ..., k-1}
	      The action taken
		"""
		action = np.random.choice(self._actions, p=self._softmax_probs)

		return action

	def perform_update(self, action, reward):
		"""
	    Performs one step of stochastic gradient ascent to (hopefully) improve
	    its action preferences.

		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action taken

		reward : Real scalar
		  The reward received for taking the action
		"""
		avg, n = self._average_tuple
		alpha = self._alpha
		a, r = (action, reward)

		update = (-1) * alpha * (r - avg) * self._softmax_probs
		update[a] = alpha * (r - avg) * (1 - self._softmax_probs[a])
		self._preferences += update

		self._sync_softmax_probs()
		n += 1
		avg += (1/n) * (reward - avg)
		self._average_tuple = (avg, n)

	def get_alike_agent(self):
		"""
	    Returns a GradientAgent initialized with the same parameters.

		Returns
		-------
		GradientAgent
		  A reinitialized GradientAgent
		"""
		return GradientAgent(self.k, self._h, self._alpha)

	def _sync_softmax_probs(self):
		"""
		A helper to keep self._softmax_probs consistent with the current
		action preference values.
		"""
		exps = np.exp(self._preferences)
		s = sum(exps)
		softmax_probs = exps / s
		self._softmax_probs = softmax_probs


class ContextualBanditAgent:
	"""
	A class that defines the interface that a contextual bandit agent must
	implement. A contextual bandit problem is one in which there is a concept
	of state, but it is simply an indication of which bandit the agent
	is facing. Actions have no effect on the future rewards received, only the
	next reward, and have no impact on what states the agent ends up in.
	ContextualBanditAgents will interact with environments of type
	ContextualBandit. The action space A is represented as {0, ..., k-1},
	and the state space is represented as {0, ..., n-1}, where k is the
	number of bandit arms and n is the number of states.

	Methods
	-------
	ContextualBanditAgent(k=K, n=N)
	  Returns a new ContextualBanditAgent that has k arms it can pull in
	  each of the n states.

	get_action(state)
	  Returns the arm to pull given the state the agent is in.

	perform_update(state, action, reward)
	  Performs the changes desired by the agent given the action it took,
	  the state it was in, and the reward it received so it can (hopefully)
	  improve its decision making.

	get_alike_agent()
	  Returns a ContextualBanditAgent initialized with the same parameters.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment

	n : Int >= 2
	  The number of different states, which must match the environment
	"""

	def __init__(self, k=K, n=N):
		"""
		Returns a new ContextualBanditAgent that has k actions it can take in
		each of the N states.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment
		
		n : Int >= 2
		  The number of different states, which must match the environment
		"""
		pass

	def get_action(self, state):
		"""
		Returns a choice of action, or arm to pull given the current state
		of the environment.
		
		Parameters
		----------
		state : Int in {0, ..., n-1}
		  The current state of the environment

		Returns
		-------
		Int
		  An int in {0, ..., k-1} - the agent's choice of arm to pull
		"""
		return None

	def perform_update(self, state, action, reward):
		"""
		Performs the changes desired by the agent given the action it took,
		the state the agent was in and the reward it received so it can
		(hopefully) improve.

		Parameters
		----------
		state : Int in {0, ..., n-1}
		  The state the agent was in
		  
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for taking the action
		"""
		pass

	def get_alike_agent(self):
		"""
		Returns a ContextualBanditAgent initialized with the same parameters

		Returns
		-------
		ContextualBanditAgent
		  A reinitialized ContextualBanditAgent
		"""
		return None


class ContextualEpsGreedyAgent(ContextualBanditAgent):
	"""
	A contextual bandit agent that acts epsilon-greedily with respect to its
	Q(state, action)-values. It uses the constant step-size method of
	updating them.
	
	Methods
	-------
	ContextualEpsGreedyAgent(k=K, n=N, q = 0, epsilon=0.01, alpha=0.1)
	  Returns a new ContextualEpsGreedyAgent that has k arms it can pull in
	  each of the n states.

	get_action(state)
	  Returns the arm to pull given the state the agent is in by acting
	  epsilon-greedily.

	perform_update(state, action, reward)
	  Updates Q(state, action) based on the reward the agent received for
	  taking action in state.
	
	get_alike_agent()
	  Returns a ContextualEpsGreedyAgent initialized with the same parameters

	Parameters
	----------
	k : Int >= 2
	  The number of arms, which must match the environment

	n : Int >= 2
	  The number of different states, which must match the environment
	"""

	def __init__(self, k=K, n=N, q=0, epsilon=0.01, alpha=0.1):
		"""
		Returns a new ContextualEpsGreedyAgent that has k actions it can take
		in each of the n states.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, which must match the environment

		n : Int >= 2
	  	  The number of different states, which must match the environment

		q : Real scalar
		  The Q(state, value) initializing value

		epsilon : Real scalar in (0, 1)
		  The probability with which to act randomly

		alpha : Real scalar > 0
		  The constant step-size parameter
		"""
		self.k = k
		self.n = n
		self._q = q
		self._epsilon = epsilon
		self._alpha = alpha

		# _q_val_arr is [ [(Q(s, a), num_taken(s, a)+1)] ]
		self._q_val_arr = [[(q, 1) for a in range(k)] for s in range(n)]

	def get_action(self, state):
		"""
		Returns a choice of action, or arm to pull given the current state of
		the environment, by acting epsilon-greedily.
		
		Parameters
		----------
		state : Int in {0, ..., n-1}
		  The current state of the environment

		Returns
		-------
		Int
		  An int in {0, ..., k-1} - the agent's choice of arm to pull
		"""
		q_vals = self._get_q_vals(state)
		num = np.random.uniform()

		if num < self._epsilon:
			return np.random.choice(self.k)
		else:
			return np.argmax(q_vals)

	def perform_update(self, state, action, reward):
		"""
		Updates the Q(state, action) value using the constant step-size method.

		Parameters
		----------
		state : Int in {0, ..., n-1}
		  The state the agent was in
		  
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for taking the action
		"""
		q_n_minus_one, n_minus_one = self._q_val_arr[state][action]
		q_n = q_n_minus_one + self._alpha * (reward - q_n_minus_one)
		n = n_minus_one + 1
		self._q_val_arr[state][action] = (q_n, n)

	def get_alike_agent(self):
		"""
		Returns a ContextualEpsGreedyAgent initialized with the same parameters

		Returns
		-------
		ContextualEpsGreedyAgent
		  A reinitialized ContextualEpsGreedyAgent
		"""
		cls = self.__class__
		return cls(self.k, self.n, self._q, self._epsilon, self._alpha)

	def _get_q_vals(self, state):
		"""
		Helper to return the Q(a) values for a given state
		"""
		pairs = self._q_val_arr[state]

		return [pair[0] for pair in pairs]



