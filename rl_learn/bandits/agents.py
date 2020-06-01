"""
agents

A module that defines the interface that any
BanditAgent must provide, as well as provide some
sample ones.

Classes
-------
BanditAgent
  The interface that all agents must implement
  so that they can interact with bandit environments.

GreedyAgent
  A bandit agent that acts greedily with respect to its Q(a)s, 
  which are computed via a sample average of the rewards 
  recieved for taking each action.
  
ExploreGreedyAgent(GreedyAgent):
  A bandit agent that will act greedily only after trying 
  each of its actions once. It uses the sample average method 
  of estimating its Q-values.

EpsilonGreedyAgent(GreedyAgent):
  A bandit agent that acts epsilon-greedily with respect to its
  Q-values. With probability (1 - epsilon) it chooses the greedy 
  action, and with small probability epsilon it acts randomly. 
  It uses the sample average method of estimating its Q-values.

IncrementalAvgEpsGreedyAgent(EpsilonGreedyAgent):
  A bandit agent that acts epsilon-greedily with respect to its
  Q-values. It uses the sample average method of estimating 
  its Q-values, which is computed incrementally.

IncrementalConstEpsGreedyAgent(EpsilonGreedyAgent):
  A bandit agent that acts epsilon-greedily with respect to its
  Q-values. It uses the incremental constant step-size update method 
  of estimating its Q-values.

UCBAgent(BanditAgent):
  A class used to represent a bandit agent that
  uses the Upper Confidence Bound (UCB) action
  selection method. It uses the constant step-size
  method of updating its Q-values.
"""

import numpy as np
from rl_learn.bandits import K ## Default k value


class BanditAgent:
	'''
	A class used to represent an agent that
	acts on bandit problems - a particular type of 
	environment where every state is the same,
	and thus, there is no concept of state.
	They interact with environments of class Bandit.
	This class serves as an interface that all
	subclasses of BanditAgent must implement.

	Methods
	-------
	BanditAgent(k=K)
	  Returns a new bandit agent that has k actions
	  it can take, or bandit arms to pull.

	get_action()
	  Returns the choice of action, or arm to pull.

	perform_update(action, reward)
	  Performs the changes desired by the agent given
	  the action it took and the reward it received so
	  it can (hopefully) improve its decision making.
	
	get_alike_agent()
	  Returns a BanditAgent initialized with 
	  the same parameters
	'''

	def __init__(self, k=K):
		"""
		Returns a new bandit agent that has k actions
		it can take, or bandit arms to pull.

		Parameters
		----------
		k : Int
		  The number of arms to pull defined by the
		  Bandit environment
		"""
		pass

	def get_action(self):
		"""
		Returns a choice of action, or arm to pull.
		Actions are represented as integers in the
		set {0, ..., k-1}, where k is the number of
		bandit arms in the Bandit environment.

		Returns
		-------
		Int
		  An int in {0, ..., k-1} representing the
		  agent's choice of arm to pull
		"""
		return None

	def perform_update(self, action, reward):
		"""
		Performs the changes desired by the agent given
		the action it took and the reward it received so
		it can (hopefully) improve.

		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for
		  taking the action
		"""
		pass

	def get_alike_agent(cls):
		"""
		Returns a BanditAgent initialized
		with the same parameters

		Returns
		-------
		BanditAgent
		  A reinitialized BanditAgent
		"""
		return None

class GreedyAgent(BanditAgent):
	"""
	A class representing a greedy bandit agent. It acts greedily
	with respect to its Q(a)s, which are computed via a sample 
	average of the rewards recieved for taking each action.

	Methods
	-------
	GreedyAgent(k=K, q=0)
	  Returns a new Greedy agent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1}.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the
	  reward received for taking that action, using
	  the sample average method.
	
	get_alike_agent()
	  Returns a new GreedyAgent initialized with the same parameters.
	"""

	def __init__(self, k=K, q=0):
		"""
		Returns a new Greedy agent with k arms. Its default
		Q-values are set to q, for each action a in {0, ..., k-1}

		Parameters
		----------
		k : Int
		  The number of arms to pull defined by the
		  Bandit environment

		q : Real scalar
		  The value to initialize the Q(a)s with
		"""
		self._k = k
		self._q = q

		## We store the Q values as a list of 2-tuples where the
		## [0] element is the up-to-date Q value(a), the average of the
		## rewards received for taking action a, and the [2] element is
		## the list of numerical rewards received for taking action a.
		## We use the helper method _get_action_values to extract the list
		## of Q-values from this list.
		self._q_val_arr = [(q, []) for i in range(k)]

	def get_action(self):
		"""
		Returns the greedy action a* in {0, ..., k-1} with
		respect to the Q(a)s.
		
		Returns
		-------
		Int
		  An action a in {0, ..., k-1}
		"""
		Q_values = self._get_action_values()

		return np.argmax(Q_values)

	def get_alike_agent(self):
		"""
		Returns a new GreedyAgent initialized with the same parameters.

		Returns
		-------
		GreedyAgent
		  A reinitialized GreedyAgent
		"""
		cls = self.__class__
		
		return cls(self._k, self._q)

	def perform_update(self, action, reward):
		"""
		Updates the estimate Q(action) based on the
		reward received for taking that action, using
		the sample average method.
		
		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for
		  taking the action
		"""
		old_q, rewards = self._q_val_arr[action]
		rewards.append(reward)
		new_q = np.mean(rewards)
		self._q_val_arr[action] = (new_q, rewards)

	def _get_action_values(self):
		"""
		Helper to return the Q(a)s as a list.
		"""
		return [i[0] for i in self._q_val_arr]


class ExploreGreedyAgent(GreedyAgent):
	"""
	A bandit agent that will act greedily only after trying 
	each of its actions once. It uses the sample average method 
	of estimating its Q-values. It keeps the implementations of 
	get_alike_agent and perform_update from GreedyAgent.
	
	Methods
	-------
	ExploreGreedyAgent(k=K, q=0)
	  Returns a new Greedy agent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1}, after
	  trying each of its actions once.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the
	  reward received for taking that action, using
	  the sample average method.
	
	get_alike_agent()
	  Returns a new ExploreGreedyAgent initialized with the 
	  same parameters.
	"""
	
	def __init__(self, k=K, q=0):
		"""
		Returns a new ExploreGreedyAgent with k arms. Its default
		Q-values are set to q, for each action a in {0, ..., k-1}.

		Parameters
		----------
		k : Int
		  The number of arms to pull defined by the
		  Bandit environment

		q : Real scalar
		  The value to initialize the Q(a)s with
		"""
		super().__init__(k, q)
		self._action_index = 0

	def get_action(self):
		"""
		Returns the greedy action a* in A = {0, ..., k-1}
		after trying each action once.
		
		Returns
		-------
		Int
		  An action a in {0, ..., k-1}
		"""
		Q_values = self._get_action_values()
		
		if self._action_index >= self._k:
			action = np.argmax(Q_values)
		else:
			action = self._action_index
			
		self._action_index += 1
		
		return action
		
class EpsilonGreedyAgent(GreedyAgent):
	"""
	A bandit agent that acts epsilon-greedily with respect to its
	Q-values. With probability (1 - epsilon) it chooses the greedy 
	action, and with small probability epsilon it acts randomly. 
	It uses the sample average method of estimating its Q-values. 
	It keeps the implementation of perform_update from GreedyAgent.

	Methods
	-------
	EpsilonGreedyAgent(k=K, q=0, epsilon=0.01)
	  Returns a new EpsilonGreedyAgent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1} with probability
	  (1 - epsilon) and a random action with probability epsilon. 

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the reward received for 
	  taking that action, using the sample average method.
	
	get_alike_agent()
	  Returns a new EpsilonGreedyAgent initialized with the same parameters.
	"""

	def __init__(self, k=K, q=0, epsilon=0.01):
		"""
		Returns a new EpsilonGreedyAgent with k arms. Its default
		Q-values are set to q, for each action a. 

		Parameters
		----------
		k : Int
		  The number of arms to pull defined by the
		  Bandit environment

		q : Real scalar
		  The value to initialize the Q(a)s with

		epsilon : Float, 0 <= epsilon <= 1
		  The probability with which to act randomly
		"""
		super().__init__(k, q)
		self._epsilon = epsilon 
		self._rng = np.random.default_rng()

	def get_action(self):
		"""
		Returns the greedy action a* in {0, ..., k-1} with probability
		(1 - epsilon) and a random action with small probability epsilon.

		Returns
		-------
		Int
		  An action a in {0, ..., k-1}
		"""
		num = self._rng.uniform(0, 1)

		if (num > self._epsilon):
		    Q_values = self._get_action_values()
		    action = np.argmax(Q_values)
		else:
		    action = self._rng.choice(self._k)

		return action

	def get_alike_agent(self):
		"""
		Returns a new EpsilonGreedyAgent initialized with 
		the same parameters.
		
		Returns
		-------
		EpsilonGreedyAgent
		  A reinitialized EpsilonGreedyAgent
		"""
		cls = self.__class__
		
		return cls(self._k, self._q, self._epsilon)


class IncrementalAvgEpsGreedyAgent(EpsilonGreedyAgent):
	"""
	A bandit agent that acts epsilon-greedily with respect to its
	Q-values. It uses the sample average method of estimating 
	its Q-values, which is computed incrementally. It keeps the 
	implementations of get_action and get_alike_agent from
	EpsilonGreedyAgent.

	Methods
	-------
	IncrementalAvgEpsGreedyAgent(k=K, q=0, epsilon=0.01)
	  Returns a new IncrementalAvgEpsGreedyAgent agent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1} with probability
	  (1 - epsilon) and a random action with small probability epsilon. 

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the
	  reward received for taking that action, using
	  the sample average method, which is computed incrementally.
	
	get_alike_agent()
	  Returns a new IncrementalAvgEpsGreedyAgent initialized with 
	  the same parameters.
	"""
	def __init__(self, k=K, q=0, epsilon=0.01):
		"""
	  	Returns a new IncrementalAvgEpsGreedyAgent agent 
	  	with k arms.
	  	
	  	Parameters
		----------
		k : Int
		  The number of arms to pull defined by the
		  Bandit environment

		q : Real scalar
		  The value to initialize the Q(a)s with

		epsilon : Float, 0 <= epsilon <= 1
		  The probability with which to act randomly
		"""
		super().__init__(k, q, epsilon)
		
		## q_val_arr has the second element of each
		## tuple in the list as the number of times action
		## a was taken plus one (we use the notation Q1(a) = q 
		## when action a has been taken 0 times, so we keep this
		## value consistent with the index for the Q-value). The
		## helper method _get_action_values as implemented by the parent
		## classes will still work with this new implementation.
		self._q_val_arr = [(q, 1)] * k

	def perform_update(self, action, reward):
		"""
		Updates the estimate Q(action) based on the
	  	reward received for taking that action, using
	  	the sample average method, which is computed incrementally.
	  	
	  	Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for
		  taking the action
	  	"""
		Q_n_minus_one, n_minus_one = self._q_val_arr[action]
		Q_n = Q_n_minus_one + (1/n_minus_one) * (reward - Q_n_minus_one)
		n = n_minus_one + 1
		self._q_val_arr[action] = (Q_n, n)


class IncrementalConstEpsGreedyAgent(EpsilonGreedyAgent):
	"""
	A bandit agent that acts epsilon-greedily with respect to its
	Q-values. It uses the incremental constant step-size update method 
	of estimating its Q-values. It keeps the implementation of get_action 
	from its parent classes.

	Methods
	-------
	IncrementalConstEpsGreedyAgent(k=K, q=0, epsilon=0.01, alpha=0.1)
	  Returns a new Greedy agent with k arms.

	get_action()
	  Returns the greedy action a* in {0, ..., k-1} with probability
	  epsilon and a random action with probability (1 - epsilon). 

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the
	  reward received for taking that action, using
	  the incremental constant update method.
	
	get_alike_agent()
	  Returns a new IncrementalConstEpsGreedyAgent initialized 
	  with the same parameters.
	"""
	def __init__(self, k=K, q=0, epsilon=0.01, alpha=0.1):
		"""
		IncrementalAvgEpsGreedyAgent(k=K, q=0, epsilon=0.01)
	  	Returns a new IncrementalAvgEpsGreedyAgent agent with k arms.
	  	
	  	Parameters
		----------
		k : Int
		  The number of arms to pull defined by the
		  Bandit environment

		q : Real scalar
		  The value to initialize the Q(a)s with

		epsilon : Real scalar, 0 <= epsilon <= 1
		  The probability with which to act randomly
		
		alpha : Real scalar
		  The constant step-size update parameter
		"""
		super().__init__(k, q, epsilon)
		self._alpha = alpha

	def perform_update(self, action, reward):
		"""
		Updates the estimate Q(action) based on the
	  	reward received for taking that action, using
	  	the constant step-size update method.
	  	
	  	Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for
		  taking the action
	  	"""
	  	## To reuse code, we keep self._q_val_arr as [(Q(a), [Rewards(a)]],
	  	## but we don't actually store or use the [Rewards(a)] in any method.
	  	## Thus, we can change [Rewards] to be some dummy value here.
		Q_n_minus_one = self._get_action_values()[action]
		Q_n = Q_n_minus_one + (self._alpha * (reward - Q_n_minus_one))
		self._q_val_arr[action] = (Q_n, 0)
	
	def get_alike_agent(self):
		"""
		Returns a newly initialized IncrementalConstEpsGreedyAgent
		using the same parameters.
		
		Returns
		-------
		IncrementalConstEpsGreedyAgent
		  A reinitialized agent
		"""
		cls = self.__class__
		return cls(self._k, self._q, self._epsilon, self._alpha)


class UCBAgent(BanditAgent):
	'''
	A class used to represent a bandit agent that
	uses the Upper Confidence Bound (UCB) action
	selection method. It uses the constant step-size
	method of updating its Q-values.

	Methods
	-------
	UCBAgent(k=K, c=2, alpha=0.1)
	  Returns a new UCBAgent that has k actions
	  it can take, or bandit arms to pull. 

	get_action()
	  Returns choice of action, or arm to pull using the
	  UCB action selection method.

	perform_update(action, reward)
	  Updates the estimate Q(action) based on the
	  reward received for taking that action, using
	  the constant step-size update method.

	get_alike_agent()
	  Returns a UCBAgent initialized with the same parameters.
	'''

	def __init__(self, k=K, c=2, alpha=0.1):
		"""
		Returns a new UCBAgent that has k actions
		it can take, or bandit arms to pull.

		Parameters
		----------
		k : Int
		  The number of arms to pull defined by the
		  Bandit environment

		c : Real scalar > 0
		  The parameter in the UCB selection algorithm
		
		alpha : Real scalar > 0
		"""
		self._k = k
		self._c = c
		self._alpha = alpha
		self._q_val_arr = [(0, 1)] * k # [(Q(a), NumTaken(a)+1]
		self._t = 0

	def get_action(self):
		"""
		Selects the action according the UCB selection method. If any 
		action has yet to be selected, it will choose that action.

		Returns
		-------
		Int in {0, ..., k-1}
		  The action chosen
		"""
		self._t += 1

		## Because our notation is that Q_1 is our first estimate 
		## but no rewards have yet been received, action_counts[i] is 
		## 1 higher than one might expect
		action_counts = np.array(self._get_action_counts())

		## Return the first action that hasn't been taken,
		## if one exists
		for i, n in enumerate(action_counts):
			if n == 1:
				return i
			
		q_vals = np.array(self._get_action_values())
		UCBs = q_vals + (self._c * (np.sqrt(np.log(self._t) / action_counts)))

		return np.argmax(UCBs)

	def perform_update(self, action, reward):
		"""
		Updates the estimate Q(action) based on the
		reward received for taking that action, using
		the constant step-size update method.

		Parameters
		----------
		action : Int in {0, ..., k-1}
		  The action the agent took

		reward : Real scalar
		  A numerical reward the agent received for
		  taking the action
		"""
		Q_n_minus_one, n_minus_one = self._q_val_arr[action]
		Q_n = Q_n_minus_one + (self._alpha * (reward - Q_n_minus_one))
		n = n_minus_one + 1
		self._q_val_arr[action] = (Q_n, n)

	def get_alike_agent(self):
		"""
		Returns a new UCBAgent initialized with the same parameters.

		Returns
		-------
		UCBAgent
		  A reinitialized UCBAgent
		"""
		cls = self.__class__
		
		return cls(self._k, self._c, self._alpha)

	def _get_action_values(self):
		"""
		A helper to return the Q-values
		"""
		return [x[0] for x in self._q_val_arr]

	def _get_action_counts(self):
		"""
		A helper to return the number of times
		each action has been taken (plus one)
		"""
		return [x[1] for x in self._q_val_arr]
