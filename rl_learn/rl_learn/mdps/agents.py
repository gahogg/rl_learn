import numpy as np
from copy import deepcopy


class ModelAgent:
	"""
	A class used to represent an agent that has a full model of its environment.
	ModelAgents interact with environments of class MDP. The action space A 
	is represented as {0, ..., A-1}, and the state space {0, ..., S-1}.

	Methods
	-------
	ModelAgent(mdp)
	  Returns a ModelAgent.

	get_action(s)
	  Returns the choice of action.

	train()
	  Performs the changes desired by the agent in an offline fashion, so the
	  agent can "learn" the MDP it's up against.
	
	get_alike_agent()
	  Returns a ModelAgent initialized with the same parameters.
	"""

	def __init__(self, mdp, gamma=.95):
		"""
		Returns a new ModelAgent.

		Parameters
		----------
		mdp : MDP
		  The mdp the agent will be interacting with
		
		gamma : Float in (0, 1)
		  The discount factor
		"""
		pass

	def get_action(self, s):
		"""
		Returns a choice of action in state s.

		Returns
		-------
		Int
		  An int in {0, ..., A-1}
		"""
		return None

	def train(self):
		"""
		Performs the changes desired by the agent in an offline fashion, so the
		agent can "learn" the MDP it's up against.
		"""
		pass

	def get_alike_agent(self):
		"""
		Returns a ModelAgent initialized with the same parameters.

		Returns
		-------
		BanditAgent
		  A reinitialized BanditAgent
		"""
		return None

class PolicyIterationAgent(ModelAgent):
	"""
	A ModelAgent that uses the Policy Iteration (with q-values) algorithm to learn its policy.

	Methods
	-------
	PolicyIterationAgent(mdp)
	  Returns a PolicyIterationAgent.

	get_action(s)
	  Returns the choice of action.

	train()
	  Performs the changes desired by the agent in an offline fashion, so the
	  agent can "learn" the MDP it's up against.
	
	get_alike_agent()
	  Returns a ModelAgent initialized with the same parameters.
	"""

	def __init__(self, mdp, tolerance=.1, gamma=.95):
		"""
		Returns a new PolicyIterationAgent.

		Parameters
		----------
		mdp : MDP
		  The mdp the agent will be interacting with
		
		tolerance : Float >= 0
		  The tolerance for value evaluation changes

		gamma : Float in (0, 1)
		  The discount factor
		"""
		self._mdp = mdp
		self._dynamics = mdp.dynamics
		self._tolerance = tolerance
		self._gamma = gamma
		self._S = mdp.S
		self._A = mdp.A
		self._R = mdp.R
		self._possible_rewards = mdp.rewards
		self._q = np.zeros(shape=(self._S, self._A), dtype=np.float)
		self._pi = np.zeros(shape=(self._S), dtype=np.int)

	def get_action(self, s):
		"""
		Returns a choice of action in state s.

		Parameters
		----------
		s : Int in {0, ..., S-1}
		  The state the agent is currently in

		Returns
		-------
		Int
		  An action in {0, ..., A-1}
		"""
		return self._pi[s]

	def train(self):
		"""
		Performs the changes desired by the agent in an offline fashion, so the
		agent can "learn" the MDP it's up against. This agent uses the Policy Iteration
		algorithm to learn its policy.
		"""
		self._value_evaluation()
		pi_is_stable = self._policy_improvement()

		if pi_is_stable:
			return
		else:
			self.train()


	def get_alike_agent(self):
		"""
		Returns a ModelAgent initialized with the same parameters.

		Returns
		-------
		ModelAgent
		  A reinitialized ModelAgent
		"""
		cls = self.__class__

		return cls(self._mdp, self._tolerance, self._gamma)
	

	def _value_evaluation(self):
		"""
		Evaluates the agent's current policy until its given tolerance is satisfied.
		"""
		tol = self._tolerance
		dynamics = self._dynamics
		S = self._S
		A = self._A
		R = self._R
		gamma = self._gamma
		possible_rewards = self._possible_rewards
		new_qs = np.zeros(shape=(S, A), dtype=np.float)

		for s in range(S):
			for a in range(A):
				for s_prime in range(S):
					for r in range(R):
						joint_prob = dynamics[s][a][s_prime][r]
						target = possible_rewards[r] + (gamma * self._q[s_prime, self._pi[s_prime]])
						new_qs[s, a] = new_qs[s, a] + joint_prob*target
		
		biggest_change = np.max(np.abs(new_qs - self._q))
		self._q = new_qs

		if biggest_change > tol:
			self._value_evaluation()
		
		return
	
	def _policy_improvement(self):
		"""
		Makes the policy greedy with respect to its q-value function.
		Returns True iff no changes were made, else False
		"""
		policy = deepcopy(self._pi)
		S = self._S

		for s in range(S):
			self._pi[s] = np.argmax(self._q[s])
		
		for s in range(S):
			if self._pi[s] != policy[s]:
				return False
		
		return True

class RandomAgent(ModelAgent):
	"""
	A ModelAgent that acts randomly.

	Methods
	-------
	RandomAgent(mdp)
	  Returns a RandomAgent.

	get_action(s)
	  Returns the choice of action, which is random.

	train()
	  For the random agent, does nothing
	
	get_alike_agent()
	  Returns a RandomAgent initialized with the same parameters.
	"""

	def __init__(self, mdp, tolerance=.1, gamma=.95):
		"""
		Returns a new RandomAgent.

		Parameters
		----------
		mdp : MDP
		  The mdp the agent will be interacting with
		
		tolerance : Float >= 0
		  The tolerance for value evaluation changes

		gamma : Float in (0, 1)
		  The discount factor
		"""
		self._A = mdp.A

	def get_action(self, s):
		"""
		Returns a random choice of action in state s.

		Parameters
		----------
		s : Int in {0, ..., S-1}
		  The state the agent is currently in

		Returns
		-------
		Int
		  An action in {0, ..., A-1}
		"""
		return np.random.choice(self._A)

	def train(self):
		"""
		The random agent doesn't do much to train...
		"""
		pass


	def get_alike_agent(self):
		"""
		Returns a RandomAgent initialized with the same parameters.

		Returns
		-------
		RandomAgent
		  A reinitialized RandomAgent
		"""
		cls = self.__class__

		return cls(self._mdp)


class ModelFreeAgent:
	"""
	An abstract class used to represent an agent that has no model of its environment.
	Although it takes as input an MDP, it samples from it instead of peeking at the dynamics, and records the
	the size of the action and state spaces. They also don't attempt to learn a model of the environment,
	and therefore do not incorporate planning. Their goal is to learn action values. 
	ModelFreeAgents interact with environments of class MDP. The action space A is represented as {0, ..., A-1}, 
	and the state space {0, ..., S-1}.

	Methods
	-------
	ModelFreeAgent(mdp, num_games=1000)
	  Returns a ModelFreeAgent.

	get_action(s)
	  Returns the choice of action.

	train()
	  Plays many games in the environment to learn action values.
	
	get_alike_agent()
	  Returns a ModelFreeAgent initialized with the same parameters.
	"""
	pass

class OnPolicyMCAgent(ModelFreeAgent):
	'''
	A OnPolicyMCAgent that uses the Monte Carlo method to estimate its action values, acting on policy.

	Methods
	-------
	OnPolicyMCAgent(mdp, num_games=1000, epsilon=.01)
	  Returns a OnPolicyMCAgent.

	get_action(s)
	  Returns the choice of action.

	train()
	  Plays many games in the environment to learn action values using Monte Carlo estimation.
	
	get_alike_agent()
	  Returns a ModelFreeAgent initialized with the same parameters.
	'''

	def __init__(self, mdp, num_games=1000):
		self._S = mdp.S
		self._A = mdp.A
		self._action_values = np.zeros(shape=(self._S, self._A))
		self._visited_num = np.zeros(shape=(self._S, self._A), dtype=np.int)
		self._mdp = mdp
		self._num_games = num_games
		self._pi = np.zeros(self._S)
	
	def get_action(self, s):
		return np.argmax(self._pi)

	def train(self):
		'''
		Plays many games in the environment to learn action values using Monte Carlo estimation.
		'''
		for _ in self._num_games:
			s_a_returns = np.zeros(shape=(self._S, self._A))
			visited = np.zeros(shape=(self._S, self._A), dtype=np.bool)

			for s, a, s_prime, r in self._generate_episode():
				visited[s, a] = True
				s_a_returns[visited] += r
			
			self._action_values = self._action_values + (1/self._visited_num)*(r - self._action_values)
			self._policy_improvement()


	def _generate_episode(self):
		'''
		Plays a single game with the MDP, follwing its current policy.
		Uses exploring (random) starts.

		Returns List[(State, Action, State, Reward)]
		'''
		s = np.random.choice(self._S)
		episode = []

		while True:
			a = self.get_action(s)
			s = self._S
			r, s_prime = self._mdp.interact(s, a)

			# Acted in terminal state
			if r == -10000:
				break
			else:
				episode.append((s, a, s_prime, r))
				s = s_prime

		return episode
	
	def _policy_improvement(self):
		"""
		Makes the policy greedy with respect to its q-value function.
		Returns True iff no changes were made, else False
		"""
		policy = deepcopy(self._pi)
		S = self._S

		for s in range(S):
			self._pi[s] = np.argmax(self._action_values[s])
		
		for s in range(S):
			if self._pi[s] != policy[s]:
				return False
		
		return True




