"""
environments

A module that defines an interface that all bandit environments must provide,
as well as provide some sample ones. They interact with agents in the
agents module.

Classes
-------
Bandit
  A class that defines the interface that any subclass of Bandit must
  provide, which is for a standard bandit environment that has
  no concept of state. They interact with agents of type BanditAgent.

KArmedTestbed(Bandit)
  A bandit environment that returns rewards according to fixed
  normal distributions.

KArmedTestbedNonStationary(KArmedTestbed)
  A variation of the k-armed testbed where the distributions are non-stationary.

ContextualBandit
  An environment that incorporates state into bandit problems. These bandits
  require a slightly different interface than subclasses of the bandit type.
  This class serves as both the interface and an implementation of the methods.
  ContextualBandits interact with agents of type ContextualBanditAgent.
"""

import numpy as np
from rl_learn.bandits import K, N  # Default number of arms and states


class Bandit:
	"""
	A class used to represent a Bandit environment where there is no concept of
	state. Bandits interact with agents of class BanditAgent. This class
	serves as an interface that all subclasses of Bandit must implement. The
	action space A is represented as {0, ..., k-1}, where k is the number of
	bandit arms.

	Methods
	-------
	Bandit(k=K)
	  Returns an instance of a Bandit with k arms.

	interact(a)
	  Performs action a and returns the numerical reward for doing so.

	get_alike_bandit()
	  Returns a Bandit that is initialized with the same parameters as the
	  instance that used this method.

	Parameters
	----------
	k : Int >= 2
	  The number of arms of the k-armed bandit
	"""

	def __init__(self, k=K):
		"""
		Returns an instance of a Bandit with k arms.

		Parameters
		----------
		k : Int >= 2
		  The number of arms of the k-armed bandit
		"""
		self.k = k

	def interact(self, a):
		"""
		Performs action a in the environment and returns the numerical reward
		for doing so.

		Parameters
		----------
		a : Int in {0, ..., k-1}
		  The choice of arm to pull 

		Returns
		-------
		Real scalar
		  The real-valued reward for taking action a
		"""
		return None

	def get_alike_bandit(self):
		"""
		Returns a Bandit that is initialized with the same parameters as the
		instance that used this method.

		Returns
		-------
		Bandit
		  A reinitialized Bandit instance with the same parameters
		"""
		return None


class KArmedTestbed(Bandit):
	"""
	A class used to represent the specific bandit problem of the k-armed
	testbed. Given k, mu, sigma1, and sigma2, a KArmedTestbed will be
	constructed where the reward for taking an action a is a draw from its
	unique normal distribution with mean Q*(a) and standard deviation sigma2,
	where Q*(a) is initialized randomly from a normal distribution with
	mean mu and standard deviation sigma1. 

	Methods
	-------
	KArmedTestbed(k=K, mu=0, sigma1=1, sigma2=1)
	  Returns a new testbed with k arms.

	interact(a)
	  Performs action a in the testbed. It returns the reward the agent
	  received for taking the action.

	get_alike_bandit()
	  Returns a KArmedTestbed initialized with the same parameters.
	
	Parameters
	----------
	k : Int >= 2
	  The number of arms, k, of the k-armed bandit
	"""

	def __init__(self, k=K, mu=0, sigma1=1, sigma2=1):
		"""
		Returns a new testbed with k arms.

		Parameters
		----------
		k : Int >= 2
		  The number of bandit arms

		mu : Real scalar
		  The mean of the normal distribution used to select each Q*(a)

		sigma1 : Real scalar > 0
		  The standard deviation of the normal distribution used to select
		  each Q*(a)

		sigma2 : Real scalar > 0
		  The standard deviation of each normal distribution with mean Q*(a)
		"""
		self.k = k
		self._mu = mu
		self._sigma1 = sigma1
		self._sigma2 = sigma2
		self._rng = np.random.default_rng()
		self._means = [self._rng.normal(mu, sigma1) for i in range(k)]

	def interact(self, a):
		"""
		Performs action a in the testbed. It returns the reward the agent
		received for taking this action, which is the draw from a normal
		distribution with mean Q*(a) and standard deviation sigma2.

		Parameters
		----------
		a : Int in {0, ..., k-1}
		  The action the agent took.

		Returns
		-------
		Real scalar
		  The reward received for taking action a
		"""
		reward = self._rng.normal(self._means[a], self._sigma2)

		return reward

	def get_alike_bandit(self):
		"""
		Returns a KArmedTestbed initialized with the same parameters.

		Returns
		-------
		KArmedTestbed
		  A reinitialized testbed instance
		"""
		cls = self.__class__

		return cls(self.k, self._mu, self._sigma1, self._sigma2)


class KArmedTestbedNonStationary(KArmedTestbed):
	"""
	A class used to represent a variation of the k-armed testbed where it is
	non-stationary. Each of Q*(a)s will change to a new value from the
	G(mu, sigma1) distribution every m timesteps.
	
	Methods
	-------
	KArmedTestbedNonStationary(k=K, mu=0, sigma1=1, sigma2=1, m=300)
	  Returns a new KArmedTestbedNonStationary.

	interact(a)
	  Performs action a in the testbed. It returns the reward the agent
	  received for taking the action.

	get_alike_bandit()
	  Returns a KArmedTestbedNonStationary initialized with the same parameters.
	
	Parameters
	----------
	k : Int >= 2
	  The number of arms of the k-armed bandit
	"""
	
	def __init__(self, k=K, mu=0, sigma1=1, sigma2=1, m=300):
		"""
		Returns a new KArmedTestbedNonStationary.

		Parameters
		----------
		k : Int >= 2
		  The number of bandit arms

		mu : Real scalar
		  The mean of the normal distribution used to select each Q*(a)

		sigma1 : Real scalar > 0
		  The standard deviation of the normal distribution used to select
		  each Q*(a)

		sigma2 : Real scalar > 0
		  The standard deviation of each normal distribution with mean Q*(a)
		
		m : Int in (0, timesteps)
		  The number of timesteps to randomize the Q*(a)s
		"""
		super().__init__(k=K, mu=0, sigma1=1, sigma2=1)
		self._m = m
		self._count = 0
	
	def interact(self, a):
		"""
		Performs action a in the testbed. It returns the reward the agent
		received for taking this action, which is the draw from a normal
		distribution with mean Q*(a) and standard deviation sigma2.

		Parameters
		----------
		a : Int in {0, ..., k-1}
		  The action the agent took.

		Returns
		-------
		Real scalar
		  The reward received for taking action a
		"""
		self._count += 1
		mu = self._mu
		s1 = self._sigma1
		rng = self._rng
		k = self.k

		if (self._count % self._m) == 0:
			self._means = [rng.normal(mu, s1) for i in range(k)]
		
		reward = self._rng.normal(self._means[a], self._sigma2)

		return reward

	def get_alike_bandit(self):
		"""
		Returns a KArmedTestbedNonStationary initialized with the
		same parameters.

		Returns
		-------
		KArmedTestbedNonStationary
		  A reinitialized testbed instance
		"""
		cls = self.__class__
		
		return cls(self.k, self._mu, self._sigma1, self._sigma2, self._m)


class ContextualBandit:
	"""
	This class implements a ContextualBandit environment where each state is
	a different KArmedTestbed instance. A contextual bandit problem is one in
	which there is a concept of state, but it is simply an indication of which
	bandit the agent is facing. Actions have no effect on the future rewards
	received, only the next reward, and have no impact on what states the agent
	ends up in. ContextualBandits interact with agents of type
	ContextualBanditAgent. Each state has a uniform chance of being
	the "next" state. The action space A is represented as {0, ..., k-1}, and
	the state space is represented as {0, ..., n-1}, where k is the number of
	bandit arms and n is the number of states.
	
	Methods
	-------
	ContextualBandit(k=K, n=N, mu=0, sigma1=1, sigma2=1)
	  Returns an instance of a ContextualBandit with k arms and n states.

	interact(a)
	  Performs action a, returns the numerical reward for doing so,
	  and the next state

	get_alike_bandit()
	  Returns a ContextualBandit that is initialized with the same parameters
	  as the instance that used this method.

	get_starting_state()
	  Returns the starting state of the environment.

	Parameters
	----------
	k : Int >= 2
	  The number of arms, k, for each of the n k-armed bandits
	
	n : Int >= 2
	  The number of different states
	"""
	
	def __init__(self, k=K, n=N, mu=0, sigma1=1, sigma2=1):
		"""
		Returns a ContextualBandit with k arms and n states.

		Parameters
		----------
		k : Int >= 2
		  The number of arms, k, for each of the n k-armed bandits

		n : Int >= 2
		  The number of different states

		mu : Real scalar
		  The mean of the normal distribution used to determine
		  the Q*(state, action)s

		sigma1 : Real scalar > 0
		  The standard deviation of the normal distribution used to determine
		  the Q*(state, action)s

		sigma2: Real scalar > 0
		  The standard deviation of each reward distribution
		  G(Q*(state, action), sigma2)
		"""
		self.k = k
		self.n = n
		self._mu = mu
		self._sigma1 = sigma1
		self._sigma2 = sigma2
		self._bandits = [KArmedTestbed(k, mu, sigma1, sigma2) for x in range(k)]

		self._current_state = 0  # start at first state

	def interact(self, a):
		"""
		Performs action a. Returns the numerical reward for doing so,
		and the next state.

		Parameters
		----------
		a : Int in {0, ..., k-1}
		  The choice of arm to pull 

		Returns
		-------
		2-tuple:
		
		  [0] : Real scalar
		    The real-valued reward for taking action a
		  
		  [1] : Int in {0, ..., n-1}
		    The next state
		"""
		reward = self._bandits[self._current_state].interact(a)
		self._current_state = np.random.choice(self.n)

		return reward, self._current_state

	def get_alike_bandit(self):
		"""
		Returns a ContextualBandit that is initialized with the same 
		parameters as the instance that used this method.

		Returns
		-------
		ContextualBandit
		  A reinitialized Bandit instance with the same params
		"""
		cls = self.__class__
		return cls(self.k, self.n, self._mu, self._sigma1, self._sigma2)

	def get_starting_state(self):
		"""
		Returns the starting state of the environment.

		Returns
		-------
		Int in {0, ..., n-1}
		  The starting environment state
		"""
		return self._current_state  # state 0
