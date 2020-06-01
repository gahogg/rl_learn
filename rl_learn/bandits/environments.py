"""
environments

A module that defines an interface that all bandit 
environments must provide, as well as provide some sample ones.
They interact with agents in the agents module.

Classes
-------
Bandit
  A class that defines the interface that any Bandit
  must provide.

KArmedTestbed
  A specific bandit problem that returns rewards according
  to normal distributions.

KArmedTestbedNonStationary
  A specific bandit problem whose reward distributions
  are not fixed.
"""

import numpy as np
from rl_learn.bandits import K ## Default number of bandit arms


class Bandit:
	"""
	A class used to represent a Bandit 
	environment - a particular type of 
	environment where every state is the same,
	and thus, there is no concept of state.
	They interact with agents of class BanditAgent.
	This class serves as an interface that all
	subclasses of Bandit must implement. The action
	space A is represented as {0, ..., k-1}.

	Methods
	-------
	Bandit(k=K)
	  Returns an instance of a Bandit with k arms.

	interact(a)
	  Performs action a, and returns the numerical 
	  reward for doing so.

	get_alike_bandit()
	  Returns a Bandit that is initialized with the same 
	  parameters as the instance that used this method.

	Parameters
	----------
	k : Int
	  The number of arms, k, of the k-armed bandit
	"""

	def __init__(self, k=K):
		"""
		Returns an instance of a Bandit with k arms.
		"""
		self.k = k

	def interact(self, a):
		"""
		Performs action a in the environment,
		and returns the numerical reward for doing so.
		Action a is an integer in {0, ..., k-1} representing
		a choice of arm to pull.

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
		Returns a Bandit that is initialized with the same 
		parameters as the instance that used this method.

		Returns
		-------
		KArmedBandit
		  A reinitialized Bandit instance with the same params
		"""
		return None


class KArmedTestbed(Bandit):
	"""
	A class used to represent the specific bandit
	problem of the k-armed testbed. Given k, mu,
	sigma1, and sigma2, a KArmedTestbed will be
	constructed where the reward for taking an action
	a is a draw from its unique normal distribution with
	mean Q*(a) and standard deviation sigma2, where Q*(a)
	is initialized randomly from a normal distribution with
	mean mu and standard deviation sigma1. 

	Methods
	-------
	KArmedTestbed(k=K, mu=0, sigma1=1, sigma2=1)
	  Returns a new testbed with k arms.

	interact(a)
	  Performs action a in the testbed. It returns
	  the reward the agent received for taking the action.

	get_alike_bandit()
	  Returns a KArmedTestbed initialized with the same parameters.
	
	Parameters
	----------
	k : Int
	  The number of arms, k, of the k-armed bandit
	"""

	def __init__(self, k=K, mu=0, sigma1=1, sigma2=1):
		"""
		Returns a new testbed with k arms.

		Parameters
		----------
		k : Int > 2
		  The number of bandit arms

		mu : Real scalar
		  The mean of the normal distribution used to
		  select each Q*(a)

		sigma1 : Real scalar
		  The standard deviation of the normal distribution
		  used to select each Q*(a)

		sigma2 : Real scalar
		  The standard deviation of each normal distribution
		  with mean Q*(a)
		"""
		self.k = k
		self._mu = mu
		self._sigma1 = sigma1
		self._sigma2 = sigma2
		self._rng = np.random.default_rng()
		self._means = [self._rng.normal(mu, sigma1) for i in range(k)]

	def interact(self, a):
		"""
		Performs action a in the testbed. It returns
		the reward the agent received for taking this action, 
		which is the draw from a normal distribution with mean
		Q*(a) and standard deviation sigma2. Q*(a) was selected
		randomly from a normal distribution of mean mu and standard
		deviation sigma1.

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
		It randomizes the Q*(a)s using the same normal distribution
		that initially selected them.

		Returns
		-------
		KArmedTestbed
		  A reinitialized testbed instance
		"""
		cls = self.__class__

		return cls(self.k, self._mu, self._sigma1, self._sigma2)


class KArmedTestbedNonStationary(KArmedTestbed):
	"""
	A class used to represent the specific bandit
	problem of the k-armed testbed that is non-stationary. 
	Given k, mu, sigma1, sigma2, and m, a KArmedTestbed will be
	constructed where the reward for taking an action
	a is a draw from its unique normal distribution with
	mean Q*(a) and standard deviation sigma2, where Q*(a)
	is initialized randomly from a normal distribution with
	mean mu and standard deviation sigma1. Each of Q*(a)s will
	change to a new value from the G(mu, sigma1) distribution
	every m timesteps.
	
	Methods
	-------
	KArmedTestbed(k=K, mu=0, sigma1=1, sigma2=1)
	  Returns a new testbed with k arms.

	interact(a)
	  Performs action a in the testbed. It returns
	  the reward the agent received for taking the action.

	get_alike_bandit()
	  Returns a KArmedTestbed initialized with the same parameters.
	
	Parameters
	----------
	k : Int
	  The number of arms, k, of the k-armed bandit
	"""
	
	def __init__(self, k=K, mu=0, sigma1=1, sigma2=1, m=300):
		"""
		KArmedTestbedNonStationary(k=K, mu=0, sigma1=1, sigma2=1)
		Returns a new testbed with k arms.

		Parameters
		----------
		k : Int
		  The number of bandit arms

		mu : Real scalar
		  The mean of the normal distribution used to
		  select each Q*(a)

		sigma1 : Real scalar
		  The standard deviation of the normal distribution
		  used to select each Q*(a)

		sigma2 : Real scalar
		  The standard deviation of each normal distribution
		  with mean Q*(a)
		
		m : Int, 0 < m < timesteps
		  The number of timesteps to reset the Q*(a)s 
		"""
		super().__init__(k=K, mu=0, sigma1=1, sigma2=1)
		self._m = m
		self._count = 0
	
	def interact(self, a):
		"""
		Performs action a in the testbed. It returns
		the reward the agent received for taking this action, 
		which is the draw from a normal distribution with mean
		Q*(a) and standard deviation sigma2. Q*(a) was selected
		randomly from a normal distribution of mean mu and standard
		deviation sigma1. After m timesteps, the Q*(a)s will become new
		values from the G(mu, sigma1) distribution.

		Parameters
		----------
		a : Int in {0, ..., k-1}
		  The action the agent took. Arms to pull, or actions to take,
		  are represented as indices.

		Returns
		-------
		Real scalar
		  The reward received for taking action a
		"""
		self._count += 1
		if (self._count % self._m == 0):
			self._means = [self._rng.normal(self._mu, self._sigma1) for i in range(self.k)]
		
		reward = self._rng.normal(self._means[a], self._sigma2)

		return reward

	def get_alike_bandit(self):
		"""
		Returns a KArmedTestbedNonStationary initialized with the same parameters.
		It randomizes the Q*(a)s using the same normal distribution
		that initially selected them.

		Returns
		-------
		KArmedTestbedNonStationary
		  A reinitialized testbed instance
		"""
		cls = self.__class__
		
		return cls(self.k, self._mu, self._sigma1, self._sigma2, self._m)
