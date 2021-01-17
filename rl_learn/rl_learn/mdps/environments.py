import numpy as np
from rl_learn.mdps.guis import MDPConstructionGUI
import os
import json


class MDP:
    
    def __init__(self, dynamics, rewards, name="MDP"):
        """
		Returns a new Markov Decision Process Environment with the dynamics of the MDP specified in dynamics.

		Parameters
		----------
        dynamics : List[List[List[List[ 0 <= Float <= 1]]]]
            The joint reward-state distribution for each possible current state s_t and action taken a_t, i.e p(S_{t+1} = s', R_{t+1} = r | S_t = s, A_t = a).
            dynamics has the shape (S, A, S, R), where A is the number of actions, S is the number of states, and R is the number of rewards.
            So it can be thought of as a matrix with S rows and A columns corresponding to the current state s and action taken a.
            However, the (s, a) element of this matrix is not a single value, but instead an S x R matrix representing the joint probability of
            transitioning to state s and obtaining the reward r, hence this inner matrix has S rows and R columns.

        rewards : List[Float]
            The list of reward values of length R, in order, matching the last dimension of the dynamics.

        name : Str
            Can optionally give the MDP a name.
		"""
        self.dynamics = np.array(dynamics)
        self.rewards = np.array(rewards)
        self.name = name

        self.S = len(dynamics)
        self.A = len(dynamics[0])
        self.R = len(dynamics[0][0][0])
    
    def interact(self, s, a):
        """
        Returns a sample transition 2-tuple s_{t+1}, r_{t+1}, the resultant next state and reward, 
        given that the current state is S_t = s and action taken is A_t = a. It is a realized value
        from the joint reward-state probability distribution defined by this MDP's dynamics.

        A returned reward of -10,000 indicates the agent tried to interact in a terminal state.

        Parameters
		----------
        s : Int in {0, ..., S-1}
            The current state, S_t = s

		a : Int in {0, ..., A-1}
		    The action taken, A_t = a

		Returns
		-------
		2-tuple:
		
		  [0] : Real scalar
		    The real-valued reward for taking action a
		  
		  [1] : Int in {0, ..., S-1}
		    The next state
        """
        joint_probability_matrix = self.dynamics[s][a]
        R = self.R
        S = self.S

        # Create a flat copy of the array
        flat = joint_probability_matrix.flatten()

        # Then, sample an index from the 1D array with the
        # probability distribution from the original array
        sample_index = np.random.choice(a=flat.size, p=flat)

        # Take this index and adjust it so it matches the original array
        adjusted_index = np.unravel_index(sample_index, (S, R))
        s, r_index = adjusted_index

        return self.rewards[r_index], s
    
    def save(self, path, name="mdp.json"):
        """
        Stores the MDP in a json file at path. name must end in ".json".
        """
        path = os.path.join(path, name)

        json_data = {"dynamics" : self.dynamics.tolist(),
                      "rewards" : self.rewards.tolist(),
                      "name"    : self.name     }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        return path
    
    @staticmethod
    def load(path):
        """
        Returns an MDP constructed from the json file referenced at path.
        """
        json_data = {}
        with open(path) as infile:
            json_data = json.load(infile)
        
        dynamics = json_data["dynamics"]
        rewards = json_data["rewards"]
        name = json_data["name"]

        return MDP(dynamics, rewards, name)


    @staticmethod
    def MDP_from_transitions_and_reward_sas_triples(state_transitions, reward_sas_triples, rewards, name="MDP"):
        """
        Returns an MDP constructed with the state_transitions p(S_{t+1} = s | A_t = a, S_t = s), 
        the reward_state_action_next_state triples p(R_{t+1} = r | S_{t+1} = s', A_t = a, S_t = s), and
        the rewards vector of length R whose ith element is the value of reward i.

        Parameters
		----------
        state_transitions : List[List[List[ 0 <= Float <= 1]]]
            The state transitions p(S_{t+1} = s | A_t = a, S_t = s), which is a list of shape (S, A, S)

		reward_sas_triples : List[List[List[List[0 <= Float <= 1]]]]
		    The reward state_action_next_state triples p(R_{t+1} = r | S_{t+1} = s', A_t = a, S_t = s)
        
        rewards : List[Float]
            The rewards vector of length R whose ith element is the value of reward i.

		Returns
		-------
		MDP
            Returns a new MDP object.
        """
        state_transitions = np.array(state_transitions)
        reward_sas_triples = np.array(reward_sas_triples)
        rewards = np.array(rewards)

        S = len(reward_sas_triples)
        A = len(reward_sas_triples[0])
        R = len(rewards)

        dynamics = np.zeros((S, A, S, R), dtype=np.float)

        ## todo: optimize
        for s_t in range(S):
            for a_t in range(A):
                for s_tp1 in range(S):
                    for r_tp1 in range(R):
                        dynamics[s_t, a_t, s_tp1, r_tp1] = reward_sas_triples[s_t, a_t, s_tp1, r_tp1] * state_transitions[s_t, a_t, s_tp1]

        mdp = MDP(dynamics, rewards, name)

        return mdp
    
    @staticmethod
    def MDP_from_gui(name="MDP"):
        gui = MDPConstructionGUI()
        inputs = gui.inputs
        state_transitions = inputs["Transitions"]
        rewards = inputs["Rewards"]
        reward_sas_triples = inputs["RewardTriples"]
        mdp = MDP.MDP_from_transitions_and_reward_sas_triples(state_transitions, reward_sas_triples, rewards, name=name)
        return mdp

