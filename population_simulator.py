import numpy as np
from builtins import AttributeError
from math import floor

class InvasiveSpecies(object):
    """
    The invasive species domain.
    
    State space: OLD_discrete in [0,K]
    Action space: OLD_discrete in {0,1}
    
    Reward: negative absolute distance from the target value - R(s_t,a_t,s_{t+1}) = -|s_{t+1} - target| / 100.0
    
    Parameters:
     - horizon: Number of years (horizon)
     - N1: initial population
     - mean_lambda: mean 
     - sigma_lambda: random noise in the population growth
     - sigma_y: random noise in the observations
     - beta1,beta2: coefficients of effectiveness
     - K: maximum capacity
     - N_hat: peak of effectiveness
     - target: target value at which the population should be taken
    """
    def __init__(self, horizon = 100, N1 = 100, mean_lambda = 1.02, sigma2_lambda = 0.02, sigma2_y = 20, beta1 = 0.001, beta2 = -0.0000021, K = 500, N_hat = 300, target = 200):
        
        self.n_states = K+1
        self.n_actions = 2
        self.horizon = horizon
        self.N1 = N1
        self.mean_lambda = mean_lambda
        self.sigma2_lambda = sigma2_lambda
        self.sigma2_y = sigma2_y
        self.beta1 = beta1
        self.beta2 = beta2
        self.K = K
        self.N_hat = N_hat
        self.target = target
        self.state = N1
        
    def reset(self):
        """
        Resets the environment to the initial state
        """
        self.state = self.N1
        return self.get_observation()
        
    def get_observation(self):
        """
        Returns an observation of the current state
        """
        return floor(min(max(self.state + np.random.randn() * np.sqrt(self.sigma2_y),0),self.K))
        
    def step(self, action):
        """
        Applies an action and returns next state and reward
        """
        if action == 0:
            lambda_t = max(0,self.mean_lambda + np.random.randn() * np.sqrt(self.sigma2_lambda))
        elif action == 1:
            lambda_t = max(0,self.mean_lambda - self.state * self.beta1 - max(0, self.state - self.N_hat)**2 * self.beta2 + np.random.randn() * np.sqrt(self.sigma2_lambda))
        else:
            raise AttributeError("Illegal action")
        
        self.state = floor(min(lambda_t * self.state, self.K))
        
        reward = -abs(self.state - self.target) / 100.0
        
        return self.get_observation(), reward
    
    def get_reward_matrix(self):
        """
        Returns the reward matrix R(s_t,a_t,s_{t+1})
        """
        
        R = np.zeros((self.n_states,self.n_actions,self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for ns in range(self.n_states):
                    R[s,a,ns] = -abs(ns - self.target) / 100.0
        return R
    
    def get_init_state_probability(self):
        """
        Returns the init state probability vector Pr(s_1)
        """
        p = np.zeros(self.n_states)
        p[self.N1] = 1
        return p
        

invasive = InvasiveSpecies()
pop = []
for i in range(500):
    pop.append(invasive.step(0)[0])

print(pop)
