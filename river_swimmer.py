import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import itertools
import pprint
from robust_mdp import *
import pandas as pd
from tqdm import tqdm
np.set_printoptions(precision=2, suppress=True)

class RiverSwimmer(object):
    def __init__(self, n, gamma, feature_type):
        self.n = n
        self.gamma = gamma # discount factor
        self.m = 2 # number of actions
        self.states = np.arange(n)
        # transition probability P_(m*(n*n))
        self.transition = self.set_transition()
        # initial state distribution p_0(n)
        self.p_0 = self.set_initial_dist()
        # rewards = get_random_reward(m, n) # doesn't matter at this point
        # self.rewards = np.array([[0.1, 0.92],[1, 0.02]])
        self.rewards = self.set_rewards()
        # Set the features matrix phi(m*(n*k))
        self.k = 2
        self.phi = self.set_feature_matrix(feature_type)
        # self.num_episodes = num_episodes
        # self.episodes_len = episode_len
        self.mdp = MDP(self.n, self.m, self.k, self.transition, self.phi,
                       self.p_0, self.gamma)
        self.opt_u = None
        self.opt_policy = None

    def create_samples(self, num_episodes, episode_len):
        # cur_policy = np.argmax(policy, axis=0)
        policy = self.opt_policy
        episodes = []
        # experiment saves the flattened version of episodes
        experiment = []
        rhos = []
        for _ in range(num_episodes):
            gamma_t = 1
            s0 = np.random.choice(self.n, p=self.p_0)      # current state
            s1 = s0      # next state
            episode = []
            for step in range(episode_len):
                a0 = np.random.choice(self.m, p=policy[:, s0])
                # sample is a state-action-reward tuple
                sample = [s0, a0] # , self.rewards[a0, s0]
                # import ipdb;ipdb.set_trace()
                s1 = np.random.choice(self.n, p=self.transition[a0, s0]) # one step
                gamma_t *= self.gamma
                episode.append(sample)
                s0 = s1                         # proceed to the next state
            episodes.append(episode)
            experiment.append([smpls for step in episode for smpls in sample[0:2]])
        return episodes

    def estimate_uE(self, num_episodes, episode_len):
        self.set_policy()
        # import ipdb; ipdb.set_trace()
        demonstrations = self.create_samples(num_episodes, episode_len)
        # passing the samples to the mdp object to estimate u^E
        u_E = self.mdp.get_u_E(demonstrations)
        return u_E

    def set_initial_dist(self):
        # p_0 = np.zeros(self.n)
        # p_0[0] = .5
        # p_0[-1] = .5
        eps = 10 ** -2
        p_0 = np.ones(self.n) * eps
        p_0[0] = 1 - (self.n - 1) * eps
        # import ipdb;ipdb.set_trace()
        return p_0

    def set_rewards(self):
        rewards = np.zeros((self.m, self.n))
        rewards[:, 0] += 5/1000
        rewards[:, self.n - 1] += 1
        # rewards[:, 0:self.n - 1] = -1 * np.ones([self.m, self.n - 1])
        # rewards[:, self.n - 1] = -5/1000 * np.ones(self.m)
        # import ipdb;ipdb.set_trace()
        # negative random rewards
        # w = np.random.rand(2)
        # while np.sum(np.abs(w)) > 1 or abs(w[0] - w[1]) < .1:
        #     w = np.random.rand(2)
        # # w = np.array([+1, -0.005])
        # r_s = - np.ones(self.n) * w[0]
        # r_s[self.n - 1] = +1 * w[1]
        # rewards = np.array([r_s for i in range(self.m)])
        return rewards

    def set_feature_matrix(self, feature_type):
        if (feature_type == 'identity_sa'):
            k = self.m * self.n
            self.k = k
            phi = np.eye(k).reshape([self.m, self.n, k])
        elif (feature_type == 'negative'):
            k = 2
            self.k = k
            n = self.n
            m = self.m
            phi_s = np.tile([1, 0], n).reshape(n, k)
            phi_s[n - 1] = np.array([0, 1])
            phi = np.array([phi_s for a in range(m)])
        # self.phi = phi
        return phi

        return phi
        # phi_s = np.eye(2)
        # return np.array([phi_s for i in range(self.m)])

    def set_transition(self):
        transitions = np.zeros((self.n, self.m, self.n))
        for s in range(self.n):
            # action left
            transitions[s, 0, max(s-1,0)] = 1
            # action right
            transitions[s, 1, s] = 0.4 if s==0 else 0.6
            transitions[s, 1, min(s+1,self.n-1)] = 0.6 if s==0 else (0.6 if s==self.n-1 else 0.35)
            transitions[s, 1, max(s-1,0)] = 0.4 if s==0 else (0.4 if s==self.n-1 else 0.05)
            # transitions[s, 1, s] = 0.2
            # transitions[s, 1, min(s+1,num_states-1)] = 0.8 if s==0 else (0.2 if s==num_states-1 else 0.5)
            # transitions[s, 1, max(s-1,0)] = 0.2 if s==0 else (0.8 if s==num_states-1 else .3) 
            # transitions[s, 1, s] = 0.3
            # transitions[s, 1, 0] = 0.2
            # transitions[s, 1, min(s+1,num_states-1)] = 0.8 if s==0 else (0.8 if s==num_states-1 else 0.5) 
        # adjusting the transition matrix according to my code
        transition = np.array([transitions[:,a,:] for a in range(self.m)])

        # bunch of assertions to check the correcness of transitions
        assert abs(np.sum(transition[0,:,:].sum(axis=1)) - self.n) < 10**-3,\
                                "Transition matrix doesn't sum to one"
        assert abs(np.sum(transition[0,:,:].sum(axis=1)) - self.n) < 10**-3,\
                                "Transition matrix doesn't sum to one"
        w, v = LA.eig(np.eye(self.n) - self.gamma * transition[0,:,:])
        assert np.all(w > -.001), "non-positive eigen value for P"
        w, v = LA.eig(np.eye(self.n) - self.gamma * transition[1,:,:])
        assert np.all(w > -.001), "non-positive eigen value for P"
        # pprint.pprint(transition)
        # import ipdb; ipdb.set_trace()
        return transition


    def set_policy(self):
        self.opt_u, opt_val = self.mdp.dual_lp(self.rewards)
        self.opt_policy = self.mdp.u_to_policy(self.opt_u)
        # self.opt_u = self.mdp.policy_to_u(self.opt_policy)

    def get_experiment_returns(self, u_E):
        rewards = self.rewards

        # Max ent stuff
        # max_ent_reward = mdp.max_ent_irl(0.9, 100)
        # print(max_ent_reward)
        # max_ent_reward = np.array([max_ent_reward for a in range(num_actions)])
        # max_ent_policy, _ = mdp.dual_lp(max_ent_reward)
        # print(max_ent_policy)

        Syeds_u_mat, Syed_opt_val_mat = self.mdp.Syed_mat()
        consistent_u_u_mat, const_u_u_opt_val_mat = self.mdp.Cons_u_optimal_u_mat()
        eps_list = [0, 1.0, 10]
        consistent_r_u_l1_epsion0, _ = self.mdp.Cons_r_optimal_u_l1_mat(eps_list[0])
        consistent_r_u_l2_epsion0, _ = self.mdp.Cons_r_optimal_u_l2_mat(eps_list[0])
        consistent_r_u_l1_epsion1, _ = self.mdp.Cons_r_optimal_u_l1_mat(eps_list[1])
        consistent_r_u_l2_epsion1, _ = self.mdp.Cons_r_optimal_u_l2_mat(eps_list[1])
        consistent_r_u_l1_epsion2, _ = self.mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
        consistent_r_u_l2_epsion2, _ = self.mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
        # consistent_r_u_l1_epsion3, _ = self.mdp.Cons_r_optimal_u_l1_mat(eps_list[3])
        # consistent_r_u_l2_epsion3, _ = self.mdp.Cons_r_optimal_u_l2_mat(eps_list[3])
        opt_return = self.mdp.get_return(rewards, self.opt_u)
        cu = self.mdp.get_return(rewards, consistent_u_u_mat)
        syed = self.mdp.get_return(rewards, Syeds_u_mat)
        l1_a = self.mdp.get_return(rewards, consistent_r_u_l1_epsion0)
        l1_b = self.mdp.get_return(rewards, consistent_r_u_l1_epsion1)
        l1_c = self.mdp.get_return(rewards, consistent_r_u_l1_epsion2)
        l2_a = self.mdp.get_return(rewards, consistent_r_u_l2_epsion0)
        l2_b = self.mdp.get_return(rewards, consistent_r_u_l2_epsion1)
        l2_c = self.mdp.get_return(rewards, consistent_r_u_l2_epsion2)
        # l1_d = self.mdp.get_return(rewards, consistent_r_u_l1_epsion3)
        # l2_d = self.mdp.get_return(rewards, consistent_r_u_l2_epsion3)
        returns_list = [opt_return, cu, l2_a, l2_b, l2_c, syed, l1_a, l1_b, l1_c]
        return returns_list


def print_resutls(returns_list):
    returns_list = np.around(returns_list, decimals=3)
    print("| Optimal U\t\t|", returns_list[0], "\t|")

    eps_list = [0, 1.0, 10]
    print("| consistent U\t\t|", returns_list[1], "\t|")
    print("| l2, eps=", eps_list[0], "\t\t|", returns_list[2], "\t|")
    print("| l2, eps=", eps_list[1],"\t\t|",  returns_list[3], "\t|")
    print("| l2, eps=", eps_list[2],"\t\t|",  returns_list[4], "\t|")

    print("| Syed\t\t\t|", returns_list[5], "\t|")
    print("| l1, eps=", eps_list[0], "\t\t|", returns_list[6], "\t|")
    print("| l1, eps=", eps_list[1], "\t\t|", returns_list[7], "\t|")
    print("| l1, eps=", eps_list[2],"\t\t|",  returns_list[8], "\t|")

    # print("| l1, eps=", eps_list[3],"\t\t|",  l1_d)
    # print("| l2, eps=", eps_list[3],"\t\t|",  l2_d)


def consistent_R_modified_uE():
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    env = RiverSwimmer(5, .9999, feature_type="negative")
    env.estimate_uE(num_episodes=10, episode_len=15)
    # gridworld.mdp.u_E = gridworld.opt_u
    true_rewards = env.rewards
    # dual_LP, _ = env.mdp.dual_lp(true_rewards)

    # find u for different methods
    u_Syed, _ = env.mdp.Syed_mat()
    cons_u_mat, _ = env.mdp.Cons_u_optimal_u_mat()
    u_const_gen = env.mdp.constraint_generation()
    u_const_gen = env.mdp.analytical_chebyshev()
    u_cons_r_l1_modified_uE = env.mdp.Cons_r_l1_modified_uE(0.1)
    # eps_list = [0.1, 1, 10.0, 100]
    eps_list = []
    cons_r_l1 = []
    cons_r_l2 = []
    returns_list = []
    for eps in eps_list:
        cons_r_l1.append(env.mdp.Cons_r_optimal_u_l1_mat(eps))
        cons_r_l2.append(env.mdp.Cons_r_optimal_u_l2_mat(eps))

    # add the returns of each u to the list
    returns_list.append(env.mdp.get_return(env.rewards, u_const_gen))
    returns_list.append(env.mdp.get_return(env.rewards, u_cons_r_l1_modified_uE))
    returns_list.append(env.mdp.get_return(env.rewards, u_Syed))
    # consistent R with l1 norm
    for i in range(len(eps_list)):
        returns_list.append(env.mdp.get_return(true_rewards, cons_r_l1[i]))
    returns_list.append(env.mdp.get_return(true_rewards, cons_u_mat))
    # consistent R with l2 norm
    for i in range(len(eps_list)):
        returns_list.append(env.mdp.get_return(true_rewards, cons_r_l2[i]))
    # optimal return
    returns_list.append(env.mdp.get_return(env.rewards, env.opt_u))
    return returns_list


def get_returns(env=None, num_episodes=10, episode_len=5):
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    if env is None:
        env = GridWorld(num_rows=8, m=4, k=3, gamma=.9999, reward_type="k-color",
                        feature_type="k-color", p=[.1, .6])

    env.estimate_uE(num_episodes=num_episodes, episode_len=episode_len)
    # gridworld.mdp.u_E = gridworld.opt_u
    true_rewards = env.rewards
    # dual_LP, _ = env.mdp.dual_lp(true_rewards)

    # find u for different methods
    u_Syed, _ = env.mdp.Syed_mat()
    cons_u_mat, _ = env.mdp.Cons_u_optimal_u_mat()
    u_const_gen = env.mdp.constraint_generation()
    u_const_cheb = env.mdp.analytical_chebyshev()
    u_cons_r_l1_modified_uE = env.mdp.Cons_r_l1_modified_uE(0.1)
    eps_list = [0.5]
    # eps_list = []
    cons_r_l1 = []
    cons_r_l2 = []
    returns_list = []
    for eps in eps_list:
        cons_r_l1.append(env.mdp.Cons_r_optimal_u_l1_mat(eps))
        cons_r_l2.append(env.mdp.Cons_r_optimal_u_l2_mat(eps))

    # add the returns of each u to the list
    returns_list.append(env.mdp.get_return(env.rewards, u_const_gen))
    returns_list.append(env.mdp.get_return(env.rewards, u_const_cheb))
    returns_list.append(env.mdp.get_return(env.rewards, u_cons_r_l1_modified_uE))
    returns_list.append(env.mdp.get_return(env.rewards, u_Syed))

    # consistent R with l1 norm
    for i in range(len(eps_list)):
        returns_list.append(env.mdp.get_return(true_rewards, cons_r_l1[i]))
    returns_list.append(env.mdp.get_return(true_rewards, cons_u_mat))

    # consistent R with l2 norm
    for i in range(len(eps_list)):
        returns_list.append(env.mdp.get_return(true_rewards, cons_r_l2[i]))

    # optimal return
    returns_list.append(env.mdp.get_return(env.rewards, env.opt_u))
    return returns_list


def same_reward_various_episode_length():
    np.random.seed(10)
    returns_list = []
    markers = ['.', 'o', 'v', '^', '<', '>', ',', '1', '2', '3', '4', '8', 's',
               'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    methods = ["Const Gen", "Chebyshev", "Modified uE", "Syed", "Const L1 eps",
               "Const L2", "Const L2 eps", "Optimal"]
    num_exp = 30

    env = RiverSwimmer(5, .9999, feature_type="negative")
    # increase the episodes length
    for i in tqdm(range(10, 40, 4)):
        exp_result = []
        # take the average of num_exp experiments for each episode length
        for j in range(num_exp):
            exp_result.append(get_returns(env, num_episodes=1, episode_len=i))
        exp_result = np.array(exp_result)
        returns_list.append(np.mean(exp_result, axis=0))

    returns_array = np.array(returns_list).T

    for j in range(returns_array.shape[0]):
        plt.plot(np.arange(5, 5 + returns_array.shape[1]), returns_array[j, :],
                 label=methods[j], marker=markers[j])
    plt.legend()
    plt.savefig("../../files/riverswimmer_various_episode_len")


def main():
    num_experiments = 1
    R_returns = []

    for i in tqdm(range(num_experiments)):
        # experiment_returns.append(consistent_R_modified_uE())
        R_returns.append(same_reward_various_episode_length())

    R_returns = np.array(R_returns)
    exp_avg = np.mean(R_returns, axis=0)
    exp_std = np.std(R_returns, axis=0)
    print(exp_avg)
    # print(exp_std)


if __name__ == "__main__":
    main()
