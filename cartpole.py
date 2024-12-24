import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pickle
import datetime
import itertools
import pprint
from robust_mdp import *
import sys
import pandas as pd
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import preprocessing
import gym

# sys.path.append("../../")
# from lib import cart_pole

class CartPole(object):
# let's assume we have a bunch of samples from cartpole experiment
# states are in the form of [x, v, theta, d_theta] and 
# samples are in the form of [s_0, a_0, s_1]
    def __init__(self, gamma, num_states):

        self.gamma = gamma
        # number of bins
        self.n = num_states
        self.m = 2
        # self.rewards = self.set_rewards(reward_type)
        self.transition = self.read_transition_cvs()
        self.p_0 = self.set_initial_dist()
        self.k = None
        self.rewards = self.set_rewards()
        self.phi = self.set_feature_matrix()
        self.policy = None
        self.mdp = MDP(self.n, self.m, self.k, self.transition, self.phi,
                       self.p_0, self.gamma)
        self.opt_u = None
        self.opt_policy = None


    def create_samples(self, num_episodes, episode_len):
        policy = self.opt_policy
        m = self.m
        n = self.n
        transition = self.transition
        p0 = self.p_0
        gamma = self.gamma
        experiment = []
        for _ in range(num_episodes):
            gamma_t = 1
            s0 = np.random.choice(n, p=p0)      # current state
            s1 = s0      # next state
            episode = []
            for step in range(episode_len):
                a0 = np.random.choice(m, p=policy[:, s0])
                sample = [s0, a0]
                s1 = np.random.choice(n, p=transition[a0, s0])
                gamma_t *= gamma
                episode.append(sample)
                s0 = s1                         # proceed to the next state
            experiment.append(episode)
        return experiment


    def estimate_uE(self, num_episodes, episode_len):
        self.set_policy()
        # import ipdb; ipdb.set_trace()
        demonstrations = self.create_samples(num_episodes, episode_len)
        # passing the samples to the mdp object
        self.mdp.set_u_E(demonstrations)
        self.mdp.u_E = self.opt_u
        return self.mdp.u_E

    def read_transition_cvs(self):
        df = pd.read_csv('../../files/cartpole_mdp.csv', header=0, names=['s0', 'a', 's1',
                                                                'p', 'r'])
        # n = len(pd.unique(df['s0']))
        n = df['s0'].max() + 1
        # df = df.drop_duplicates()
        df['true_p'] = df.groupby(['s0', 'a', 's1'])['p'].transform('sum')
        m = 2
        # print(n, df['s0'].max())
        transitions = np.zeros([m, n, n])
        for elem in df.values:
            s0, a, s1, _, _, p = elem
            transitions[int(a), int(s0), int(s1)] = p
            # import ipdb; ipdb.set_trace()
        assert abs(np.sum(transitions[0,:,:].sum(axis=1)) - n) < 10**-3,\
                                "Transition matrix doesn't sum to one"
        assert abs(np.sum(transitions[0,:,:].sum(axis=1)) - n) < 10**-3,\
                                "Transition matrix doesn't sum to one"
        # import ipdb; ipdb.set_trace()
        # w, v = LA.eig(np.eye(n) - gamma * transitions[0,:,:])
        # assert np.all(w > 0), "non-positive eigen value for P"
        # w, v = LA.eig(np.eye(n) - gamma * transitions[1,:,:])
        # assert np.all(w > 0), "non-positive eigen value for P"

        # transition = np.array([transitions[:,a,:] for a in range(m)])
        # create p_0
        return transitions

    def read_uE_cvs():
        df = pd.read_csv('../../files/expert.csv', header=0, names=['step','s0',
                                                                    'a', 's1'])
        exp_time_step = df.index[df['step'] == 0].tolist()
        exp_time_step.append(df.shape[0] - 1)
        experiments = []
        for i in range(len(exp_time_step) - 1):
            experiments.append(df.loc[exp_time_step[i]:exp_time_step[i + 1] - 1,
                                    ['s0', 'a', 's1']]. to_numpy())
        # import ipdb; ipdb.set_trace()
        return experiments

    def read_policy_cvs(self):
        n = self.n
        m = self.m
        k = self.k
        df = pd.read_csv('../../files/policy_nn.csv', header=0,
                         names=['f0', 'f1', 'f2', 'f3', 's', 'a', 'v'])
        policy = np.zeros([m, n])
        filled_ind = df[['s', 'a']].to_numpy()
        for ind in filled_ind:
            i, j = ind
            policy[j, i] = 1
        # import ipdb; ipdb.set_trace()
        # phi = np.zeros([k, m, n])
        # for v in df.values:
        #     f0, f1, f2, f3, s, a, _ = v
        #     phi[:, int(a), int(s)] = np.array([f0, f1, f2, f3])
        # for a in range(m):
        #     phi[:, a, :] = df[['f0', 'f1', 'f2', 'f3']].T.to_numpy()
        # import ipdb;ipdb.set_trace()
        # return phi

        return policy


    def standup(self, position):
        pos = position.to_numpy()[0][0:4]
        if abs(pos[1]) < 0.4: #   and abs(pos[2] < .2) and abs(pos[3] < 0.2)
            return True
        else:
            return False

    def set_rewards(self):
        n = self.n
        m = self.m
        k = self.k
        r = np.zeros(n)
        df = pd.read_csv('../../files/policy_nn.csv', header=0,
                         names=['f0', 'f1', 'f2', 'f3', 's', 'a', 'v'])
        for s in range(n):
            r[s] = 0 if self.standup(df[(df['s'] == s)]) else -1
        # import ipdb; ipdb.set_trace()
        return np.array([r for i in range(m)])


    def set_feature_matrix(self, feature_type=None):
        k = self.m * self.n
        self.k = k
        phi = np.eye(k).reshape([self.m, self.n, k])
        return phi

    def set_initial_dist(self):
        p_0 = np.ones(self.n) / self.n
        return p_0

    def set_policy(self):
        # read the policy from the rscript outputs
        # policy_opt = self.read_policy_cvs()
        # # import ipdb; ipdb.set_trace()
        # self.opt_policy = policy_opt
        # u_opt = self.mdp.policy_to_u(policy_opt)
        # self.opt_u = u_opt

        self.opt_u, opt_val = self.mdp.dual_lp(self.rewards)
        self.opt_policy = self.mdp.u_to_policy(self.opt_u)

    def get_irl_returns(self):
        # Max ent stuff
        # max_ent_reward = mdp.max_ent_irl(0.9, 100)
        # print(max_ent_reward)
        # max_ent_reward = np.array([max_ent_reward for a in range(num_actions)])
        # max_ent_policy, _ = mdp.dual_lp(max_ent_reward)
        # print(max_ent_policy)
        mdp = self.mdp

        Syeds_u_mat, Syed_opt_val_mat = mdp.Syed_mat()
        consistent_u_u_mat, const_u_u_opt_val_mat = mdp.Cons_u_optimal_u_mat()
        eps_list = [0, 1.0, 10]
        consistent_r_u_l1_epsion0, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[0])
        consistent_r_u_l2_epsion0, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[0])
        consistent_r_u_l1_epsion1, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[1])
        consistent_r_u_l2_epsion1, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[1])
        consistent_r_u_l1_epsion2, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
        consistent_r_u_l2_epsion2, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
        distance_u = []
        distance_u.append(LA.norm(Syeds_u_mat - mdp.u_E, ord=2))
        distance_u.append(LA.norm(consistent_u_u_mat - mdp.u_E, ord=2))
        distance_u.append(LA.norm(consistent_r_u_l1_epsion0 - mdp.u_E, ord=2))
        distance_u.append(LA.norm(consistent_r_u_l1_epsion1 - mdp.u_E, ord=2))
        distance_u.append(LA.norm(consistent_r_u_l1_epsion2 - mdp.u_E, ord=2))
        distance_u.append(LA.norm(consistent_r_u_l2_epsion0 - mdp.u_E, ord=2))
        distance_u.append(LA.norm(consistent_r_u_l2_epsion1 - mdp.u_E, ord=2))
        distance_u.append(LA.norm(consistent_r_u_l2_epsion2 - mdp.u_E, ord=2))
        # consistent_r_u_l1_epsion3, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[3])
        # consistent_r_u_l2_epsion3, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[3])
        # opt_return = mdp.get_return(rewards, opt_u)
        # cu = mdp.get_return(rewards, consistent_u_u_mat)
        # syed = mdp.get_return(rewards, Syeds_u_mat)
        # l1_a = mdp.get_return(rewards, consistent_r_u_l1_epsion0)
        # l1_b = mdp.get_return(rewards, consistent_r_u_l1_epsion1)
        # l1_c = mdp.get_return(rewards, consistent_r_u_l1_epsion2)
        # l2_a = mdp.get_return(rewards, consistent_r_u_l2_epsion0)
        # l2_b = mdp.get_return(rewards, consistent_r_u_l2_epsion1)
        # l2_c = mdp.get_return(rewards, consistent_r_u_l2_epsion2)
        # l1_d = self.mdp.get_return(rewards, consistent_r_u_l1_epsion3)
        # l2_d = self.mdp.get_return(rewards, consistent_r_u_l2_epsion3)
        # returns_list = [opt_return, cu, l2_a, l2_b, l2_c, syed, l1_a, l1_b, l1_c]
        # return returns_list
        return distance_u

def get_action(observation):
    position, velocity, angle, angle_velocity = observation
    action = int(1. * angle + angle_velocity > 0.)
    return action


def sample_to_transition_matrix_kmeans(s0, a, s1, init_states, k=100):
    # a simple example to test the function
    # num_sample = 1000
    # x0 = 2 * np.random.random(num_sample) - 1
    # v0 = np.random.random(num_sample)
    # theta0 = np.random.random(num_sample) - 1
    # d_theta0 = np.random.random(num_sample) - 1 
    # x1 = 2 * np.random.random(num_sample) - 1
    # v1 = np.random.random(num_sample)
    # theta1 = np.random.random(num_sample) - 1
    # d_theta1 = np.random.random(num_sample) - 1 
    # a = np.random.randint(0, 2, num_sample)
    # s0 = np.array([x0, v0, theta0, d_theta0])
    # s1 = np.array([x1, v1, theta1, d_theta1])
    # transition = sample_to_transition_matrix(s0, a, s1)
    ####### transition matrix calculation based on an experiment ########
    s = np.vstack([s0, s1])
    # import ipdb; ipdb.set_trace()
    df = pd.DataFrame(s, columns=["x", "v", "theta", "d_theta"])
    kmodel = KMeans(n_clusters=k)
    kmeans = kmodel.fit(df)
    D = np.vstack([kmeans.predict(s0), a, kmeans.predict(s1)]).T
    df = pd.DataFrame(D,  columns=['s0', 'a', 's1'])
    fix_zeros = pd.DataFrame({'s0': np.repeat(np.arange(k), 2),
                              'a': np.tile(np.arange(2), k),
                              's1': np.repeat(np.arange(k), 2)})
                              # 'sum_sas':[1] * k * 2,
                              # 'sum_sa':[1] * k * 2})
    df_fixed = pd.merge(df, fix_zeros, how='outer', on=['s0', 'a', 's1'])
    # df = df_fixed
    # df['sum_sas'] = df.groupby(['s0', 'a'])['a'].transform('count')
    # sas_grouped = df.groupby(by=["s0", "a", "s1"], as_index=False)
    # sa_sum = sas_grouped.join(sas_grouped['s1'].count(), on='s1', rsuffix='_c')
    # sas_sum = df.groupby(by=["s0", "a"])['s1'].count()
    # result = pd.merge(sa_sum, sas_sum, how="left", on=['s0', 'a'])
    # result["p"] = np.where(result['s1_y'] < 1, result['s1_y'], result['s1_x'] /
    #                        result['s1_y'])
    df_fixed['sum_sas'] = df_fixed.groupby(['s0', 'a', 's1'])['s1'].transform('count')
    df_fixed['sum_sa'] = df_fixed.groupby(['s0', 'a'])['a'].transform('count')
    df_fixed = df_fixed.drop_duplicates().sort_values(['s0', 'a', 's1'])
    # fixed = pd.merge(df, fix_zeros, how='left', on=['s0', 'a', 's1', 'sum_sas', 'sum_sa'])
    df_fixed["p"] = np.where(df_fixed['sum_sas'] < 1, df_fixed['sum_sas'], df_fixed['sum_sas'] /
                           df_fixed['sum_sa'])
    T = df_fixed.to_numpy()
    # print(kmeans.labels_)
    n = k
    m = 2
    transitions = np.zeros([n, m, n])
    for elem in (T):
        s0, a, s1, _, _, p = elem
        transitions[int(s0), int(a), int(s1)] = p
    # import ipdb; ipdb.set_trace()
    assert(abs(np.sum(transitions[:,0,:].sum(axis=1)) - n) < 10**-3)
    assert(abs(np.sum(transitions[:,1,:].sum(axis=1)) - n) < 10**-3)   
    transition = np.array([transitions[:,a,:] for a in range(m)])
    # create p_0
    p0_df = pd.DataFrame(kmeans.predict(np.array(init_states)), columns=["s0"])
    # p0 = p0_df.groupby(by=["s0"]).count()
    return transition, df

# def discretize(experiment):
#     exp = []
#     episodes = []
#     for exper in experiment:
#         for step in exper:
            

def sample_to_transition_matrix_knn(s0, a, s1, init_states, k=100):
    ####### transition matrix calculation based on an experiment ########
    s = np.vstack([s0, s1])
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(s)
    # df = pd.DataFrame(x_scaled)
    df = pd.DataFrame(s, columns=["x", "v", "theta", "d_theta"])
    y = np.arange(len(s))
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn = knn_model.fit(df, y)
    D = np.vstack([knn.predict(s0), a, knn.predict(s1)]).T
    df = pd.DataFrame(D,  columns=['s0', 'a', 's1'])
    fix_zeros = pd.DataFrame({'s0': np.repeat(np.arange(k), 2),
                              'a': np.tile(np.arange(2), k),
                              's1': np.repeat(np.arange(k), 2)})
    df_fixed = pd.merge(df, fix_zeros, how='outer', on=['s0', 'a', 's1'])
    df_fixed['sum_sas'] = df_fixed.groupby(['s0', 'a', 's1'])['s1'].transform('count')
    df_fixed['sum_sa'] = df_fixed.groupby(['s0', 'a'])['a'].transform('count')
    df_fixed = df_fixed.drop_duplicates().sort_values(['s0', 'a', 's1'])
    # fixed = pd.merge(df, fix_zeros, how='left', on=['s0', 'a', 's1', 'sum_sas', 'sum_sa'])
    df_fixed["p"] = np.where(df_fixed['sum_sas'] < 1, df_fixed['sum_sas'], df_fixed['sum_sas'] /
                           df_fixed['sum_sa'])
    T = df_fixed.to_numpy()
    # print(kmeans.labels_)
    n = 2 * k
    m = 2
    transitions = np.zeros([n, m, n])
    for elem in (T):
        s0, a, s1, _, _, p = elem
        transitions[int(s0), int(a), int(s1)] = p
    # import ipdb; ipdb.set_trace()
    import ipdb; ipdb.set_trace()
    assert(abs(np.sum(transitions[:,0,:].sum(axis=1)) - n) < 10**-3)
    assert(abs(np.sum(transitions[:,1,:].sum(axis=1)) - n) < 10**-3)   
    transition = np.array([transitions[:,a,:] for a in range(m)])
    # create p_0
    p0_df = pd.DataFrame(knn.predict(np.array(init_states)), columns=["s0"])
    # p0 = p0_df.groupby(by=["s0"]).count()
    return transition, df
            

def print_resutls(mdp, u_E, norm):
    num_actions = mdp.m
    mdp.set_u_expert(u_E)
    # print(mdp.u_E)
    # mdp.u_E = u_E
    # uncomment this line to set u^E to the optimal u
    # mdp.u_E = opt_occ_freq
    # max_ent_reward = mdp.max_ent_irl(0.9, 100)
    # print(max_ent_reward)
    # max_ent_reward = np.array([max_ent_reward for a in range(num_actions)])
    # max_end_policy, _ = mdp.dual_lp(max_ent_reward)
    # print(max_end_policy)
    # Abbeels_lp_u_mat, Abbeel_reward_mat, Abbeel_lp_opt_val_mat = mdp.\
    #                                         Abbeel_dual_lp_mat()
    Syeds_u_mat, Syed_opt_val_mat = mdp.Syed_mat()
    consistent_u_u_mat, const_u_u_opt_val_mat = mdp.Cons_u_optimal_u_mat()
    # consisten R using different epsilons
    eps_list = [0, 0.1, 1.0, 10]
    consistent_r_u_l1_epsion0, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[0])
    consistent_r_u_l2_epsion0, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[0])
    consistent_r_u_l1_epsion1, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[1])
    consistent_r_u_l2_epsion1, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[1])
    consistent_r_u_l1_epsion2, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
    consistent_r_u_l2_epsion2, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
    consistent_r_u_l1_epsion3, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[3])
    consistent_r_u_l2_epsion3, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[3])

    syed = round(LA.norm(mdp.u_E - Syeds_u_mat), ord=norm)
    cu =   round(LA.norm(mdp.u_E - consistent_u_u_mat), ord=norm)
    l1_a = round(LA.norm(mdp.u_E - consistent_r_u_l1_epsion0), ord=norm)
    l1_b = round(LA.norm(mdp.u_E - consistent_r_u_l1_epsion1), ord=norm)
    l1_c = round(LA.norm(mdp.u_E - consistent_r_u_l1_epsion2), ord=norm)
    l1_d = round(LA.norm(mdp.u_E - consistent_r_u_l1_epsion3), ord=norm)
    l2_a = round(LA.norm(mdp.u_E - consistent_r_u_l2_epsion0), ord=norm)
    l2_b = round(LA.norm(mdp.u_E - consistent_r_u_l2_epsion1), ord=norm)
    l2_c = round(LA.norm(mdp.u_E - consistent_r_u_l2_epsion2), ord=norm)
    l2_d = round(LA.norm(mdp.u_E - consistent_r_u_l2_epsion3), ord=norm)
    error = cu, syed, l1_a, l1_b, l1_c, l1_d,  l2_a, l2_b, l2_c, l2_d
    return error
    
def print_norm_diff(method, u_E, u):
    print(method, '\t\t{:.3f}'.format(LA.norm(u_E - u)))
def IRL():
    cartpole = CartPole(gamma=0.9999, num_states=1000)
    u_E = cartpole.estimate_uE(num_episodes = 1, episode_len = 1000)
    num_experiments = 10
    # import ipdb;ipdb.set_trace()
    # policy = np.vstack([np.ones(num_states), np.zeros(num_states)])
    returns_list = []
    for exper in range(num_experiments):
        print(exper, 'th experiment')
        experiment = create_samples(num_actions, num_states, transition, p_0,
                                    gamma, policy_opt, num_episodes, episodes_len)
        # estimate u^E using the demonstrations
        mdp.calculate_u_expert(experiment)

        # run IRL methods
        returns_list.append(get_irl_returns(mdp))
    experim = np.array(returns_list)
    average = experim.mean(axis=0)
    std = experim.std(axis=0)
    print(average)
    print(std)
 
def main():
    ##### Running Cartpole #####
    env = gym.make("CartPole-v1")
    observation_space = env.observation_space.shape[0]
    num_actions = env.action_space.n
    while True:
        state = env.reset()
        while True:
            env.render()
            action = get_action(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state = state_next
            if terminal:
                break

    # import ipdb;ipdb.set_trace()
    # table = np.hstack([average[:,0:2], std[:,0:2]])
    # pd.DataFrame(table).round(3).to_csv('../../files/cartpole_various_uE.csv',
    #                 header=['l1_mean', 'l2_mean', 'l1_std', 'l2_std'])

if __name__ == "__main__":
    main()
