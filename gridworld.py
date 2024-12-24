import sys
import numpy as np
import numpy.linalg as LA
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import random
from robust_mdp import MDP

random.seed(10)
np.set_printoptions(precision=2, suppress=True)


class GridWorld(object):
    """This is a gridworld example designed for trying out robust irl methods.

    Attributes:
        gamma (float): Discount value
        k (int): Number of features
        num_rows (int): Number of rows in the squared gridworld
        phi (nd_numpy): Feature matrix, m * n * k
        policy (nd_numpy): Policy, function from states to actions
        r_max (float): Maximum value for the rewards
        r_min (float): Minimun value for the rewards
        rewards (nd_numpy): Reward function, an m * n matrix
        transition (nd_numpy): Transition matrix, an m * n * n matrix
    """

    def __init__(self, num_rows, m, k, gamma, reward_type,
                 feature_type, p=[.1, .6]):
        # super(GridWorld, self).__init__()
        self.gamma = gamma
        self.num_rows = num_rows
        self.m = m
        self.n = num_rows ** 2
        self.k = k
        self.phi = self.set_feature_matrix(feature_type)
        self.rewards = self.set_rewards(reward_type)
        self.transition = None
        self.set_transition_probability(p)
        self.p_0 = self.set_initial_dist()
        self.policy = None
        self.mdp = MDP(self.n, self.m, self.k, self.transition, self.phi,
                       self.p_0, self.gamma)
        self.opt_u = None
        self.opt_policy = None

    def create_samples(self, policy, num_episodes, episode_len):
        p0 = self.p_0
        np.random.seed()
        n = self.num_rows ** 2
        gamma = self.gamma
        episodes = []
        returns = []
        rewards = self.rewards
        # cur_policy = np.argmax(policy, axis=0)
        # import ipdb; ipdb.set_trace()

        for i in range(num_episodes):
            gamma_t = 1
            # import ipdb; ipdb.set_trace()
            s0 = np.random.choice(n, p=p0)      # current state
            s1 = s0 # next state
            # rho = 0 # return or \rho
            episode = []
            for step in range(episode_len):
                # a0 = int(cur_policy[s0]) # current action
                a0 = np.random.choice(self.m, p=policy[:, s0])
                sample = [s0, a0] #, rewards[a0, s0]
                s1 = self.next_state(s0, a0)
                # rho += gamma_t * rewards[a0, s0]
                gamma_t *= gamma
                episode.append(sample)
                s0 = s1
            # print(episode)
            # returns.append(rho)
            episodes.append(episode)
        return episodes


    def set_feature_matrix(self, feature_type):
        r_min = 0
        r_max = 5
        n = self.num_rows ** 2
        m = self.m
        if feature_type == 'identity_sa':
            # set every pair of (a,s) an identical vector of 1 at the position
            # [a,s] and zero for the rest
            self.k = m * n
            phi = np.eye(m * n).reshape(m, n, m * n)
            self.phi = phi
        elif feature_type == 'identity_s':
            # set every pair of (a,s) an identical vector of 1 at the position
            # [a,s] and zero for the rest
            self.k = n
            phi = np.array([np.eye(n) for i in range(m)])
            self.phi = phi
        # set phi based on terrain type
        elif feature_type == 'terrain_type':
            self.k = int(r_max/2 - r_min + 1)
            k = self.k
            phi_s = []
            colors = np.random.rand(k, k)
            for i in range(n):
                phi_s.append(colors[random.randint(0, k - 1 )])
                # phi_s.append(np.eye(k)[random.randint(0, k - 1)])
                # phi_s.append(np.eye(k) @ (0.3 + np.random.rand(k)))
            phi_s = np.array(phi_s)
            phi = np.tile(phi_s, (m, 1, 1)) # m will be the first index of phi
            # phi_s = np.random.randint(r_min, int(r_max/2), (n, k))
            # phi_s = np.zeros((n, k))
            # for s in range(n):
                # if rewards[s] <= r_max:
                    # phi_s[s, rewards[s]] = 1
                # else:
                    # phi_s[s,k-1] = 1
            # phi = np.tile(phi_s, (m,1,1)) # m will be the first index of phi
        elif (feature_type == 'k-color'):
            k = self.k
            phi_s = np.zeros([n, k])
            indices = np.random.randint(0, k, size=n)
            for i in range(n):
                phi_s[i, indices[i]] = 1
            phi = np.array([phi_s for a in range(m)])
            # import ipdb;ipdb.set_trace()
            # print(phi)
        elif (feature_type == 'sparse'):
            k = 2
            self.k = k
            # phi_s = np.zeros([n, k])
            phi_s = np.vstack([np.ones(n), np.zeros(n)]).T
            indices = np.random.choice(n, size=n // 5, replace=False)
            for ind in indices:
                # import ipdb;ipdb.set_trace()
                phi_s[ind] = np.array([0, 1])
            phi = np.array([phi_s for a in range(m)])
            # import ipdb;ipdb.set_trace()
            # print(phi)
        elif (feature_type == 'negative'):
            k = 2
            self.k = k
            phi_s = np.tile([1, 0], n).reshape(n, k)
            phi_s[(n // 2 - 1):(n // 2 + 2),:] = np.tile([0, 1], 3).reshape(3, k)
            phi = np.array([phi_s for a in range(m)])

        # self.phi = phi
        return phi

    def set_initial_dist(self):
        # p_0 = np.zeros(self.n)
        # p_0[0] = .5
        # p_0[-1] = .5
        eps = 10 ** -3
        p_0 = np.ones(self.n) * eps
        p_0[0] = 1 - (self.n - 1) * eps
        # p_0 = np.zeros(self.n)
        # p_0[0] = 1
        p_0 = 1 / self.n * np.ones(self.n)
        # import ipdb;ipdb.set_trace()
        return p_0

    def set_rewards(self, reward_type):
        n = self.num_rows ** 2
        m = self.m
        if (reward_type == 'zero-one'):
            rewards = np.zeros(n)
            rewards[0] = -1
            rewards[-1] = 1
            rewards = np.array([rewards for i in range(m)])

        elif (reward_type == 'random'):
            r_s = 0.7 * np.random.rand(n)
            rewards = np.array([r_s for i in range(m)])
            rewards[:,0] = 0 * np.ones(m)
            rewards[:, n - 1] = 1 * np.ones(m)
            # rewards = 1/4 * np.random.rand(m, n)
            # rewards = np.random.randint(self.r_min, self.r_max - 1, (m, n))

        elif (reward_type == 'random-color'):
            k = self.k
            w = 2 * np.random.rand(k) - 1
            while np.sum(np.abs(w)) >= 1:
                w = 2 * np.random.rand(k) - 1
            # import ipdb;ipdb.set_trace()
            rewards_s = (self.phi[0] @ w) # / np.max(np.abs(w))
            rewards = np.array([rewards_s for a in range(m)])
            # print(rewards[0])

        elif (reward_type == 'negative'):
            # import ipdb; ipdb.set_trace()
            w = np.random.rand(2)
            # w = np.array([0.64, 0.35])
            while np.sum(np.abs(w)) > 1:
                w = np.random.rand(2)
            r_s = - np.ones(n) * w[0]
            # set 3 cells at the middle to negative
            r_s[(n // 2 - 1):(n // 2 + 2)] = +1 * w[1] * np.ones(3)
            rewards = np.array([r_s for i in range(m)])

        elif (reward_type == 'linear'):
            # w = 0.5 + np.random.rand(self.k)
            r_min = 0
            r_max = 5
            w = np.ones(self.k)
            rewards = self.phi @ w
            rewards[:, 0] = r_min
            rewards[:, -1] = r_max

        elif (reward_type == 'k-color'):
            # w = 0.5 + np.random.rand(self.k)
            k = self.k
            signs = np.random.choice([-1, 1], size=self.k)
            # signs = -1 * np.ones(k)
            w = signs * np.arange(k)
            rewards_s = (self.phi[0] @ w) / np.max(np.abs(w))
            rewards = np.array([rewards_s for a in range(m)])
            # import ipdb;ipdb.set_trace()
        # self.rewards = rewards
        return rewards

    def set_transition_probability(self, p):
        n = self.num_rows ** 2
        m = 4
        num_rows = self.num_rows
        transition = np.zeros((4,num_rows**2,num_rows**2))
        p1, p2 = p # .2, .2
        assert(4 * p1 + p2 == 1.0)
        action_prob = p1 * np.ones((4, 4)) + p2 * np.eye(4)
        for action in range(m):
            for state in range(n):
                # if action == 0:
                # the agent goes to the right
                if (state + 1) % num_rows != 0:
                    transition[action, state, state + 1] += action_prob[action, 0]
                else:
                    transition[action, state, state] += action_prob[action, 0]
                # the agent goes up
                if state < (n - num_rows):
                    transition[action, state, state + num_rows] += action_prob[action, 1]
                else:
                    transition[action, state, state] += action_prob[action, 1]
                # the agent goes to the left
                if (state + 1) % num_rows != 1:
                    transition[action, state, state - 1] += action_prob[action, 2]
                else:
                    transition[action, state, state] += action_prob[action, 2]
                # the agent goes down
                if state >= num_rows:
                    transition[action, state, state - num_rows] += action_prob[action, 3]
                else:
                    transition[action, state, state] += action_prob[action, 3]

        # import ipdb; ipdb.set_trace()
        assert((np.abs(np.sum(transition[:,:], axis=2) - np.ones([m, n])) < 
                            10 ** -3 * np.ones([m, n])).all())
        self.transition = transition

    def estimate_uE(self, num_episodes, episode_len):
        # clear the demonstrations
        self.mdp.demonstrations = None
        # set the optimal policy as the demonstrator policy
        self.opt_u, _ = self.mdp.dual_lp(self.rewards)
        self.opt_policy = self.mdp.u_to_policy(self.opt_u)

        # plot the policy
        self.plot_gridworld('gridworld'+str(self.num_rows), self.opt_policy)

        # create samples
        demonstrations = self.create_samples(self.opt_policy, num_episodes,
                                             episode_len)
        # passing the samples to the mdp object
        u_E = self.mdp.set_u_E(demonstrations)
        return u_E

    def get_feature_matrix(self):
        return self.phi

    def get_transition_probability(self):
        return self.transition

    def get_u_E_consistent(self):
        policy = np.zeros([self.m, self.n])
        for episode in self.mdp.demonstrations:
            for s, a in episode:
                policy[a, s] = 1
        for policy_s in policy.T:
            if abs(np.sum(policy_s)) < 10 ** -2:
                policy_s[0] = 1
                policy_s = 1/self.m * np.ones(self.m)
        u_E = self.mdp.policy_to_u(policy)
        # gamma = self.gamma
        # u_E = u_E / (np.sum(u_E) * (1 - gamma))
        # print("normalized", u)
        return u_E

    def plot_gridworld(self, filename, policy):
        num_rows = self.num_rows
        rewards = self.rewards[0,:].reshape(num_rows, num_rows)
        r_min = -1
        r_max = 1
        var = 0.1
        r_iterval = r_max - r_min
        viridis = cm.get_cmap('viridis', r_iterval)
        newcolors = viridis(np.linspace(0, 1, r_iterval+1))
        red = np.array([256/256, 0/256, 0/256, 1])
        newcolors[-1,:] = red
        newcmp = ListedColormap(newcolors)
        # data
        x = np.arange(num_rows*num_rows)
        # fig, axs = plt.subplots(figsize=(n, n), constrained_layout=True)
        fig = plt.figure()
        axs = fig.add_subplot(111)
        psm = axs.pcolormesh(rewards, cmap=newcmp, rasterized=True,
                             vmin=r_min-var, vmax=r_max+var)
        fig.colorbar(psm, ax=axs) # , ticks=range(r_iterval)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.tick_params(axis='both', left=False, top=False, right=False, 
                        bottom=False)
        if policy is not None:
            self.plot_policy(axs, policy)
        # plt.show()
        plt.savefig('../../files/'+filename+'.png') # , dpi=100, bbox_inches='tight', pad_inches=0.0
        plt.close()
        return plt

    def plot_policy(self, ax, policy):
        num_rows = self.num_rows
        num_actions = self.m
        X = np.tile(np.arange(0, num_rows, 1).repeat(4), num_rows) + 0.5
        Y = np.arange(0, num_rows, 1).repeat(4 * num_rows) + 0.5
        # U = np.zeros([X.shape[0], X.shape[0]])
        # V = np.zeros([X.shape[0], X.shape[0]])
        U = np.zeros(X.shape[0])
        V = np.zeros(X.shape[0])
        dir_x = 0
        dir_y = 0

        for s in range(num_rows ** 2):
            # action = np.argmax(policy[:,s])
            action_prob = policy[:, s]
            for action in range(num_actions):
                dir_x = 1/2 * action_prob[action] * int(math.cos(action * math.pi/2))
                dir_y = 1/2 * action_prob[action] * int(math.sin(action * math.pi/2))
                i = s * num_actions + action
                # j = num_rows * int(s % num_rows) + int(s / num_rows)
                # print(s, action, i, action_prob[action], dir_x, dir_y)
                U[i] = dir_x
                V[i] = dir_y
        # import ipdb; ipdb.set_trace()
        # Q = ax.quiver(X, Y, U, V, width=1/300, angles='xy', scale_units='xy',
        #                         scale=1, headwidth=4)# linewidth=0.2)
        # qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
        #            coordinates='figure')

    def next_state(self, s, a):
        states_list = np.arange(self.num_rows ** 2)
        P = self.transition
        return np.random.choice(states_list, p=P[a, s])

    def save_to_file(self, episodes):
        import csv
        with open('samples.csv', 'w') as f:
            writer = csv.writer(f)
            for episode in episodes:
                writer.writerows(episode)
        f.close()
        # np.savetxt("samples.csv", episodes, delimiter=",")

    def read_samples(self, file_name, num_episodes, episode_len):
        import csv
        episode = []
        episodes = []
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',') #,quoting=csv.QUOTE_NONNUMERIC
            line_count = 0
            for row in csv_reader:
                if line_count % episode_len == 0:
                    episodes.append(episode)
                    episode = []
                episode.append([int(val) for val in row])
                line_count += 1
        episodes.append(episode)
        episodes.pop(0)
        assert len(episodes) == num_episodes
        return episodes

    def return_cvar(self, returns, alpha):
        import pandas as pd

        df = pd.DataFrame(returns, columns=['returns'])
        # df = df.sort_values('returns', inplace=True, ascending=True)
        var_alpha = df.quantile(alpha)[0]
        cvar = np.mean([b for b in returns if b > var_alpha])
        return cvar


def uE_consistent():
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    gridworld = GridWorld(30, 4, 5, .9999, "random-color", "k-color")
    u_E = gridworld.estimate_uE(num_episodes=10, episode_len=30)
    true_rewards = gridworld.rewards
    # primal_LP = mdp.primal_lp(true_rewards)
    dual_LP, dual_opt_val = gridworld.mdp.dual_lp(true_rewards)
    u_Syed, _ = gridworld.mdp.Syed_mat()
    Syed_return = gridworld.mdp.get_return(gridworld.rewards, u_Syed)

    # find the modified uE
    # A = A @ (A.T @ A) 
    # u_E_fixed = np.linalg.solve(A_T, gridworld.p_0)
    P = gridworld.transition
    A_T = np.hstack([(np.eye(gridworld.n) - gridworld.gamma * P[a].T)
                                        for a in range(gridworld.m)])
    # A_T = (gridworld.mdp.I - gridworld.gamma * P).\
    #                 reshape([gridworld.m * gridworld.n, gridworld.n]).T
    A_pinv = np.linalg.pinv(A_T)
    u_E_fixed = A_pinv @ gridworld.p_0
    gridworld.mdp.u_E = u_E_fixed
    # import ipdb; ipdb.set_trace()
    # set the mdp's u_E to the modified u_E
    u_E_consistent = gridworld.get_u_E_consistent()
    gridworld.mdp.u_E = u_E_consistent
    # print(gridworld.opt_u)
    # print(u_E)
    # print(u_E_consistent)
    # print("diff imp {:.2f}".format(LA.norm(u_E - gridworld.opt_u)))
    # print("diff cons {:.2f}".format(LA.norm(u_E_consistent -
    #                                         gridworld.opt_u)))

    u_Syed_fixed, _ = gridworld.mdp.Syed_mat()
    Syed_return_fixed = gridworld.mdp.get_return(gridworld.rewards, u_Syed_fixed)
    opt_return = gridworld.mdp.get_return(gridworld.rewards, gridworld.opt_u)
    # print(Syed_return, Syed_return_fixed, opt_return)
    return [Syed_return, Syed_return_fixed, opt_return]


def gridworld_experiment():
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    gridworld = GridWorld(10, 4, 3, .9999, "random-color", "k-color")
    u_E = gridworld.estimate_uE(num_episodes=10, episode_len=10)
    true_rewards = gridworld.rewards
    # dual_LP, _ = gridworld.mdp.dual_lp(true_rewards)
    u_Syed, _ = gridworld.mdp.Syed_mat()
    cons_u_mat, _ = gridworld.mdp.Cons_u_optimal_u_mat()
    eps_list = [0.1, 1, 10.0, 100]
    cons_r_l1_eps0 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[0])
    cons_r_l2_eps0 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[0])
    cons_r_l1_eps1 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[1])
    cons_r_l2_eps1 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[1])
    cons_r_l1_eps2 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
    cons_r_l2_eps2 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
    cons_r_l1_eps3 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[3])
    cons_r_l2_eps3 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[3])

    returns_list = []
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_Syed))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps0))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps1))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps2))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps3))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_u_mat))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps0))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps1))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps2))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps3))

    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, gridworld.opt_u))
    return returns_list


def different_environment():
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    gridworld = GridWorld(4, 4, 3, .99, "k-color", "k-color", [.1, .6])
    gridworld.estimate_uE(num_episodes=10, episode_len=150)
    gridworld.mdp.u_E = gridworld.opt_u
    true_rewards = gridworld.rewards
    # dual_LP, _ = gridworld.mdp.dual_lp(true_rewards)
    gridworld.set_transition_probability([.2, .2])
    u_Syed, _ = gridworld.mdp.Syed_mat()
    cons_u_mat, _ = gridworld.mdp.Cons_u_optimal_u_mat()
    eps_list = [0.1, 1, 10.0, 100]
    cons_r_l1_eps0 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[0])
    cons_r_l2_eps0 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[0])
    cons_r_l1_eps1 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[1])
    cons_r_l2_eps1 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[1])
    cons_r_l1_eps2 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
    cons_r_l2_eps2 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
    cons_r_l1_eps3 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[3])
    cons_r_l2_eps3 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[3])

    returns_list = []
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_Syed))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps0))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps1))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps2))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps3))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_u_mat))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps0))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps1))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps2))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps3))

    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, gridworld.opt_u))
    return returns_list


def consistent_R_modified_uE():
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    env = GridWorld(8, 4, 3, .9999, "k-color", "k-color", [.1, .6])
    env.estimate_uE(num_episodes=1, episode_len=9)
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


def get_returns(env=None, num_episodes=10, episode_len=5):
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    if env is None:
        env = GridWorld(num_rows=8, m=4, k=3, gamma=.9999, reward_type="k-color",
                        feature_type="k-color", p=[.1, .6])

    env.estimate_uE(num_episodes=num_episodes, episode_len=episode_len)
    # print(episode_len, "%.2f" % np.linalg.norm(env.mdp.u_E - env.opt_u))

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


def various_episode_len():
    returns_list = []
    markers = ['.', 'o', 'v', '^', '<', '>', ',', '1', '2', '3', '4', '8', 's',
               'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    methods = ["Const Gen", "Chebyshev", "Modified uE", "Syed", "Const L1 eps",
               "Const L2", "Const L2 eps", "Optimal"]

    for i in range(15):
        exp_result = []
        for j in range(20):
            exp_result.append(get_returns(10, 15 + i))
        exp_result = np.array(exp_result)
        returns_list.append(np.mean(exp_result, axis=0))
    returns_array = np.array(returns_list).T
    for j in range(returns_array.shape[0]):
        plt.plot(np.arange(5, 5 + returns_array.shape[1]), returns_array[j, :],
                 label=methods[j], marker=markers[j])
    plt.legend()
    plt.savefig("../../files/gridworld_various_episode_len")


def same_reward_various_episode_length():
    np.random.seed(10)
    returns_list = []
    markers = ['.', 'o', 'v', '^', '<', '>', ',', '1', '2', '3', '4', '8', 's',
               'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    methods = ["Const Gen", "Chebyshev", "Projected uE", "Syed", "Const L1 eps",
               "Const L2", "Const L2 eps", "Optimal"]

    num_exp = 10
    # take the average of num_exp experiments for each episode length
    for j in tqdm(range(num_exp)):
        env = GridWorld(num_rows=5, m=4, k=2, gamma=.9999,
                        reward_type="k-color",
                        feature_type="k-color",
                        p=[.1, .6])
        exp_result = []
        # increase the episodes length
        for i in range(10, 10000, 1000):
            exp_result.append(get_returns(env, num_episodes=1, episode_len=i))
        # exp_result = np.array(exp_result)
        # returns_list.append(np.mean(exp_result, axis=0))
        returns_list.append(exp_result)

    # import ipdb; ipdb.set_trace()
    returns_array = np.mean(np.array(returns_list), axis=0).T

    for j in range(returns_array.shape[0]):
        plt.plot(np.arange(5, 5 + returns_array.shape[1]), returns_array[j, :],
                 label=methods[j], marker=markers[j])
    plt.legend()
    plt.savefig("../../files/gridworld_various_episode_len")


def huang_experiment():
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    gridworld = GridWorld(10, 4, 3, .99, "k-color", "k-color")
    u_E = gridworld.estimate_uE(num_episodes = 100, episode_len = 150)
    true_rewards = gridworld.rewards
    # dual_LP, _ = gridworld.mdp.dual_lp(true_rewards)
    u_Syed, _ = gridworld.mdp.Syed_mat()
    cons_u_mat, _ = gridworld.mdp.Cons_u_optimal_u_mat()
    u_huang_l1 = gridworld.mdp.Huang_l1(epsilon=1.01)
    u_huang_l2 = gridworld.mdp.Huang_l2(epsilon=1.01)
    eps_list = [0.1, 1, 10.0, 100]
    cons_r_l1_eps0 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[0])
    cons_r_l2_eps0 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[0])
    cons_r_l1_eps1 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[1])
    cons_r_l2_eps1 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[1])
    cons_r_l1_eps2 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
    cons_r_l2_eps2 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
    cons_r_l1_eps3 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[3])
    cons_r_l2_eps3 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[3])


    returns_list = []
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_huang_l1))
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_huang_l2))
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_Syed))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps0))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps1))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps2))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps3))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_u_mat))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps0))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps1))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps2))
    returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps3))

    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, gridworld.opt_u))
    return returns_list
    

def main(argv):
    num_experiments = 1
    # methods = ["", "E[R2]", "Syed", "BIRL", "max_ent", "opt"]
    R_returns = []
    for i in tqdm(range(num_experiments)):
        # R_returns.append(gridworld_experiment())
        # R_returns.append(different_environment())
        # R_returns.append(uE_consistent())
        # R_returns.append(consistent_R_modified_uE())
        # R_returns.append(various_episode_len())
        R_returns.append(same_reward_various_episode_length())
        # R_returns.append(huang_experiment())
    R_returns = np.array(R_returns)
    exp_avg = np.mean(R_returns, axis=0)
    exp_std = np.std(R_returns, axis=0)
    print(exp_avg)
    # print(exp_std)


if __name__ == "__main__":
    main(sys.argv[1:])
