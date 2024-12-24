import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from robust_mdp import MDP
np.set_printoptions(precision=1, suppress=True)


class PlotConvex(object):
    def __init__(self):
        print("a")

    def feasible_point(self, A, b):
        # finds the center of the largest sphere fitting in the convex hull
        norm_vector = np.linalg.norm(A, axis=1)
        A_ = np.hstack((A, norm_vector[:, None]))
        b_ = b[:, None]
        c = np.zeros((A.shape[1] + 1,))
        c[-1] = -1
        res = linprog(c, A_ub=A_, b_ub=b[:, None], bounds=(None, None))
        return res.x[:-1]

    def hs_intersection(self, A, b):
        interior_point = feasible_point(A, b)
        halfspaces = np.hstack((A, -b[:, None]))
        hs = HalfspaceIntersection(halfspaces, interior_point)
        return hs

    def plt_halfspace(self, a, b, bbox, ax):
        if a[1] == 0:
            ax.axvline(b / a[0])
        else:
            x = np.linspace(bbox[0][0], bbox[0][1], 100)
            ax.plot(x, (b - a[0]*x) / a[1])

    def add_bbox(self, A, b, xrange, yrange):
        A = np.vstack((A, [
            [-1,  0],
            [ 1,  0],
            [ 0, -1],
            [ 0,  1],
        ]))
        b = np.hstack((b, [-xrange[0], xrange[1], -yrange[0], yrange[1]]))
        return A, b

    def solve_convex_set(self, A, b, bbox, ax=None):
        A_, b_ = add_bbox(A, b, *bbox)
        interior_point = feasible_point(A_, b_)
        hs = hs_intersection(A_, b_)
        points = hs.intersections
        hull = ConvexHull(points)
        return points[hull.vertices], interior_point, hs

    def plot_convex_set(self, A, b, bbox, ax=None):
        # solve and plot just the convex set (no lines for the inequations)
        points, interior_point, hs = solve_convex_set(A, b, bbox, ax=ax)
        if ax is None:
            _, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(bbox[0])
        ax.set_ylim(bbox[1])
        ax.fill(points[:, 0], points[:, 1], 'r')
        return points, interior_point, hs

    def plot_inequalities(self, A, b, bbox, ax=None):
        # solve and plot the convex set,
        # the inequation lines, and
        # the interior point that was used for the halfspace intersections
        points, interior_point, hs = plot_convex_set(A, b, bbox, ax=ax)
        ax.plot(*interior_point, 'o')
        for a_k, b_k in zip(A, b):
            plt_halfspace(a_k, b_k, bbox, ax)
        return points, interior_point, hs


class TwoStates(object):
    def __init__(self):
        self.n = 2
        self.gamma = 0.9 # discount factor
        self.m = 2 # number of actions
        # transition probability P_(m*(n*n))
        self.transition = np.array([[[.8, .2], [0.5, 0.5]], [[.1, .9], [0.5, .5]]])
        # self.transition = np.array([[[.99, .01], [0.99, .01]],
        #                             [[.99, .01], [0.99, .01]]])
        # initial state distribution p_0(n)
        self.p_0 = 1 / 2 * np.ones(self.n)
        # rewards = get_random_reward(m, n) # doesn't matter at this point
        self.rewards = np.array([[0.01, 0.9], [0.01, 0.9]])
        # self.rewards = self.set_rewards()
        # Set the features matrix phi(m*(n*k))
        self.k = 2
        phi_s = np.eye(2)
        self.phi = np.array([phi_s for i in range(self.m)])
        self.mdp = MDP(self.n, self.m, self.k, self.transition, self.phi,
                       self.p_0, self.gamma)
        # self.num_episodes = num_episodes
        # self.episodes_len = episode_len
        self.opt_u = None
        self.opt_policy = None

    def set_rewards(self, r_s=None):
        # r_s = np.ones(self.n)
        r_s = 2 * np.random.rand(self.n) - 1
        while np.sum(np.abs(r_s)) >= 1:
            r_s = 2 * np.random.rand(self.n) - 1

        return np.array([r_s for i in range(self.m)])

    def create_samples(self, num_episodes, episode_len):
        m = self.m
        n = self.n
        transition = self.transition
        gamma = self.gamma
        policy = self.opt_policy
        episodes = []
        # experiment saves the flattened version of episodes
        experiment = []
        expr_dict = dict.fromkeys([(s, a) for s in range(self.n) for a in
                                   range(self.m)], 0)
        for _ in range(num_episodes):
            gamma_t = 1
            s0 = np.random.choice(self.n, p=self.p_0)      # current state
            s1 = 0      # next state
            episode = []
            for step in range(episode_len):
                # import ipdb;ipdb.set_trace()
                a0 = np.random.choice(m, p=policy[:, s0])
                # sample is a state-action-reward tuple
                sample = [s0, a0]
                s1 = np.random.choice(n, p=transition[a0, s0]) # one step
                gamma_t *= gamma
                episode.append(sample)
                expr_dict[(s0, a0)] += 1
                # print('[', s0, a0, s1, ']->', end='')
                s0 = s1                         # proceed to the next state
            episodes.append(episode)
            experiment.append([smpls for step in episode for smpls in sample[0:2]])
        # np.savetxt('../../files/two_states.csv', experiment, delimiter=',')
        print(expr_dict)
        return episodes

    def estimate_uE(self, num_episodes, episode_len):
        self.opt_u, opt_val = self.mdp.dual_lp(self.rewards)
        self.opt_policy = self.mdp.u_to_policy(self.opt_u)
        demonstrations = self.create_samples(num_episodes, episode_len)
        # passing the samples to the mdp object
        self.mdp.set_u_E(demonstrations)
        print(self.mdp.u_E)
        return self.mdp.u_E

    def get_mdp(self):
        return self.mdp
def get_random_reward(m, n):
    """get_random_reward.

    :param m:
    :param n:
    """
    rewards = np.zeros((m,n))

    r2 = np.random.choice([0,10], 1, p=None)
    for i in range(m):
        rewards[i] = np.array([5, r2])
    return rewards

def create_samples(m, n, transition, rewards, gamma, policy, num_episodes, episode_len):
    # cur_policy = np.argmax(policy, axis=0)
    episodes = []
    # experiment saves the flattened version of episodes
    experiment = []
    rhos = []
    for _ in range(num_episodes):
        gamma_t = 1
        s0 = 0      # current state
        s1 = 0      # next state
        rho = 0     # return or \rho
        episode = []
        for step in range(episode_len):
            # a0 = int(cur_policy[s0])            # current action
            # a0 = np.argmax(occ_freq[:,s0], axis=0)
            a0 = np.random.choice(m, p=policy[:, s0])
            # sample is a state-action-reward tuple
            # give the rewards when you leave the previous state
            sample = [s0, a0, rewards[a0, s0]]
            s1 = np.random.choice(n, p=transition[a0, s0]) # one step
            rho += gamma_t * rewards[a0, s0]
            gamma_t *= gamma
            episode.append(sample)
            # print('[', s0, a0, s1, ']->', end='')
            s0 = s1                         # proceed to the next state
            # or you can put it in here to give the rewards when coming in to
            # a state
        # print()
        rhos.append(rho)
        episodes.append(episode)
        experiment.append([smpls for step in episode for smpls in sample[0:2]])
    #   r2s.append(rewards[0, 1])
    # print(np.mean(r2s))
    # just for saving the experiment as a csv file
#   experiment = [j for epi in episodes for j in epi]
#   print(experiment)
    np.savetxt('../../files/two_states.csv', experiment, delimiter=',')
    return episodes, rhos

def plot_area(n, m, k, gamma, P, p0, phi, u_E, mu_E, v_opt):
    from scipy.spatial import ConvexHull
    from scipy.spatial import HalfspaceIntersection
    # fig = plt.figure()
    epsilon = 0.1
    I = np.array([np.eye(n) for i in range(m)])
    assert(I.shape == P.shape)
    A = (I - gamma * P)
    C = np.append(mu_E.reshape([k,1]), np.transpose(-phi.reshape(m * n, k),
                                                    [1, 0]), axis=1)
    d = np.append(np.dot(p0, v_opt) - epsilon, 
                    np.reshape(- A @ v_opt, [m * n]))

    # draw the lines
    assert(C.shape == (k, 1 + m * n))
    slope_list = [1, -1]
    intercept_list = [1, -1]
    for slope in slope_list:
        for inter in intercept_list:
            abline(plt, slope, inter)

    # import ipdb;ipdb.set_trace()
    for i in range(C.shape[1]):
        if (np.abs(C[1, i]) > 10 ** -2):
            slope =float(-C[0, i]/C[1, i])
            intercept = float(d[i]/C[1, i])
            abline(plt, slope, intercept)
            if (C[1, i] > 0):
                print("w_1 > {:.2f} w_0 + {:.2f}".format(slope, intercept))
            else:
                print("w_1 < {:.2f} w_0 + {:.2f}".format(slope, intercept))

        elif (np.abs(C[0, i]) > 10 ** -2):
            intercept = float(d[i]/C[0, i])
            plt.axvline(x=intercept, ls='--', linewidth=.3)
            if (C[0, i] > 0):
                print("w_0 > {:.2f}".format(intercept))
            else:
                print("w_0 < {:.2f}".format(intercept))

        else:
            print("There is something wrong with the plotting function")
    plt.fill([-1, 0, 1, 0], [0, 1, 0, -1], 'yellow', alpha=0.5)
    plt.fill([-0.025, 0, 0], [0.975, .96, 1], 'red', alpha=0.5)
    plt.legend(fontsize=6)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("../../files/twostates_W.png", dpi=1500)
    # plt.show()
   
def abline(plt, slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    l = 1.2
    axes.set_xlim(-l, l)
    axes.set_ylim(-l, l)
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '-', linewidth=.3, label='({:.2f}, {:.2f})'.
                                    format(slope, intercept))
        
def solve_inqualities(n, m, k, gamma, P, p0, phi, u_E, mu_E):
    I = np.array([np.eye(n) for i in range(m)])
    flat_A_T = (I - gamma * P).reshape([m * n, n]).transpose([1, 0])
    flat_u_E = u_E.reshape(m * n)
    flat_phi = phi.reshape([m * n, k]).transpose([1, 0])
    mu_E = (flat_phi @ flat_u_E).reshape([k, 1])
    p0_t = p0.reshape([1,n])
    W = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    Znn = np.zeros([n, n])
    Znk = np.zeros([n, k])
    Z1nm = np.zeros([1, n * m])
    Znmnm = np.zeros([n * m, n * m])
    Z2k = np.zeros([2 ** k, n * m + n])
    epsilon = 2
    var_dim = (m * n) + (m * n) + 1 + 1
    l_bound = np.hstack([np.zeros(var_dim - 1), -GRB.INFINITY])
    # x = [u, v, w]
    # Constraints
    #   A' u                           <= p0
    #  -A' u                           <= -p0
    #   p0' v               - U' Phi w <= epsilon
    #  -A v                 +    Phi w <= 0
    #   w                              <= 1
    # - w                              <= 1
    A1 = np.vstack([
                    # np.hstack([flat_A_T,    Znn,        Znk]),
                    # np.hstack([-flat_A_T,   Znn,        Znk]),
                    np.hstack([Z1nm,       p0_t,    -mu_E.T]),
                    np.hstack([Znmnm, -flat_A_T.T, flat_phi.T]),
                    np.hstack([Z2k, W])
                    ])
    # A = A1
    # b1 = np.hstack([-p0, p0, -epsilon, np.zeros(n * m), -np.ones(2 ** k)])
    A = A1[:, -(n + k):]
    b1 = np.hstack([-epsilon, np.zeros(n * m), -np.ones(2 ** k)])
    halfspaces = np.hstack([A, b1.reshape(b1.shape[0], 1)])
    feasible_point, r = get_interior_point(halfspaces)
    # halfspaces : ndarray of floats, shape (nineq, ndim+1)
    #     Stacked Inequalities of the form Ax + b <= 0 in format [A; b]
    # interior_point : ndarray of floats, shape (ndim,)
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    halfspaces = halfspaces[:,-3:]
    hs_proj = hs.intersections[:,-2:]
    print(halfspaces,'\n', hs_proj)

    import ipdb;ipdb.set_trace()
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    xlim, ylim = (-2, 2), (-2, 2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    x = np.linspace(-2, 2, 100)
    # symbols = ['-', '+', 'x', '*']
    # signs = [0, 0, -1, -1]
    fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}
    # for h, sym, sign in zip(halfspaces, symbols, signs):
    for h in halfspaces:
        # hlist = h.tolist()
        # fmt["hatch"] = sym
        if h[1]== 0:
            ax.axvline(-h[2]/h[0])#, label='{}x+{}y+{}=0'.format(*hlist))
            # xi = np.linspace(xlim[sign], -h[2]/h[0], 100)
            # ax.fill_between(xi, ylim[0], ylim[1], **fmt)
        else:
            ax.plot(x, (-h[2]-h[0]*x)/h[1])#, label='{}x+{}y+{}=0'.format(*hlist))
            # ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], **fmt)
    # x, y = zip(*hs.intersections)
    x, y = zip(*hs_proj)
    ax.plot(x, y, 'o', markersize=8)
    # plt.fill(x, y)
    plt.show()

def get_interior_point(halfspaces):
    from scipy.optimize import linprog
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
                                (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    x = res.x[:-1]
    y = res.x[-1]
    return x, y
def _experiment():
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    twostates = TwoStates()
    u_E = twostates.estimate_uE(num_episodes = 1, episode_len = 15)
    true_rewards = twostates.rewards
    # dual_LP, _ = gridworld.mdp.dual_lp(true_rewards)
    # u_Syed, _ = gridworld.mdp.Syed_mat()
    # cons_u_mat, _ = gridworld.mdp.Cons_u_optimal_u_mat()
    # eps_list = [0.1, 1, 10.0, 100]
    # cons_r_l1_eps0 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[0])
    # cons_r_l2_eps0 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[0])
    # cons_r_l1_eps1 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[1])
    # cons_r_l2_eps1 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[1])
    # cons_r_l1_eps2 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
    # cons_r_l2_eps2 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
    # cons_r_l1_eps3 = gridworld.mdp.Cons_r_optimal_u_l1_mat(eps_list[3])
    # cons_r_l2_eps3 = gridworld.mdp.Cons_r_optimal_u_l2_mat(eps_list[3])


    # returns_list = []
    # returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_Syed))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps0))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps1))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps2))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l1_eps3))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_u_mat))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps0))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps1))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps2))
    # returns_list.append(gridworld.mdp.get_return(true_rewards, cons_r_l2_eps3))

    # returns_list.append(gridworld.mdp.get_return(gridworld.rewards, gridworld.opt_u))
    # return returns_list
    
def varied_rewards_experiment():
    twostates = TwoStates()
    r_s = np.array([0, .1])
    for i in range(10):
        rewards = twostates.set_rewards(r_s)
        u_opt, _ = twostates.mdp.dual_lp(rewards)
        print(u_opt)
        r_s += 0.05
    u_E = twostates.estimate_uE(num_episodes = 1, episode_len = 15)
    true_rewards = twostates.rewards


def consistent_R_modified_uE():
    if env is None:
        env = TwoStates()
    env.estimate_uE(num_episodes=1, episode_len=2)
    # gridworld.mdp.u_E = gridworld.opt_u
    true_rewards = env.rewards
    # dual_LP, _ = env.mdp.dual_lp(true_rewards)

    # find u for different methods
    u_Syed, _ = env.mdp.Syed_mat()
    cons_u_mat, _ = env.mdp.Cons_u_optimal_u_mat()
    u_const_gen = env.mdp.constraint_generation()
    u_cheb = env.mdp.analytical_chebyshev()
    u_cons_r_l1_modified_uE = env.mdp.Cons_r_l1_modified_uE(0.1)
    # eps_list = [0.1, 1, 10.0, 100]
    eps_list = [0.5]
    cons_r_l1 = []
    cons_r_l2 = []
    returns_list = []
    for eps in eps_list:
        cons_r_l1.append(env.mdp.Cons_r_optimal_u_l1_mat(eps))
        cons_r_l2.append(env.mdp.Cons_r_optimal_u_l2_mat(eps))

    # add the returns of each u to the list
    returns_list.append(env.mdp.get_return(env.rewards, u_const_gen))
    returns_list.append(env.mdp.get_return(env.rewards, u_cheb))
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


def get_returns(env=None, num_episodes=1, episode_len=5):
    if env is None:
        env = TwoStates()

    env.estimate_uE(num_episodes=num_episodes, episode_len=episode_len)
    # gridworld.mdp.u_E = gridworld.opt_u
    print(episode_len, "%.2f" % np.linalg.norm(env.mdp.u_E - env.opt_u))
    # print(env.mdp.u_E)
    true_rewards = env.rewards
    # dual_LP, _ = env.mdp.dual_lp(true_rewards)

    # find u for different methods
    u_Syed, _ = env.mdp.Syed_mat()
    cons_u_mat, _ = env.mdp.Cons_u_optimal_u_mat()
    u_const_gen = env.mdp.constraint_generation()
    u_cheb = env.mdp.analytical_chebyshev()
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
    returns_list.append(env.mdp.get_return(env.rewards, u_cheb))
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
    line_styles = ['-', '--', '-.', ':']

    num_exp = 1
    env = TwoStates()
    for i in range(1, 40, 4):
        exp_result = []
        for j in range(num_exp):
            exp_result.append(get_returns(env, 1, i))
        exp_result = np.array(exp_result)
        returns_list.append(np.mean(exp_result, axis=0))
    returns_array = np.array(returns_list).T
    for j in range(returns_array.shape[0]):
        plt.plot(np.arange(5, 5 + returns_array.shape[1]), returns_array[j,:],
                 label=methods[j], marker=markers[j],
                 linestyle=line_styles[j % 4], alpha=.8)
    plt.legend()
    plt.savefig("../../files/twostates_various_episode_len")




def same_reward_various_episode_len():
    returns_list = []
    markers = ['.', 'o', 'v', '^', '<', '>', ',', '1', '2', '3', '4', '8', 's',
               'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    methods = ["Const Gen", "Chebyshev", "Projected uE", "Syed", "Const L1 eps",
               "Const L2", "Const L2 eps", "Optimal"]
    line_styles = ['-', '--', '-.', ':']

    env = TwoStates()
    for i in range(1, 150, 15):
        exp_result = []
        # import ipdb; ipdb.set_trace()
        for j in range(10):
            exp_result.append(get_returns(env, num_episodes=1, episode_len=i))
        exp_result = np.array(exp_result)
        returns_list.append(np.mean(exp_result, axis=0))

    returns_array = np.array(returns_list).T

    for j in range(returns_array.shape[0]):
        plt.plot(np.arange(5, 5 + returns_array.shape[1]), returns_array[j,:],
                 label=methods[j], marker=markers[j],
                 linestyle=line_styles[j % 4], alpha=.8)
    plt.legend()
    plt.savefig("../../files/twostates_various_episode_len")


def imperical_uE():
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    env = TwoStates()
    u_E = env.estimate_uE(num_episodes=100, episode_len=20)
    print(env.opt_u)
    print(u_E)
 
def main(argv):
    np.random.seed(85)
    num_experiments = 1
    # methods = ["", "E[R2]", "Syed", "BIRL", "max_ent", "opt"]
    R_returns = []
    for i in tqdm(range(num_experiments)):
        # R_returns.append(gridworld_experiment())
        # R_returns.append(varied_rewards_experiment())
        # R_returns.append(consistent_R_modified_uE())
        R_returns.append(same_reward_various_episode_len())
        # R_returns.append(various_episode_len())
        # imperical_uE()
        # R_returns.append(huang_experiment())
    R_returns = np.array(R_returns)
    exp_avg = np.mean(R_returns, axis=0)
    exp_std = np.std(R_returns, axis=0)
    print(exp_avg)
    print(exp_std)

if __name__== "__main__":
    main(sys.argv[1:])
