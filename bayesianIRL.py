import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gurobipy as gp
import numpy.linalg as LA
from gurobipy import GRB
# from scipy.optimize import linprog
# from robust_mdp import MDP
from twostates import TwoStates
from birl import BayesianIRL
from river_swimmer import RiverSwimmer
from cartpole import CartPole
from gridworld import GridWorld

np.set_printoptions(precision=1, suppress=True)
# from scipy.spatial import HalfspaceIntersection, ConvexHull


def plot_points(plt, R, color, alpha=.7):
    plt.axis('scaled')
    plt.grid()
    plt.set_xlim(-1.2, 1.2);
    plt.set_ylim(-1.2, 1.2);
    if len(R.shape) == 2:
        x, y = R[:, 0], R[:, 1]
    else:
        x, y = R[:, 0, 0], R[:, 0, 1]
    # import ipdb; ipdb.set_trace()
    plt.scatter(x, y, color=color, alpha=alpha)
    # plt.hlines(y=0, xmin=-1, xmax=1.2, linestyle='-.')
    # plt.hlines(y=1, xmin=-1, xmax=1.2, linestyle='-.')
    # plt.vlines(x=0, ymin=-1, ymax=1.2, linestyle='-.')
    # plt.vlines(x=1, ymin=-1, ymax=1.2, linestyle='-.')


def plot_boundaries(plt):
    plt.axline((0, 1), (1, 0), color='b', linestyle='-.')
    plt.axline((0, 1), (-1, 0), color='b', linestyle='-.')
    plt.axline((-1, 0), (0, -1), color='b', linestyle='-.')
    plt.axline((0, -1), (1, 0), color='b', linestyle='-.')


def plot_experiment(plt, R_real, R0, R2, R_bayesian):
    plot_boundaries(plt)
    plot_points(plt, R0, 'blue')
    plot_points(plt, R2, 'red')
    plot_points(plt, R_real, 'black')
    plot_points(plt, R_bayesian, 'green')
    # # plt.show()


def generate_point_R0(k, num_mcmc_samples):
    # create random samples for R in L1 (with radius 1)
    r = 1
    # generate n iid samples using generalized guassian distribution
    # for l1, the ggd becomes a laplace distro
    loc, scale = 1., 1.
    R0 = []
    for i in range(num_mcmc_samples):
        e = np.random.laplace(loc, scale, k)
        # generate n indipendent random signs
        s = np.random.choice([-1, 1], size=k)
        x = s * e
        z = np.power(np.random.rand(k), 1/1)
        # return y = r z x / ||x||_p
        R0.append(r * ((z * x) / np.sum(np.abs(x))))
    return np.array(R0)


def get_R(mdp, R1, epsilon, u_E):
    in_R2 = []
    for r in R1:
        if point_in_R2(mdp, r, epsilon, u_E):
            in_R2.append(r)
    return np.array(in_R2)


def get_R2(mdp, R1, epsilon):
    in_R2 = []
    for r in R1:
        if point_in_R2(mdp, r, epsilon):
            in_R2.append(r)
    return np.array(in_R2)


def get_R2_plain(mdp, R1, epsilon):
    in_R2 = []
    for r in R1:
        if point_in_R2_plain(mdp, r, epsilon):
            in_R2.append(r)
    return np.array(in_R2)


def point_in_R0(r):
    mask = np.sum(np.abs(r), axis=1) <= 1
    return r[mask, :]


def point_in_R2(mdp, r, epsilon, u_E=None):
    method = "baysian"
    n = mdp.n
    m = mdp.m
    # k_num = mdp.k
    # phi = mdp.phi
    I = mdp.I
    P = mdp.transition
    p_0 = mdp.p_0
    gamma = mdp.gamma
    if u_E is None:
        u_E = mdp.u_E
    flat_A = (I - gamma * P).reshape([m * n, n])
    flat_r = r.reshape(m * n)
    # flat_phi = phi.reshape([m * n, k_num]).transpose([1, 0])
    flat_u_E = u_E.reshape(m * n)
    # mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])

    # Model
    model = gp.Model("matrix")
    # x = [v]
    var_dim = n
    # l_bound = np.hstack([np.zeros(var_dim)])
    x = model.addMVar(shape=var_dim, name="x", lb=-GRB.INFINITY)
    # setting the objective
    model.setObjective(1, GRB.MINIMIZE)
    # Constraints
    # A v >= r
    model.addMConstr(flat_A, x, '>=', flat_r, name="C1")
    # p0 v <= r^T u^E + eps
    A2 = p_0.reshape([n, 1]).T
    b2 = flat_r.T @ flat_u_E + epsilon
    # import ipdb; ipdb.set_trace()
    model.addConstr(A2 @  x <= b2)
    # Checking the correctness of constraints
    model.write("../../files/" + method + ".lp")
    # Solve
    model.Params.OutputFlag = 0
    model.optimize()
    # model.printStats()
    # model.computeIIS()
    if model.status == GRB.Status.OPTIMAL:
        return True
    return False


def point_in_R2_plain(mdp, r, epsilon):
    method = "plain-R2" 
    n = mdp.n
    m = mdp.m
    k_num = mdp.k
    phi = mdp.phi
    I = mdp.I
    P = mdp.transition
    p_0 = mdp.p_0
    gamma = mdp.gamma
    u_E = mdp.u_E
    # flat_A = (I - gamma * P).reshape([m * n, n])
    flat_A = np.hstack([(np.eye(n) - gamma * P[a].T) for a in range(m)])
    flat_r = r.reshape(m * n)
    # flat_phi = phi.reshape([m * n, k_num]).transpose([1, 0])
    flat_u_E = u_E.reshape(m * n)
    # mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])

    # Model
    model = gp.Model("matrix")
    # x = [v]
    var_dim = m * n
    # l_bound = np.hstack([np.zeros(var_dim)])
    u = model.addMVar(shape=var_dim, name="u")
    # setting the objective 
    model.setObjective(flat_r @ (u), GRB.MAXIMIZE)
    # Constraints
    # A^T u  = p_0
    A0 = flat_A
    b0 = p_0#.reshape([n, 1])
    model.addConstr(A0 @ u == b0, name="C0")
    # import ipdb; ipdb.set_trace()
    # Checking the correctness of constraints
    model.write("../../files/" + method + ".lp")
    # Solve
    model.Params.OutputFlag = 0
    model.optimize()
    # model.printStats()
    # model.computeIIS()
    if (model.status == GRB.Status.OPTIMAL and 
        model.objVal - flat_r @ flat_u_E < epsilon):
        return True
    return False
        

def get_next_distinctive_index(phi):
    i = 1
    while np.sum(np.abs(phi[0, i] - phi[0, 0])) < .1:
        i += 1
    return i


def birl(mdp):
    mdp_env = mdp
    demonstrations = mdp_env.demonstrations
    beta = 10.0
    step_stdev = 0.2
    birl = BayesianIRL(mdp_env, beta, step_stdev, debug=False, mcmc_norm="l1")
    map_w, map_u, r_chain, u_chain = birl.sample_posterior(demonstrations, 10)
    # print("map_weights", map_w)
    # map_r = np.dot(birl.state_features, map_w)
    # import ipdb;ipdb.set_trace()
    # r_chain = np.repeat(r_chain, mdp_env.m, axis=0).reshape(r_chain.shape[0], mdp_env.m, mdp_env.n)
    if map_u is None:
        print("Bayesian IRL has no solution")
        map_u = np.zeros([mdp_env.m, mdp_env.n])
    bayesian_u = map_u.reshape(mdp_env.n, mdp_env.m).T
    # import ipdb;ipdb.set_trace()
    bu = mdp.closest_consistent_u(bayesian_u)
    return r_chain, bu


def evaluate_u(mdp, list_u):
    returns_list = []
    for u in list_u:
        returns_list.append(mdp.mdp.get_return(mdp.rewards, u))
    return returns_list


def two_states_experiment():
    DEBUG = False
    np.random.seed()
    two_states = TwoStates()
    u_E = two_states.estimate_uE(num_episodes=1, episode_len=10)
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 200
    bayesian_R, u_bayesian = birl(two_states.mdp)
    # R0 = 2 * np.random.rand(num_mcmc_samples, two_states.n) - 1
    W0 = generate_point_R0(two_states.k, num_mcmc_samples)
    # W0 = np.tile(W0, (1, 2)).reshape(num_mcmc_samples, two_states.m, two_states.n)
    flat_phi = two_states.phi.reshape([two_states.m * two_states.n,
                                          two_states.k]).T
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, two_states.m, two_states.n])
    # import ipdb;ipdb.set_trace()
    R0_bar = R0.mean(axis=0)
    # print("R1_bar:\n", R1_bar)
    u_R0 = two_states.mdp.min_u(R0_bar)
    R2 = get_R2(two_states.mdp, R0, epsilon=1.0)
    print(R2.shape[0], "/", R0.shape[0])
    # update the number of samples
    # num_mcmc_samples = R2.shape[0]
    R_real = np.reshape(two_states.rewards, [1, two_states.m, two_states.n])
    fig, axs = plt.subplots()
    # fig.set_size_inches(20, 4)
    # fig.set_dpi(100)
    plot_experiment(axs, R_real, R0, R2, bayesian_R)
    # plt.show()
    plt.savefig('../../files/baysian.png')
    R2_bar = R2.mean(axis=0)
    u_R2 = two_states.mdp.min_u(R2_bar)
    u_Syed, _ = two_states.mdp.Syed_mat()
    R1_return = two_states.mdp.get_return(two_states.rewards, u_R0)
    R2_return = two_states.mdp.get_return(two_states.rewards, u_R2)
    Syed_return = two_states.mdp.get_return(two_states.rewards, u_Syed)
    bayesian_return = two_states.mdp.get_return(two_states.rewards, u_bayesian)
    opt_return = two_states.mdp.get_return(two_states.rewards, two_states.opt_u)
    # if Syed_return > R2_return:
    #     import ipdb;ipdb.set_trace()
    if DEBUG == True:
        print("u^*:\n", two_states.opt_u)
        print("u_E:\n", u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
    # print("R1 return:\t", R1_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    return R1_return, R2_return, Syed_return, bayesian_return, opt_return


def two_states_R2_R3():
    DEBUG = False
    np.random.seed()
    two_states = TwoStates()
    # estimated uE for R2
    hat_u_E = two_states.estimate_uE(num_episodes=10, episode_len=100)
    # real uE (optimal u) for R3
    u_E = two_states.opt_u
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 200
    bayesian_R, u_bayesian = birl(two_states.mdp)

    # Plotting
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(10, 4)
    fig.set_dpi(100)
    # plot_points(axs[0], bayesian_R, 'green', alpha=1)

    # R0 = 2 * np.random.rand(num_mcmc_samples, two_states.n) - 1
    W0 = generate_point_R0(two_states.k, num_mcmc_samples)
    # W0 = np.tile(W0, (1, 2)).reshape(num_mcmc_samples,
    #                     two_states.m, two_states.n)
    flat_phi = two_states.phi.reshape([two_states.m * two_states.n,
                                      two_states.k]).T
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, two_states.m, two_states.n])
    # import ipdb;ipdb.set_trace()
    R0_bar = R0.mean(axis=0)
    # print("R1_bar:\n", R1_bar)
    u_R0 = two_states.mdp.min_u(R0_bar)
    # R3 = get_R(two_states.mdp, R0, epsilon=1.0, u_E=u_E)
    # update the number of samples
    # num_mcmc_samples = R2.shape[0]
    R_real = np.reshape(two_states.rewards, [1, two_states.m, two_states.n])
    axs[0, 0].scatter(R_real[0, 0, 0], R_real[0, 0, 1], color="black")
    # plot_experiment(axs, R_real, R0, R2, bayesian_R)
    eps_list = [0.1, 1, 5, 10]
    color_list = ["brown", "red", "cyan", "darkgreen", "brown", "cyan"]
    for i, eps in enumerate(eps_list):
        R2 = get_R(two_states.mdp, R0, epsilon=eps, u_E=hat_u_E)
        print(R2.shape[0], "/", R0.shape[0], "for eps=", eps)
        W2 = R2[:, 0, :]
        plot_boundaries(axs[0, i])
        plot_points(axs[0, i], W2, color_list[i], alpha=.3)

    for i, eps in enumerate(eps_list):
        R3 = get_R(two_states.mdp, R0, epsilon=eps, u_E=u_E)
        print(R3.shape[0], "/", R0.shape[0], "for eps=", eps)
        W3 = R3[:, 0, :]
        plot_boundaries(axs[1, i])
        plot_points(axs[1, i], W3, color_list[i], alpha=.3)

    # plt.show()
    plt.savefig('../../files/baysian-twostates.png')
    # R2 = get_R(two_states.mdp, R0, epsilon=1.0, u_E=hat_u_E)
    # print(R2.shape[0], "/", R0.shape[0])
    # R2_bar = R2.mean(axis=0)
    # returns_list = []
    # import ipdb;ipdb.set_trace()
    R2_bar = R2.mean(axis=0)
    u_R2 = two_states.mdp.min_u(R2_bar)
    u_Syed, _ = two_states.mdp.Syed_mat()
    R1_return = two_states.mdp.get_return(two_states.rewards, u_R0)
    R2_return = two_states.mdp.get_return(two_states.rewards, u_R2)
    Syed_return = two_states.mdp.get_return(two_states.rewards, u_Syed)
    bayesian_return = two_states.mdp.get_return(two_states.rewards, u_bayesian)
    opt_return = two_states.mdp.get_return(two_states.rewards, two_states.opt_u)
    # if Syed_return > R2_return:
    #     import ipdb;ipdb.set_trace()
    if DEBUG is True:
        print("u^*:\n", two_states.opt_u)
        print("u_E:\n", u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
    # print("R1 return:\t", R1_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    return R1_return, R2_return, Syed_return, bayesian_return, opt_return


def river_swimmer_experiment():
    np.random.seed()
    river_swimmer = RiverSwimmer(n=5, gamma=.99, feature_type="negative")
    u_E = river_swimmer.estimate_uE(num_episodes=1, episode_len=5)
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 300
    bayesian_R, u_bayesian = birl(river_swimmer.mdp)
    # create random samples for R
    # W0 = 2 * np.random.rand(num_mcmc_samples, river_swimmer.k) - 1
    # W0 = point_in_R0(W0)
    W0 = generate_point_R0(river_swimmer.k, num_mcmc_samples)
    # update the number of samples
    num_mcmc_samples = W0.shape[0]
    flat_phi = river_swimmer.phi.reshape([river_swimmer.m * river_swimmer.n,
                                          river_swimmer.k]).transpose([1, 0])
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, river_swimmer.m,
                                  river_swimmer.n])
    R0_bar = R0.mean(axis=0)
    # print("R0_bar:\n", R0_bar)
    u_R0 = river_swimmer.mdp.min_u(R0_bar)
    R2 = get_R2(river_swimmer.mdp, R0, epsilon=1.1)
    W2 = R2[:, 0, [0, river_swimmer.n - 1]]
    # print(R0.shape[0], '\t  in R0')
    # print(R2 '\t points in R2')
    # plotting the points
    R_real = river_swimmer.rewards[:, [0, river_swimmer.n - 1]].\
                            reshape([1, river_swimmer.m, river_swimmer.k])
    # plot_experiment(R_real, W0, W2, bayesian_R)
    R2_bar = R2.mean(axis=0)
    # import ipdb;ipdb.set_trace()
    u_R2 = river_swimmer.mdp.min_u(R2_bar)
    u_Syed, _ = river_swimmer.mdp.Syed_mat()
    R0_return = river_swimmer.mdp.get_return(river_swimmer.rewards, u_R0)
    R2_return = river_swimmer.mdp.get_return(river_swimmer.rewards, u_R2)
    Syed_return = river_swimmer.mdp.get_return(river_swimmer.rewards, u_Syed)
    bayesian_return = river_swimmer.mdp.get_return(river_swimmer.rewards,
                                                   u_bayesian)
    opt_return = river_swimmer.mdp.get_return(river_swimmer.rewards,
                                              river_swimmer.opt_u)
    DEBUG = False
    # if Syed_return > R0_return:
        # import ipdb;ipdb.set_trace()

    if DEBUG is True:
        print(R0.shape[0])
        print(R2.shape[0])
        print("u_E:\n", u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
    # print("R0 return:\t", R0_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    return R0_return, R2_return, Syed_return, bayesian_return, opt_return


def river_swimmer_change_dynamics():
    np.random.seed()
    river_swimmer = RiverSwimmer(n=5, gamma=.99, feature_type="negative")
    u_E = river_swimmer.estimate_uE(num_episodes=1, episode_len=5)
    river_swimmer.mdp.u_E = river_swimmer.opt_u
    u_E = river_swimmer.mdp.u_E
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 300
    bayesian_R, u_bayesian = birl(river_swimmer.mdp)
    # create random samples for R
    # W0 = 2 * np.random.rand(num_mcmc_samples, river_swimmer.k) - 1
    # W0 = point_in_R0(W0)
    W0 = generate_point_R0(river_swimmer.k, num_mcmc_samples)
    # update the number of samples
    num_mcmc_samples = W0.shape[0]
    flat_phi = river_swimmer.phi.reshape([river_swimmer.m * river_swimmer.n,
                                          river_swimmer.k]).transpose([1, 0])
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, river_swimmer.m,
                                  river_swimmer.n])
    R0_bar = R0.mean(axis=0)
    # print("R0_bar:\n", R0_bar)
    u_R0 = river_swimmer.mdp.min_u(R0_bar)
    R2 = get_R2(river_swimmer.mdp, R0, epsilon=1.1)
    W2 = R2[:, 0, [0, river_swimmer.n - 1]]
    # print(R0.shape[0], '\t  in R0')
    # print(R2 '\t points in R2')
    # plotting the points
    # R_real = river_swimmer.rewards[:, [0, river_swimmer.n - 1]].\
    #                         reshape([1, river_swimmer.m, river_swimmer.k])
    # plot_experiment(R_real, W0, W2, bayesian_R)
    R2_bar = R2.mean(axis=0)
    # import ipdb;ipdb.set_trace()
    u_R2 = river_swimmer.mdp.min_u(R2_bar)
    u_Syed, _ = river_swimmer.mdp.Syed_mat()
    R0_return = river_swimmer.mdp.get_return(river_swimmer.rewards, u_R0)
    R2_return = river_swimmer.mdp.get_return(river_swimmer.rewards, u_R2)
    Syed_return = river_swimmer.mdp.get_return(river_swimmer.rewards, u_Syed)
    bayesian_return = river_swimmer.mdp.get_return(river_swimmer.rewards,
                                                   u_bayesian)
    opt_return = river_swimmer.mdp.get_return(river_swimmer.rewards,
                                              river_swimmer.opt_u)
    DEBUG = False
    # if Syed_return > R0_return:
        # import ipdb;ipdb.set_trace()

    if DEBUG is True:
        print(R0.shape[0])
        print(R2.shape[0])
        print("u_E:\n", u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
    # print("R0 return:\t", R0_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    return R0_return, R2_return, Syed_return, bayesian_return, opt_return


def river_swimmer_varied_eps_experiment():
    # DEBUG = True
    DEBUG = False
    np.random.seed()
    river_swimmer = RiverSwimmer(n=5, gamma=.99, feature_type="negative")
    u_E = river_swimmer.estimate_uE(num_episodes=10, episode_len=5)
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 300
    bayesian_R, u_bayesian = birl(river_swimmer.mdp)
    # Plotting
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(20, 4)
    fig.set_dpi(100)
    plot_points(axs[0], bayesian_R, 'green', alpha=1)
    # create random samples for R
    # W0 = 2 * np.random.rand(num_mcmc_samples, river_swimmer.k) - 1
    # W0 = point_in_R0(W0)
    W0 = generate_point_R0(river_swimmer.k, num_mcmc_samples)
    # update the number of samples
    num_mcmc_samples = W0.shape[0]
    flat_phi = river_swimmer.phi.reshape([river_swimmer.m * river_swimmer.n,
                                          river_swimmer.k]).transpose([1, 0])
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, river_swimmer.m, river_swimmer.n])
    # import ipdb; ipdb.set_trace()
    R0_bar = R0.mean(axis=0)
    plot_boundaries(axs[0])
    plot_points(axs[0], W0, 'blue', alpha=.3)

    R_real = river_swimmer.rewards[:, [river_swimmer.n - 2, river_swimmer.n - 1]]
    plot_points(axs[0], R_real, 'black', alpha=1)

    # print("R0_bar:\n", R0_bar)
    u_R0 = river_swimmer.mdp.min_u(R0_bar)
    eps_list = [0.8, 3, 10]
    color_list = ["brown", "red", "cyan", "darkgreen", "brown", "cyan"]
    for i, eps in enumerate(eps_list):
        R2 = get_R2(river_swimmer.mdp, R0, epsilon=eps)
        print(R2.shape[0], "/", R0.shape[0], "for eps=", eps)
        W2 = R2[:, 0, [0, river_swimmer.n - 1]]
        plot_boundaries(axs[i + 1])
        plot_points(axs[i + 1], W2, color_list[i], alpha=.3)
    # print(R0.shape[0], '\t  in R0')
    # print(R2 '\t points in R2')
    # plotting the points
    plt.savefig('../../files/baysian-riverswimmer.png')
    R2_bar = R2.mean(axis=0)
    u_R2 = river_swimmer.mdp.min_u(R2_bar)
    u_Syed, _ = river_swimmer.mdp.Syed_mat()
    returns_list = []
    returns_list.append(river_swimmer.mdp.get_return(river_swimmer.rewards, u_R0))
    returns_list.append(river_swimmer.mdp.get_return(river_swimmer.rewards, u_R2))
    returns_list.append(river_swimmer.mdp.get_return(river_swimmer.rewards, u_Syed))
    returns_list.append(river_swimmer.mdp.get_return(river_swimmer.rewards,
                                                   u_bayesian))
    returns_list.append(river_swimmer.mdp.get_return(river_swimmer.rewards,
                                              river_swimmer.opt_u))
    # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    if DEBUG == True:
        print("u_E:\n", u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
        print("u opt:\n", river_swimmer.opt_u)
    # print("R0 return:\t", R0_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    return returns_list


def gridworld_experiment():
    plot_flag = 0
    # DEBUG = True
    # DEBUG = False
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    gridworld = GridWorld(5, 4, 4, .99, "k-color", "k-color")
    u_E = gridworld.estimate_uE(num_episodes=100, episode_len=50)
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 300
    # import ipdb;ipdb.set_trace()
    bayesian_R, u_bayesian = birl(gridworld.mdp)
    # bayesian_R = bayesian_R[:, 0, [0, gridworld.n // 2]]
    # generate random samples for R
    # W0 = 2 * np.random.rand(num_mcmc_samples, gridworld.k) - 1
    # W0 = point_in_R0(W0)
    W0 = generate_point_R0(gridworld.k, num_mcmc_samples)

    # flat_phi [k x (m . n)]
    flat_phi = gridworld.phi.reshape([gridworld.m * gridworld.n, gridworld.k]).T
    # update the number of samples
    num_mcmc_samples = W0.shape[0]
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, gridworld.m, gridworld.n])
    R0_bar = R0.mean(axis=0)
    # print("R0_bar:\n", R0_bar)
    u_R0 = gridworld.mdp.min_u(R0_bar)
    R2 = get_R2(gridworld.mdp, R0, epsilon=7.812)
    if len(R2) == 0:
        exit("ERROR: set R2 is empty")
    # print(R0.shape[0], '\t  in R0')
    # print(R2 '\t points in R2')
    # import ipdb;ipdb.set_trace()
    W2 = R2[:, 0, [0, gridworld.n // 2]]
    # R_real = gridworld.rewards[:, [0, gridworld.n // 2]].reshape([1,
    #                                                     gridworld.m, gridworld.k])
    # plotting the points
    if plot_flag: plot_experiment(R_real, W0, W2, bayesian_R)
    R2_bar = R2.mean(axis=0)
    u_R2 = gridworld.mdp.min_u(R2_bar)
    u_Syed, _ = gridworld.mdp.Syed_mat()
    # max_ent_reward = gridworld.mdp.max_ent_irl(0.9, 100)
    max_ent_reward = np.zeros(gridworld.n)
    max_ent_reward = np.array([max_ent_reward for a in range(gridworld.mdp.m)])
    max_ent_u = gridworld.mdp.min_u(max_ent_reward)
    # import ipdb;ipdb.set_trace()
    R0_return = gridworld.mdp.get_return(gridworld.rewards, u_R0)
    R2_return = gridworld.mdp.get_return(gridworld.rewards, u_R2)
    Syed_return = gridworld.mdp.get_return(gridworld.rewards, u_Syed)
    bayesian_return = gridworld.mdp.get_return(gridworld.rewards, u_bayesian)
    max_ent_return = gridworld.mdp.get_return(gridworld.rewards, max_ent_u)
    opt_return = gridworld.mdp.get_return(gridworld.rewards, gridworld.opt_u)
    print(R2.shape[0], "/", R0.shape[0])
    # if Syed_return > R2_return:
    #     import ipdb;ipdb.set_trace()
    if DEBUG == True:
        print(R0.shape[0])
        print(R2.shape[0])
        print("u^*:\n", gridworld.opt_u)
        print("u_E:\n", u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
    # print("R0 return:\t", R0_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    returns_list = [R0_return, R2_return, Syed_return, bayesian_return,
                    max_ent_return, opt_return]
    return returns_list


def gridworld_varied_eps_experiment():
    # DEBUG = True
    DEBUG = False
    np.random.seed()
    num_episodes, episode_len = 1, 15
    # gridworld = GridWorld(5, 4, 2, .99, "k-color", "k-color")
    gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    u_E = gridworld.estimate_uE(num_episodes, episode_len)
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 300
    bayesian_R, u_bayesian = birl(gridworld.mdp)
    plot_boundaries()
    plot_points(bayesian_R, 'green', alpha=1)
    # create random samples for R
    # W0 = 2 * np.random.rand(num_mcmc_samples, gridworld.k) - 1
    # W0 = point_in_R0(W0)
    W0 = generate_point_R0(gridworld.k, num_mcmc_samples)
    # update the number of samples
    num_mcmc_samples = W0.shape[0]
    flat_phi = gridworld.phi.reshape([gridworld.m * gridworld.n, gridworld.k]).T
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, gridworld.m, gridworld.n])
    # import ipdb; ipdb.set_trace()
    R0_bar = R0.mean(axis=0)
    plot_points(W0, 'blue', alpha=.3)
    R_real = gridworld.rewards[:, [gridworld.n - 2, gridworld.n - 1]]
    print(R_real[0])
    plot_points(R_real, 'black', alpha=1)
    # # plt.show()

    # print("R0_bar:\n", R0_bar)
    u_R0 = gridworld.mdp.min_u(R0_bar)
    eps_list = [3, 10, 50]
    color_list = ["yellow", "red", "cyan", "darkgreen", "brown", "cyan"]
    for i, eps in enumerate(eps_list):
        R2 = get_R2(gridworld.mdp, R0, epsilon=eps)
        print(R2.shape[0], "/", R0.shape[0], "for eps=", eps)
        W2 = R2[:, 0, [0, gridworld.n - 1]]
        plot_points(W2, color_list[i], alpha=.3)
    # print(R0.shape[0], '\t  in R0')
    # print(R2 '\t points in R2')
    # plotting the points
    plt.savefig("../../files/baysian-gridworld-" + str(episode_len) + ".png")
    R2_bar = R2.mean(axis=0)
    u_R2 = gridworld.mdp.min_u(R2_bar)
    u_Syed, _ = gridworld.mdp.Syed_mat()
    returns_list = []
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_R0))
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_R2))
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards, u_Syed))
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards,
                                                   u_bayesian))
    returns_list.append(gridworld.mdp.get_return(gridworld.rewards,
                                              gridworld.opt_u))
    # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    if DEBUG == True:
        print("u_E:\n", u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
        print("u opt:\n", gridworld.opt_u)
    # print("R0 return:\t", R0_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    return returns_list


def gridworld_change_dynamics():
    plot_flag = 0
    # DEBUG = True
    # DEBUG = False
    np.random.seed()
    # gridworld = GridWorld(5, 4, 2, .99, "negative", "negative")
    gridworld = GridWorld(5, 4, 4, .99, "k-color", "k-color")
    u_E = gridworld.estimate_uE(num_episodes=10, episode_len=50)
    gridworld.set_transition_probability([.2, .2])
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 300
    # import ipdb;ipdb.set_trace()
    bayesian_R, u_bayesian = birl(gridworld.mdp)
    # bayesian_R = bayesian_R[:, 0, [0, gridworld.n // 2]]
    # generate random samples for R
    # W0 = 2 * np.random.rand(num_mcmc_samples, gridworld.k) - 1
    # W0 = point_in_R0(W0)
    W0 = generate_point_R0(gridworld.k, num_mcmc_samples)

    # flat_phi [k x (m . n)]
    flat_phi = gridworld.phi.reshape([gridworld.m * gridworld.n, gridworld.k]).T
    # update the number of samples
    num_mcmc_samples = W0.shape[0]
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, gridworld.m, gridworld.n])
    R0_bar = R0.mean(axis=0)
    # print("R0_bar:\n", R0_bar)
    u_R0 = gridworld.mdp.min_u(R0_bar)
    R2 = get_R2(gridworld.mdp, R0, epsilon=7.812)
    if len(R2) == 0:
        exit("ERROR: set R2 is empty")
    # print(R0.shape[0], '\t  in R0')
    # print(R2 '\t points in R2')
    # import ipdb;ipdb.set_trace()
    W2 = R2[:, 0, [0, gridworld.n // 2]]
    # R_real = gridworld.rewards[:, [0, gridworld.n // 2]].reshape([1,
    #                                                     gridworld.m, gridworld.k])
    # plotting the points
    if plot_flag: plot_experiment(R_real, W0, W2, bayesian_R)
    R2_bar = R2.mean(axis=0)
    u_R2 = gridworld.mdp.min_u(R2_bar)
    u_Syed, _ = gridworld.mdp.Syed_mat()
    # max_ent_reward = gridworld.mdp.max_ent_irl(0.9, 100)
    max_ent_reward = np.zeros(gridworld.n)
    max_ent_reward = np.array([max_ent_reward for a in range(gridworld.mdp.m)])
    max_ent_u = gridworld.mdp.min_u(max_ent_reward)
    # import ipdb;ipdb.set_trace()
    R0_return = gridworld.mdp.get_return(gridworld.rewards, u_R0)
    R2_return = gridworld.mdp.get_return(gridworld.rewards, u_R2)
    Syed_return = gridworld.mdp.get_return(gridworld.rewards, u_Syed)
    bayesian_return = gridworld.mdp.get_return(gridworld.rewards, u_bayesian)
    max_ent_return = gridworld.mdp.get_return(gridworld.rewards, max_ent_u)
    opt_return = gridworld.mdp.get_return(gridworld.rewards, gridworld.opt_u)
    print(R2.shape[0], "/", R0.shape[0])
    # if Syed_return > R2_return:
    #     import ipdb;ipdb.set_trace()
    if DEBUG == True:
        print(R0.shape[0])
        print(R2.shape[0])
        print("u^*:\n", gridworld.opt_u)
        print("u_E:\n", u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
    # print("R0 return:\t", R0_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    returns_list = [R0_return, R2_return, Syed_return, bayesian_return,
                    max_ent_return, opt_return]
    return returns_list


def gridworld_R2_R3():
    DEBUG = False
    np.random.seed()
    gridworld = GridWorld(4, 4, 2, .99, "random-color", "k-color")
    # estimated uE for R2
    hat_u_E = gridworld.estimate_uE(num_episodes=50, episode_len=300)
    # real uE (optimal u) for R3
    u_E = gridworld.opt_u
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 200
    bayesian_R, u_bayesian = birl(gridworld.mdp)

    eps_list = [0.1, 1, 5, 10, 30, 40, 100]
    # Plotting
    fig, axs = plt.subplots(2, len(eps_list))
    fig.set_size_inches(10, 4)
    fig.set_dpi(100)
    # plot_points(axs[0], bayesian_R, 'green', alpha=1)

    # R0 = 2 * np.random.rand(num_mcmc_samples, gridworld.n) - 1
    W0 = generate_point_R0(gridworld.k, num_mcmc_samples)
    # W0 = np.tile(W0, (1, 2)).reshape(num_mcmc_samples,
    #                     gridworld.m, gridworld.n)
    flat_phi = gridworld.phi.reshape([gridworld.m * gridworld.n,
                                      gridworld.k]).T
    # import ipdb;ipdb.set_trace()
    ind = get_next_distinctive_index(gridworld.phi)
    # print(ind)
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, gridworld.m, gridworld.n])
    R0_bar = R0.mean(axis=0)
    # print("R1_bar:\n", R1_bar)
    u_R0 = gridworld.mdp.min_u(R0_bar)
    # R3 = get_R(gridworld.mdp, R0, epsilon=1.0, u_E=u_E)
    # update the number of samples
    # num_mcmc_samples = R2.shape[0]
    color_list = ["brown", "red", "cyan", "darkgreen", "brown", "cyan"]
    R_real = np.reshape(gridworld.rewards, [1, gridworld.m, gridworld.n])
    axs[0, 0].scatter(R_real[0, 0, 0], R_real[0, 0, 1], color="black")
    # plot_experiment(axs, R_real, R0, R2, bayesian_R)
    for i, eps in enumerate(eps_list):
        R2 = get_R(gridworld.mdp, R0, epsilon=eps, u_E=hat_u_E)
        print(R2.shape[0], "/", R0.shape[0], "for eps=", eps)
        plot_boundaries(axs[0, i])
        if R2.shape[0] != 0:
            W2 = R2[:, 0, [0, ind]]
            plot_points(axs[0, i], W2, color_list[i % len(color_list)], alpha=.3)

    for i, eps in enumerate(eps_list):
        R3 = get_R(gridworld.mdp, R0, epsilon=eps, u_E=u_E)
        print(R3.shape[0], "/", R0.shape[0], "for eps=", eps)
        plot_boundaries(axs[1, i])
        if R3.shape[0] != 0:
            W3 = R3[:, 0, [0, ind]]
            plot_points(axs[1, i], W3, color_list[i % len(color_list)], alpha=.3)

    # plt.show()
    plt.savefig('../../files/baysian-gridworld.png')
    # R2 = get_R(gridworld.mdp, R0, epsilon=1.0, u_E=hat_u_E)
    # print(R2.shape[0], "/", R0.shape[0])
    R2_bar = R2.mean(axis=0)
    # returns_list = []
    # import ipdb;ipdb.set_trace()
    R2_bar = R2.mean(axis=0)
    u_R2 = gridworld.mdp.min_u(R2_bar)
    u_Syed, _ = gridworld.mdp.Syed_mat()
    R1_return = gridworld.mdp.get_return(gridworld.rewards, u_R0)
    R2_return = gridworld.mdp.get_return(gridworld.rewards, u_R2)
    Syed_return = gridworld.mdp.get_return(gridworld.rewards, u_Syed)
    bayesian_return = gridworld.mdp.get_return(gridworld.rewards, u_bayesian)
    opt_return = gridworld.mdp.get_return(gridworld.rewards, gridworld.opt_u)
    # if Syed_return > R2_return:
    #     import ipdb;ipdb.set_trace()
    if DEBUG:
        print("u^*:\n", gridworld.opt_u)
        print("u_E:\n", hat_u_E)
        print("u R0:\n", u_R0)
        print("u R2:\n", u_R2)
        print("u Syed:\n", u_Syed)
        print("u Bayesian:\n", u_bayesian)
    print("difference", LA.norm(hat_u_E - u_E))
    # print("R1 return:\t", R1_return)
    # print("R2 return:\t", R2_return)
    # print("Syed return:\t", Syed_return)
    return R1_return, R2_return, Syed_return, bayesian_return, opt_return


def cartpole_experiment():
    plot_flag = 0
    DEBUG = 0
    np.random.seed()
    cartpole = CartPole(gamma=0.99, num_states=1000)
    u_E = cartpole.estimate_uE(num_episodes = 1, episode_len = 100)
    # number of reward samples drawn from a uniform disribution
    num_mcmc_samples = 300
    # import ipdb;ipdb.set_trace()
    # create random samples for W_[num_samples, k]
    # start = time.time()
    # end = time.time()
    # print("BIRL", end - start)
    W0 = generate_point_R0(cartpole.k, num_mcmc_samples)
    # flat_phi [k x (m . n)]
    flat_phi = cartpole.phi.reshape([cartpole.m * cartpole.n, cartpole.k]).T
    R0 = (W0 @ flat_phi).reshape([num_mcmc_samples, cartpole.m, cartpole.n])
    R0_bar = R0.mean(axis=0)
    u_R0 = cartpole.mdp.min_u(R0_bar)
    start = time.time()
    R2 = get_R2(cartpole.mdp, R0, epsilon=0.045)
    print(R2.shape[0], "/", R0.shape[0])
    end = time.time()
    print("creating R2", end - start)
    if len(R2) == 0: exit("ERROR: set R2 is empty")
    # import ipdb;ipdb.set_trace()
    # plotting the points
    if plot_flag: plot_experiment(R_real, W0, W2, bayesian_R)
    start = time.time()
    R2_bar = R2.mean(axis=0)
    u_R2 = cartpole.mdp.min_u(R2_bar)
    end = time.time()
    print("max E[R2]", end - start)
    start = time.time()
    u_Syed, _ = cartpole.mdp.Syed_mat()
    end = time.time()
    print("Syed", end - start)
    start = time.time()
    # max_ent_reward = cartpole.mdp.max_ent_irl(0.9, 100)
    max_ent_reward = np.zeros([cartpole.m, cartpole.n])
    # max_ent_reward = np.array([max_ent_reward for a in range(cartpole.mdp.m)])
    u_max_ent = cartpole.mdp.min_u(max_ent_reward)
    end = time.time()
    print("Max ent", end - start)
    start = time.time()
    bayesian_R, u_bayesian = birl(cartpole.mdp)
    # bayesian_R, u_bayesian = np.zeros([cartpole.m, cartpole.n]), np.zeros(
    #                                             [cartpole.m, cartpole.n])
    end = time.time()
    print("BIRL", end - start)
    u_list = []
    u_list.append(u_R0)
    u_list.append(u_R2)
    u_list.append(u_Syed)
    u_list.append(u_bayesian)
    u_list.append(cartpole.opt_u)
    # import ipdb;ipdb.set_trace()
    returns_list = evaluate_u(cartpole, u_list)
    return returns_list

def main():
    num_experiments = 1
    methods = ["E[R0]", "E[R2]", "Syed", "BIRL", "max_ent", "opt"]
    R_returns = []
    for i in tqdm(range(num_experiments)):
        # R_returns.append(two_states_experiment())
        # R_returns.append(two_states_R2_R3())
        # R_returns.append(river_swimmer_experiment())
        # R_returns.append(river_swimmer_varied_eps_experiment())
        # R_returns.append(gridworld_experiment())
        # R_returns.append(gridworld_change_dynamics())
        # R_returns.append(gridworld_varied_eps_experiment())
        R_returns.append(gridworld_R2_R3())
        # R_returns.append(cartpole_experiment())
    R_returns = np.array(R_returns)
    exp_avg = np.mean(R_returns, axis=0)
    exp_std = np.std(R_returns, axis=0)
    print(exp_avg)
    print(exp_std)
    # print("|", "\t|\t".join(methods), "\t|")
    # print("|", np.array2string(exp_avg, separator='\t|\t')[1:-1], "\t|")
    # print("|", np.array2string(exp_std, separator='\t|\t')[1:-1], "\t|")


if __name__ == "__main__":
    main()
