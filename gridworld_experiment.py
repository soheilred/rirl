import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pickle
import pprint
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from os import cpu_count
import itertools
import random
import time
from timeit import default_timer as timer
from robust_mdp import *
from gridworld import GridWorld

np.set_printoptions(precision=2, suppress=True)
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
plt.rcParams['font.family'] = "Comfortaa" # font.fantasy # sans-serif
name_list = ["Optimal", "Abbeel LP", "Consistent U", "Syed", "Consisten R l1", 
            "Consistent R L2"]

def print_resutls():
    print("optimal values")
    print("Abbeel dual vi\t\t", Abbeel_vi_opt_val)
    print("Abbeel dual lp\t\t", Abbeel_lp_opt_val)
    # print("Abbeel prob\t\t", simplex_opt_val)
    print("="*20)
    print("Optimal\n", optimal_policy)
    print("Primal\n", primal_LP)
    print("Dual \n", dual_LP)
    print("Abbeel dual vi\n", Abbeels_vi_u)
    print("Abbeel dual lp\n", Abbeels_lp_u)
    # print("Abbeel probability simplex\n", Abbeels_prob_simplex)
    print("Abbeel consistent U's u\n", consistent_u_u)
    # print("Abbeel consistent U's r\n", Abbeels_consistent_u_r)
    print("Abbeel consistent R's u\n", Abbeels_consistent_r_u)
    # print("Abbeel consistent R's r\n", Abbeels_consistent_r_r)
    # print("Abbeel consistent original\n", Abbeels_consistent_policy)
    print("Syed\n", Syeds_policy)
    print("RIRL\n", robust_policy)

    u_opt = mdp.policy_to_u(optimal_policy)
    print("optimal u, true reward\t", mdp.get_return(true_rewards, u_opt))
    print("Abbeel dual vi\t\t", mdp.get_return(true_rewards, Abbeels_vi_u))
    print("Abbeel dual lp\t\t", mdp.get_return(true_rewards, Abbeels_lp_u))
    # print("Abbeel prob\t\t", mdp.get_return(true_rewards, Abbeels_prob_simplex))
    print("Abbeel consistent U\t", 
                        mdp.get_return(true_rewards, consistent_u_u))

def print_time(run_time):
    mean_time = np.mean(run_time, axis=0)
    std_time = np.std(run_time, axis=0)
    print("=" * 40)
    print("%" * 12, " RUNNING TIME ", "%" * 12)
    print("=" * 40)
    print("Method\t\t", "Mean(s)\t", "STD")
    print("=" * 40)
    print("Abbeel VI:\t", round(mean_time[0], 3), "\t\t", round(std_time[0], 3))
    print("Const U:\t", round(mean_time[1], 3), "\t\t", round(std_time[2], 3))
    print("Syed:\t\t", round(mean_time[2], 3), "\t\t", round(std_time[1], 3))
    print("Const R l1:\t", round(mean_time[3], 3), "\t\t", round(std_time[3], 3))
    print("Const R l2:\t", round(mean_time[4], 3), "\t\t", round(std_time[4], 3))

def make_gridworld(reward_method, feature_type, n):
    num_rows = int(n) # number of rows in the square gridworld
    n = num_rows * num_rows # number of states
    m = 4 # number of actions
    r_min = 0
    r_max = 5
    gamma = 0.9 # discount factor
    gridworld = GridWorld(num_rows, m, r_min, r_max, gamma)
    # k = r_max - 2 * r_min + 1 # number of features
    gridworld.set_feature_matrix(feature_type)
    k = gridworld.k
    # features matrix phi(m*(n*k))
    phi = gridworld.get_feature_matrix()
    # transition probability P_(m*(n*n))
    P = gridworld.get_transition_probability()
    # initial state distribution p_0(n)
    p_0 = np.zeros(n)
    p_0[0] = 1
    # p_hat = 1/n * np.ones(n)
    p_hat = np.zeros(n)
    p_hat[2] = 1
    p_hat = p_0
    # p_0 = 1/n * np.ones(n)
    # true reward - creating reward for each action rewards(a,s)
    true_rewards = gridworld.set_rewards(reward_method)
    # pprint.pprint(true_rewards)
    mdp = MDP(n, m, k, P, phi, p_0, gamma)
    # using value iteration
    opt_occ_freq, _ = mdp.dual_lp(true_rewards)
    # import ipdb; ipdb.set_trace()
    optimal_policy = mdp.u_to_policy(opt_occ_freq)
    gridworld.plot_gridworld('gridworld'+str(n), optimal_policy)

    num_episodes = 10
    episodes_len = int(2 * np.sqrt(n))
    # episodes_opt is the experiment that the optimal policy is obtained from
    # uncomment if you want to generate samples
    episodes_opt, returns_opt = gridworld.create_sample(optimal_policy, p_hat,
                                                    num_episodes, episodes_len)
    # gridworld.save_to_file(episodes_opt)

    # uncomment if you want to load samples
    # episodes_opt = gridworld.read_samples('samples.csv', num_episodes, 
                                             # episodes_len)

    # passing the samples to the mdp object
    mdp.get_u_E(episodes_opt)
    # mdp.u_E = opt_occ_freq
    return mdp, optimal_policy, true_rewards

def get_returns(mdp, optimal_policy, true_rewards, epsilon):
    opt_u_dual,dual_opt_val = mdp.dual_lp(true_rewards)
    start_time = time.time()
    # Abbeels_vi_u, Abbeel_vi_reward, Abbeel_vi_opt_val = mdp.Abbeel_dual_vi()
    Abbeels_lp_u, Abbeel_lp_reward, Abbeel_lp_opt_val = mdp.Abbeel_dual_lp_mat()
    Abbeel_lp_time = time.time() - start_time
    start_time = time.time()
    Syed_u, Syed_opt_val = mdp.Syed_mat()
    Syed_time = time.time() - start_time
    start_time = time.time()
    consistent_u_u, const_u_u_opt_val = mdp.Cons_u_optimal_u_mat()
    const_u_time = time.time() - start_time
    start_time = time.time()
    consistent_r_u_l1, const_r_opt_val_l1 = mdp.Cons_r_optimal_u_l1_mat(epsilon)
    const_r_l1_time = time.time() - start_time
    start_time = time.time()
    consistent_r_u_l2, const_r_opt_val_l2 = mdp.Cons_r_optimal_u_l2_mat(epsilon)
    const_r_l2_time = time.time() - start_time
    time_list = [Abbeel_lp_time, const_u_time, Syed_time, const_r_l1_time, 
                 const_r_l2_time]

    # assert(Syed_opt_val + const_r_u_opt_val >= 0)
    u_opt = mdp.policy_to_u(optimal_policy)
    opt_return = mdp.get_return(true_rewards, u_opt)
    Abbeel_lp_return = mdp.get_return(true_rewards, Abbeels_lp_u)
    Syed_return = mdp.get_return(true_rewards, Syed_u)
    cons_u_return = mdp.get_return(true_rewards, consistent_u_u)
    cons_r_l1_return = mdp.get_return(true_rewards, consistent_r_u_l1)
    cons_r_l2_return = mdp.get_return(true_rewards, consistent_r_u_l2)

    return_list = [opt_return, Abbeel_lp_return, cons_u_return, Syed_return, 
                    cons_r_l1_return, cons_r_l2_return]

    # calculate the norm difference of occupancy frequencies
    Abbeel_lp_u_dist = LA.norm(opt_u_dual - Abbeels_lp_u, ord=2)
    # Abbeel_lp_u_dist = LA.norm(opt_u_dual - Abbeels_lp_u, ord=2)
    Syed_u_dist = LA.norm(opt_u_dual - Syed_u, ord=2)
    const_u_dist = LA.norm(opt_u_dual - consistent_u_u, ord=2)
    const_r_dist = LA.norm(opt_u_dual - consistent_r_u_l1, ord=2)
    dist_list = [Abbeel_lp_u_dist, Syed_u_dist, const_u_dist, const_r_dist]
    return np.array(return_list), np.array(dist_list), np.array(time_list)

def get_returns_test_mat(mdp, optimal_policy, true_rewards):
    epsilon = 1.0
    opt_u_dual,dual_opt_val = mdp.dual_lp(true_rewards)
    start_time = time.time()
    # Abbeels_vi_u, Abbeel_vi_reward, Abbeel_vi_opt_val = mdp.Abbeel_dual_vi()
    Abbeels_lp_u, Abbeel_lp_reward, Abbeel_lp_opt_val = mdp.Abbeel_dual_lp()
    Abbeel_lp_time = time.time() - start_time
    start_time = time.time()
    Abbeels_lp_u_mat, Abbeel_lp_reward_mat, Abbeel_lp_opt_val_mat = mdp.\
                                                    Abbeel_dual_lp_mat()
    Abbeel_lp_time_mat = time.time() - start_time
    assert(abs(Abbeel_lp_opt_val - Abbeel_lp_opt_val_mat) < 10**-1)
    assert(abs(np.sum(Abbeels_lp_u - Abbeels_lp_u_mat)) < 10**-3)
    start_time = time.time()
    # Abbeels_lp_u, Abbeel_reward, Abbeel_lp_opt_val = mdp.Abbeel_dual_lp()
    Syed_u, Syed_opt_val = mdp.Syed()
    Syed_time = time.time() - start_time
    start_time = time.time()
    Syed_u_mat, Syed_opt_val_mat = mdp.Syed_mat()
    Syed_mat_time = time.time() - start_time
    assert(abs(Syed_opt_val - Syed_opt_val_mat) < 10**-3)
    assert(abs(np.sum(Syed_u_mat - Syed_u)) < 10**-3)
    start_time = time.time()
    consistent_u_u, const_u_u_opt_val = mdp.Cons_u_optimal_u()
    const_u_time = time.time() - start_time
    start_time = time.time()
    consistent_u_u_mat, const_u_u_opt_val_mat = mdp.Cons_u_optimal_u_mat()
    const_u_time_mat = time.time() - start_time
    assert(abs(const_u_u_opt_val - const_u_u_opt_val_mat) < 10**-3)
    assert(abs(np.sum(consistent_u_u - consistent_u_u_mat)) < 10**-3)
    start_time = time.time()
    consistent_r_u_l1, const_r_opt_val_l1 = mdp.Cons_r_optimal_u_l1()
    const_r_l1_time = time.time() - start_time
    start_time = time.time()
    consistent_r_u_l1_mat, const_r_opt_val_l1_mat = mdp.Cons_r_optimal_u_l1_mat(epsilon)
    const_r_l1_mat_time = time.time() - start_time
    assert(abs(const_r_opt_val_l1 - const_r_opt_val_l1_mat) < 10**-3)
    assert(abs(np.sum(consistent_r_u_l1 - consistent_r_u_l1_mat)) < 10**-3)
    start_time = time.time()
    consistent_r_u_l2_mat, const_r_opt_val_l2_mat = mdp.Cons_r_optimal_u_l2_mat()
    const_r_l2_mat_time = time.time() - start_time
    time_list = [Abbeel_lp_time, Syed_time, const_u_time, const_r_l1_time, 
                Abbeel_lp_time_mat, const_u_time_mat, Syed_mat_time,
                const_r_l1_mat_time, const_r_l2_mat_time]

    # assert(Syed_opt_val + const_r_u_opt_val >= 0)
    u_opt = mdp.policy_to_u(optimal_policy)
    opt_return = mdp.get_return(true_rewards, u_opt)
    Abbeel_lp_return = mdp.get_return(true_rewards, Abbeels_lp_u)
    Abbeel_lp_return_mat = mdp.get_return(true_rewards, Abbeels_lp_u_mat)
    Syed_return = mdp.get_return(true_rewards, Syed_u)
    Syed_return_mat = mdp.get_return(true_rewards, Syed_u_mat)
    cons_u_return = mdp.get_return(true_rewards, consistent_u_u)
    cons_u_return_mat = mdp.get_return(true_rewards, consistent_u_u_mat)
    cons_r_l1_return = mdp.get_return(true_rewards, consistent_r_u_l1)
    cons_r_l1_return_mat = mdp.get_return(true_rewards, consistent_r_u_l1_mat)
    cons_r_l2_return_mat = mdp.get_return(true_rewards, consistent_r_u_l2_mat)

    return_list = [opt_return, Abbeel_lp_return, Abbeel_lp_return_mat,
                    Syed_return, Syed_return_mat,
                    cons_u_return, cons_u_return_mat,
                    cons_r_l1_return, cons_r_l1_return_mat, 
                    cons_r_l2_return_mat]

    # calculate the norm difference of occupancy frequencies
    Abbeel_lp_u_dist = LA.norm(opt_u_dual - Abbeels_lp_u, ord=2)
    # Abbeel_lp_u_dist = LA.norm(opt_u_dual - Abbeels_lp_u, ord=2)
    Syed_u_dist = LA.norm(opt_u_dual - Syed_u, ord=2)
    const_u_dist = LA.norm(opt_u_dual - consistent_u_u, ord=2)
    const_r_dist = LA.norm(opt_u_dual - consistent_r_u_l1, ord=2)
    dist_list = [Abbeel_lp_u_dist, Syed_u_dist, const_u_dist, const_r_dist]
    return np.array(return_list), np.array(dist_list), np.array(time_list)

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() - rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6)

def change_num_states():
    """
    The first experiment with gridworld, change the number of states and see how
    different approaches of IRL perform under different scenarios of gridworld,
    with random grids or with all the grids set to zero except for the first and
    the last.
    """
    # other reward types are'zero-one' 'linear'
    reward_type = 'random'
    # other feature matrices type 'identity_s', 'terrain_type'
    feature_type = 'identity_sa'
    zero_one_returns = []
    random_results = []
    my_range = range(2,7)
    my_list = list(my_range)
    for i in tqdm(my_range):
        print("=" * 20, i, "=" * 20)
        mdp, opt_pol, true_rewards = make_gridworld(reward_type, feature_type,i)
        experminet_returns, distances,_= get_returns(mdp, opt_pol, true_rewards)
        zero_one_returns.append(experminet_returns)
        # print(distances)
    # zero_one_returns = np.array(zero_one_returns)
    pickle.dump(zero_one_returns, open('../../files/gridworld_experiment', 'wb'))

    labels = name_list
    width = 0.1
    # plt.rcParams["figure.figsize"] = (8,6)
    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Returns')
    ax.set_xlabel('Number of rows in a square GridWorld')
    ax.set_title('GridWorld Experiment')
    # ax.set_xticklabels(labels)

    plt.grid(color='w', alpha=.35, linestyle='--')
    ax.patch.set_facecolor(color='gray')
    ax.patch.set_alpha(.35)

    i = 0
    experiments = np.array(zero_one_returns).T
    num_exper = len(zero_one_returns)
    x_pos = np.array(my_list)
    for experim in experiments:
        rect = ax.bar(x_pos + width * (i - int(num_exper/2)), experim,
                        align='center', width=width, alpha=0.5, 
                        label=labels[i])
        autolabel(rects=rect, ax=ax)
        i += 1

    fig.tight_layout()
    # fig.suptitle("Grid World comparision")
    ax.set_xticks(my_list)
    ax.legend()
    # ax.set_title("something")
    # for ax in axs.flat:
        # ax.set(xlabel='x-label', ylabel='y-label')
    # ax.show()
    fig.savefig('../../files/gridworld_experiment_figure.png')
    plt.close(fig)
    # Random gridworld experiment
    # for i in tqdm(my_range):
        # print("=" * 20, i, "=" * 20)
        # random_results.append(make_gridworld('random', i))
    # # print(random_results)
    # pickle.dump(random_results, open(../'../files/gridworld_experiment', 'wb'))
    # random_results = np.array(random_results)
    # plt.plot(my_list, random_results[:, 0], label='Optimal', color='r')
    # plt.plot(my_list, random_results[:, 1], label='Abbeel-vi', color='b')
    # plt.plot(my_list, random_results[:, 2], label='Abbeel-lp', color='g')
    # plt.plot(my_list, random_results[:, 3], label='Abbeel-cons-u',color='y')
    # plt.plot(my_list, random_results[:, 4], label='Abbeel-cons-r',color='pink')
    # plt.legend()
    # #plt.show()
    # plt.savefig('../../files/gridworld_experiment_figure-random.png')
    # plt.close()
    print("hey")

def fix_gridworld(grid_size, num_exper):
    zero_one_returns = fix_gridworld_experiment_parallel(grid_size, num_exper)
    # zero_one_returns = fix_gridworld_experiment(grid_size, num_exper)
    fix_gridworld_plot(grid_size, num_exper, zero_one_returns, name_list)

def one_experiment(grid_size, num_exper, epsilon = 0.0):
    # grid_size, num_exper = args[0], args[1] 
    # other reward types are'zero-one' 'linear'
    reward_type = 'random'
    # other feature matrices type 'identity_s', 'terrain_type'
    feature_type = 'identity_sa'
    np.random.seed()
    mdp, opt_pol, true_rewards = make_gridworld(reward_type, feature_type,
                                                grid_size)
    experminet_returns, _, run_time = get_returns(mdp, opt_pol, true_rewards, epsilon)
    return experminet_returns, run_time

def fix_gridworld_experiment_parallel(grid_size, num_exper):
    start = timer()
    pool = Pool(cpu_count(), maxtasksperchild=100)
    output = pool.starmap(one_experiment, 
                        [(grid_size, num_exper) for i in range(num_exper)])
    zero_one_returns =  []
    run_time = []
    for x in output:
        zero_one_returns.append(x[0])
        run_time.append(x[1])
    # print(output_list)
    # print(len(output_list))
    pool.close()
    # pool.join()
    print("with multiprocessing:", timer()-start) 
    print_time(run_time)
    return zero_one_returns 

def fix_gridworld_experiment(grid_size, num_exper):
    """TODO: Docstring for fix_gridworld.

    Args:
        arg1 (TODO): TODO

    Returns: TODO

    """
    start = timer()
    num_methods = len(name_list)
    zero_one_returns = np.zeros([num_exper, num_methods])
    run_time_arr = np.zeros([num_exper, num_methods - 1])
    for i in tqdm(range(num_exper)):
        experminet_returns, run_time, = one_experiment(grid_size, num_exper)
        zero_one_returns[i,:] = experminet_returns
        run_time_arr[i,:] = run_time
    print("without multiprocessing:", timer()-start) 
    print_time(run_time_arr)
    return zero_one_returns

def fix_gridworld_plot(grid_size, num_exper, zero_one_returns, names):
    zero_one_returns = np.array(zero_one_returns)
    pickle.dump(zero_one_returns, open('../../files/gridworld_experiment', 'wb'))
    experiments = np.array(zero_one_returns).T
    exp_means = np.around(np.mean(experiments, axis=1), decimals=2)
    exp_std = np.around(np.std(experiments, axis=1), decimals=2)
    exp_min = np.min(experiments, axis=1)
    exp_max = np.max(experiments, axis=1)
    # import ipdb; ipdb.set_trace()
    pprint.pprint(exp_means)

    labels = names
    width = 0.1
    plt.rcParams["figure.figsize"] = (10,6)
    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Returns')
    ax.set_xlabel( str(grid_size) + 'x' + str(grid_size) + ' GridWorld')
    ax.set_title(str(num_exper) + ' Experiment in a '+ str(grid_size) + 'x' +
                str(grid_size) + ' GridWorld')

    ax.errorbar(labels, exp_means, exp_std, fmt='ok', lw=3)
    ax.errorbar(labels, exp_means, [exp_means - exp_min, exp_max - exp_means],
                fmt='.k', ecolor='gray', lw=1)

    # rect = ax.bar(labels, exp_means, width, yerr=exp_std)
    # autolabel(rects=rect, ax=ax)
    # for experim in experiments:
        # rect = ax.bar(x_pos + width * (i - int(num_exper/2)), experim,
                        # align='center', width=width, alpha=0.5, 
                        # label=labels[i])
        # autolabel(rects=rect, ax=ax)
        # i += 1
    fig.tight_layout()
    # fig.suptitle("Grid World comparision")
    # ax.set_xticks(my_list)
    # ax.legend()
    # ax.set_title("something")
    # for ax in axs.flat:
        # ax.set(xlabel='x-label', ylabel='y-label')
    # ax.show()
    fig.savefig('../../files/gridworld_experiment_figure_'+str(grid_size)+'.png')
    plt.close(fig)

def epsilon_experiment(grid_size, num_episodes, episode_len):
    # grid_size, num_exper = args[0], args[1] 
    # other reward types are'zero-one' 'linear'
    reward_type = 'negative'
    # other feature matrices type 'identity_s', 'terrain_type'
    feature_type = 'negative'
    np.random.seed()
    # mdp, opt_pol, true_rewards = make_gridworld(reward_type, feature_type,
    #                                             grid_size)
    gridworld = GridWorld(grid_size, 4, 0.99, reward_type, feature_type)
    u_E = gridworld.estimate_uE(num_episodes = num_episodes, episode_len = episode_len)
    opt_pol = gridworld.opt_policy
    true_rewards = gridworld.rewards
    # opt_u_dual, dual_opt_val = mdp.dual_lp(true_rewards)
    # Abbeels_lp_u, Abbeel_lp_reward, Abbeel_lp_opt_val = mdp.Abbeel_dual_lp_mat()
    Syed_u, _ = gridworld.mdp.Syed_mat()
    consistent_u_u, _ = gridworld.mdp.Cons_u_optimal_u_mat()
    # epsilon = [0.1, 1, 10]
    consistent_r_u_l1_epsion0, _ = gridworld.mdp.Cons_r_optimal_u_l1_mat(0.0)
    consistent_r_u_l2_epsion0, _ = gridworld.mdp.Cons_r_optimal_u_l2_mat(0.0)
    consistent_r_u_l1_epsion1, _ = gridworld.mdp.Cons_r_optimal_u_l1_mat(0.5)
    consistent_r_u_l2_epsion1, _ = gridworld.mdp.Cons_r_optimal_u_l2_mat(0.5)
    consistent_r_u_l1_epsion2, _ = gridworld.mdp.Cons_r_optimal_u_l1_mat(1.0)
    consistent_r_u_l2_epsion2, _ = gridworld.mdp.Cons_r_optimal_u_l2_mat(1.0)
    consistent_r_u_l1_epsion3, _ = gridworld.mdp.Cons_r_optimal_u_l1_mat(10.0)
    consistent_r_u_l2_epsion3, _ = gridworld.mdp.Cons_r_optimal_u_l2_mat(10.0)
    # import ipdb; ipdb.set_trace()
    u_opt = gridworld.mdp.policy_to_u(opt_pol)
    opt_return = gridworld.mdp.get_return(true_rewards, u_opt)
    # Abbeel_lp_return = mdp.get_return(true_rewards, Abbeels_lp_u)
    Syed_return = gridworld.mdp.get_return(true_rewards, Syed_u)
    cons_u_return = gridworld.mdp.get_return(true_rewards, consistent_u_u)
    cons_r_l1_return_epsilon0 = gridworld.mdp.get_return(true_rewards, consistent_r_u_l1_epsion0)
    cons_r_l2_return_epsilon0 = gridworld.mdp.get_return(true_rewards, consistent_r_u_l2_epsion0)
    cons_r_l1_return_epsilon1 = gridworld.mdp.get_return(true_rewards, consistent_r_u_l1_epsion1)
    cons_r_l2_return_epsilon1 = gridworld.mdp.get_return(true_rewards, consistent_r_u_l2_epsion1)
    cons_r_l1_return_epsilon2 = gridworld.mdp.get_return(true_rewards, consistent_r_u_l1_epsion2)
    cons_r_l2_return_epsilon2 = gridworld.mdp.get_return(true_rewards, consistent_r_u_l2_epsion2)
    cons_r_l1_return_epsilon3 = gridworld.mdp.get_return(true_rewards, consistent_r_u_l1_epsion3)
    cons_r_l2_return_epsilon3 = gridworld.mdp.get_return(true_rewards, consistent_r_u_l2_epsion3)
    return_list = [opt_return, cons_u_return,  
                    cons_r_l2_return_epsilon0, cons_r_l2_return_epsilon1,
                    cons_r_l2_return_epsilon2, cons_r_l2_return_epsilon3,
                    Syed_return,
                    cons_r_l1_return_epsilon0, cons_r_l1_return_epsilon1,
                    cons_r_l1_return_epsilon2, cons_r_l1_return_epsilon3
                   ]
    return return_list

def change_epsilon(grid_size, num_experim, num_episodes, episode_len):
    parralel = True
    parralel = False
    experiment_returns =  []
    if parralel:
        start = timer()
        pool = Pool(cpu_count(), maxtasksperchild=100)
        experiment_returns = pool.starmap(epsilon_experiment, 
                                        [(grid_size, num_episodes, episode_len)
                                        for i in range(num_experim)])
        # for x in tqdm(pool.starmap(epsilon_experiment, 
        #                     [(grid_size, num_exper) for i in range(num_exper)]),
        #                     total=num_exper):
        #     experiment_returns.append(x)
        # for x in output:
        #     experiment_returns.append(x)
            # run_time.append(x[1])
        pool.close()
        # pool.join()
        print("with multiprocessing:", timer()-start) 
    else:
        for i in tqdm(range(num_experim)):
            experiment_returns.append(epsilon_experiment(grid_size, num_episodes,
                                                         episode_len))

    names = ["Optimal", "Cons U", 
             "l2 e=0", "l2 e=0.5", "l2 e=1", "l2 e=10",
             "Syed", "l1 e=0", "l1 e=.5", "l1 e=1", "l1 e=10"
             ] 
    fix_gridworld_plot(grid_size, num_experim, experiment_returns, names)

def main(argv):
    # change_num_states()
    # import ipdb; ipdb.set_trace()
    # fix_gridworld(4, 100)
    change_epsilon(15, 10, 1, 5)

if __name__ == "__main__":
    main(sys.argv[1:])
