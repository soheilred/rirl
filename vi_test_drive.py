import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
from robust_mdp import *
from gridworld import GridWorld 
import math
np.set_printoptions(precision=2, suppress=True)

def test_drive(num_rows):
# num_rows = 5 # number of rows in the square gridworld
    n = num_rows * num_rows # number of states
    r_min = -1
    r_max = 2
    gamma = 0.95 # discount factor

    gridworld = GridWorld(num_rows, r_min, r_max, gamma)

    m = 4 # number of actions
    k = r_max - 2 * r_min + 1 # number of features
# transition probability P_(m*(n*n))
    P = gridworld.get_transition_probability()
# initial state distribution p_0(n)
    p_0 = np.zeros(n)
    p_0[0] = 1

# true reward - creating reward for each action r(a,s)
    rewards = np.zeros((m,n))
    for i in range(m):
        rewards[i] = gridworld.get_reward()

# features matrix phi(m*(n*k))
    phi = gridworld.get_feature_matrix()

    mdp = MDP(n, m, k, P, phi, p_0, gamma)
# using value iteration
    optimal_policy = mdp.value_iteration(rewards)
    print("Optimal\n", optimal_policy)
    gridworld.plot_gridworld('gridworld' + str(num_rows), optimal_policy)

def test_with_one_reward_set():
    n = 2
    gamma = 0.95 # discount factor
    m = 2 # number of actions
    k = 2
# # initial state distribution p_0(n)
    p_0 = np.zeros(n)
# # features matrix phi(m*(n*k))
    phi = np.zeros([m, n, k])
# transition probability P_(m*(n*n))
    transition = np.array([[[.8, .2], [0.5, 0.5]], [[.1, .9], [0.5, .5]]])
    # rewards = np.array([[0, 1],[0, 1]])
    rewards = -.71 * np.ones([2,2])
    mdp = MDP(n, m, k, transition, phi, p_0, gamma)
    for i in range(10):
        optimal_policy = mdp.value_iteration(rewards)
        # print("Optimal\n", optimal_policy)

    # p_0[0] = 1
    # phi[0, 0] = np.array([1, 0])
    # phi[0, 1] = np.array([0, 1])
    # phi[1, 0] = np.array([1, 0])
    # phi[1, 1] = np.array([0, 1])

def main():
    # for i in range(2, 10):
        # test_drive(i)
    test_with_one_reward_set()

if __name__=="__main__":
    main()
