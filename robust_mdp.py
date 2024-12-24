#!/usr/bin/python
import pprint

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from numpy import linalg as LA
from numpy.linalg import inv
np.set_printoptions(precision=2, suppress=True)

DEBUG = 0

class MDP(object):
    """docstring for MDP"""
    def __init__(self, n, m, k, transition, phi, p_0, gamma, reward=None):
        # super(MDP, self).__init__()
        self.n = n # number of states
        self.m = m # number of actions
        self.k = k # number of features
        # building I_(m*n*n)
        self.I = np.array([np.eye(n) for i in range(m)])
        self.gamma = gamma # discount factor
        # transition probability P_(m*(n*n))
        self.transition = transition
        # initial state distribution p_0(n)
        self.p_0 = p_0
        # reward(m*n)
        if reward != None:
            self.reward = reward
        # features matrix phi(m*(n*k))
        self.phi = phi
        # occupancy frequency of an expert's policy u[m x n]
        self.u_E = None
        # feature expectation of an expert's policy mu_[k x 1]
        self.mu_E = None
        self.demonstrations = None 
        self.weights = np.zeros(k)

    def get_u_E(self, demonstrations):
        if self.u_E is None:
            self.set_u_E(demonstrations)
        return self.u_E

    def set_u_E(self, demonstrations):
        """Calculates expert's occupancy frequency given a set of trajectories.
        
        Args:
            demonstrations (n_episod * n_samples nparray): An experiment, a set of
            episodes. Episode is a single experiment defined by a T, the length
            of experiment.
        
        Returns:
            m * n nparray: Occupancy frequency.
        """
        self.demonstrations = demonstrations
        M = len(demonstrations)
        u = np.zeros([self.m, self.n])
        gamma = self.gamma
        for episode in demonstrations:
            gamma_t = 1
            # each sample has the form of [s,a,r]
            for sample in episode:
                u[int(sample[1]), int(sample[0])] += gamma_t
                gamma_t *= gamma

        u = u / (M)
        # nornalized
        # u = u / (np.sum(u) * (1 - gamma) * M)
        self.u_E = u

        mu_E = self.multi_phi(u).flatten()
        self.mu_E = mu_E

        assert (np.sum(u) - 1/(1 - self.gamma) < 10**-3), 'u^E is wrong!'


    def get_u_E_consistent(self):
        policy = np.zeros([self.m, self.n])
        for episode in self.demonstrations:
            for s, a in episode:
                policy[a, s] = 1
        for policy_s in policy.T:
            if abs(np.sum(policy_s)) < 10 ** -2:
                # policy_s[0] = 1
                # policy_s = 1/self.m * np.ones(self.m)
                policy_s[np.random.randint(0, self.m)] = 1
        u_E = self.policy_to_u(policy)
        # gamma = self.gamma
        # u_E = u_E / (np.sum(u_E) * (1 - gamma))
        # print("normalized", u)
        return u_E

    def print_solution(self, model, method=None):
        if model.status == GRB.Status.OPTIMAL:
            print('\t\t\tSolution found for', method)
            for v in model.getVars():
                print('%s %g' % (v.varName, v.x))
        else:
            print('\t\t\tNo solution for', method)

    def get_return(self, rewards, u):
        """Returns the return of a policy and a reward function.

        :param rewards: m * n numpy array
        :param u: m * n numpy array
        """
        return round(np.dot(rewards.flatten(), u.flatten()), 3)

    def get_transition(self, policy):
        """Computes the transition matrix of a given policy
        
        Args:
            policy (m * n nparray): A function from states to actions, 
        
        Returns:
            n * n nparray: Transition matrix corresponding to the policy pi
        """
        transition = self.transition
        m = self.m
#       ipdb.set_trace()
        P_pi_list = [np.multiply(transition[i].T, policy[i]).T for i in range(m)]
        P_pi = np.sum(P_pi_list, axis=0)
#       for j in range(n):
#           P_pi[j] = transition[policy[j], j,:]
        return P_pi

    def multi_phi(self, in_matrix):
        """Multiplies the input matrix to the feature matrix. The multiplication
        Phi @ in comes up a lot in the furmula. It's worth to have a funciton
        for it.

        :param in_matrix: A k * (...) matrix
        :return ( k * 1 ) matrix
        """
        # print('in_matrix dim', in_matrix.shape)
        assert(in_matrix.shape == (self.m, self.n))
        multiphied = np.transpose(self.phi,[2, 0, 1]).reshape([self.k, self.m * self.n]) @ \
                            np.reshape(in_matrix, [self.m * self.n, 1])
        return multiphied
    
    def policy_to_u(self, policy):
        n = self.n
        m = self.m
        gamma = self.gamma
        I = self.I
        assert(policy.shape == (m, n))
        # calculate u^pi(s)
        P_pi = self.get_transition(policy)
        d_pi = np.linalg.solve((np.eye(n) - gamma * P_pi.T), self.p_0)
        if not (abs(sum(d_pi) - 1/(1-gamma)) <= 0.1 ):
            import ipdb; ipdb.set_trace()
        # u = np.transpose(np.multiply(np.transpose(policy, [1, 0]),
                            # np.reshape(u_pi, [n, 1])), [1, 0])
        D = np.diag(d_pi)
        u = (D @ policy.transpose([1, 0])).transpose([1, 0])
        if DEBUG == 1:
            print('u', [round(num, 2) for num in u.flatten()])
        return u
    
    def u_to_policy(self, u):
        """
        Turns an occupancy frequency into a policy
        """
        assert(u.shape[0] == self.m)
        assert(u.shape[1] == self.n)
        assert((sum(sum(u)) - 1/(1-self.gamma)) <= 0.01 )
        policy = np.zeros([self.m, self.n])
        sum_u_s = np.sum(u, axis=0)
        for s in range(self.n):
            for a in range(self.m):
                policy[a, s] = u[a, s] / max(sum_u_s[s], 0.0000001)
        return policy

    def convert_stock_policy_determin(self, policy):
        actions = np.argmax(policy, axis=0)
        det_policy = np.zeros([self.m, self.n])
        for s in range(self.n):
            det_policy[actions[s], s] = 1
        return det_policy

    def is_row_in_array(self, mu, mu_array):
        for mu_i in mu_array:
            if np.all(np.sum(np.abs(mu_i - mu)) < 10 ** -2):
                return True
        return False

    def value_iteration(self, rewards):
        """This method solves mdps using value function.
        
        Args:
            r (ndarray, optional): Reward function.
        
        Returns:
            ndarray: Optimal policy.
        """
        P = self.transition
        gamma = self.gamma
        n = self.n
        m = self.m
        Q = np.zeros(m)
        V0 = np.zeros(n)
        V1 = np.zeros(n)
        V1[0] = 1
        policy = np.zeros([m, n])
        Q = np.zeros([m, n])
        optimal_policy = np.zeros(n, dtype=int)
        i = 0

        while (LA.norm(V0 - V1) > .001):
            V1 = np.copy(V0)
            Q = rewards + gamma * (P @ V1)
            V0 = np.max(Q, axis=0)
            # print(P @ V1)
            # print((P @ V1).flatten())
            # for s in range(n):
                # Q = rewards[:,s] + gamma * P[:,s,:] @ V1
                # V0[s] = max(Q)
                # optimal_policy[s] = int(np.argmax(Q))
            i += 1
        Q = np.around(Q, 3)
        optimal_policy = np.argmax(Q, axis=0)
        # optimal_policy = (np.isclose(Q, Q.max()))
        if DEBUG == 1:
            print('VI policy', optimal_policy)
        # print('VI took', i, 'loops')
        for state, action in enumerate(optimal_policy):
            policy[action, state] = 1 
        assert((np.sum(policy) - n) < 10**-3)
        return policy, V1

    def primal_lp(self, rewards):
        """This function solves the problem of min p0^T v in order to find the optimal value function.
        
        Returns:
            ndarray: Optimal value function
        """
        m = self.m
        n = self.n
        I = self.I
        P = self.transition
        p_0 = self.p_0
        gamma = self.gamma

        # Model
        model = gp.Model("mdp")
        v = model.addVars(n, name="v")
        # setting the return rho = p_0*v = u*r as the objective
        model.setObjective((gp.quicksum(v[i] * p_0[i] 
                            for i in range(n))),
                            GRB.MINIMIZE)

        # Constraints
        model.addConstrs((  gp.quicksum((I[i] - gamma*P[i])[j,k] * v[k]
                            for k in range(n))
                            >= rewards[i,j]
                            for j in range(n)
                            for i in range(m) ),
                            name="C0")

        # Checking the correctness of constraints
        model.write("../../files/primal_regular_mdp.lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        value = []
        v = model.getVars()
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                value.append(v.x)
            # print(u)
        else:
            print("No solution for primal lp")
        return np.array(value)

    def dual_lp(self, rewards):
        """ This method solves the problem of Bellman Flow Constraint. This 
        problem is sometimes called the dual problem of min p0^T v, which finds
        the optimal value function.
        
        Returns:
            ndarray: The optimal policy
        """
        method = 'LP'
        m = self.m
        n = self.n
        I = self.I
        P = self.transition
        p_0 = self.p_0
        gamma = self.gamma
        flat_r = rewards.reshape([m * n, 1])
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T

        # Model
        model = gp.Model("mdp")
        var_dim = (m * n) 
        l_bound = np.zeros(var_dim)
        u = model.addMVar(shape=var_dim, name="u", lb=l_bound)
        # setting the objective 
        model.setObjective(flat_r.T @ u , GRB.MAXIMIZE)
        # Constraints
        A0 = flat_A_T
        b0 = p_0.reshape([n, 1])
        model.addMConstr(A0, u, '=', b0, name="C0")

        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        # Checking the correctness of constraints
        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        p1 = model.objVal
        u_flat = self.gurobi_to_numpy(model, m * n)
        u_mat = u_flat.reshape([m,n])
        assert((np.sum(u_mat) - 1/(1 - gamma) < 10**-3))
        return u_mat, p1

    def min_u(self, rewards):
        """ This method solves the problem of Bellman Flow Constraint. This 
        problem is sometimes called the dual problem of min p0^T v, which finds
        the optimal value function.
        
        Returns:
            ndarray: The optimal policy
        """
        method = 'Min_u'
        m = self.m
        n = self.n
        P = self.transition
        p_0 = self.p_0
        gamma = self.gamma
        I = self.I
        # flat_r = rewards.reshape([m * n, 1])
        flat_r = rewards.flatten()
        flat_A = (I - gamma * P).reshape([m * n, n]).T
        # flat_A = np.hstack([(np.eye(n) - gamma * P[a].T) for a in range(m)])

        # Model
        model = gp.Model("mdp")
        var_dim = (m * n) 
        l_bound = np.zeros(var_dim)
        u = model.addMVar(shape=var_dim, name="u", lb=l_bound)
        # setting the objective 
        model.setObjective(flat_r @ u, GRB.MINIMIZE)
        # Constraints
        A0 = flat_A
        b0 = p_0#.reshape([n, 1])
        # model.addMConstr(A0, u, '=', b0, name="C0")
        model.addConstr(A0 @ u == b0, name="C0")

        # Checking the correctness of constraints
        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        u_flat = self.gurobi_to_numpy(model, m * n, method)
        u_mat = u_flat.reshape([m,n])
        assert(abs(np.sum(u_mat) - 1/(1 - gamma)) < 10**-3)
        if model.status != GRB.Status.OPTIMAL:
            print(abs(np.sum(u_mat) - 1/(1 - gamma)))
            import ipdb;ipdb.set_trace()
        # assert(abs(np.sum(flat_A.T @ u_flat - p_0)) < 10**-3)
        return u_mat

    def closest_consistent_u(self, u_a):
        """Calculates the closest u to u_a that is consistent with A^T u = p_0
        Args:
            u_a (m x n): the input u
        
        Returns:
            u (m x n): the projected u in the A^T u = p_0 plane
        """
        method = "closest u" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , method, '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        phi = self.phi
        I = self.I
        P = self.transition
        gamma = self.gamma
        p_0 = self.p_0
        u_E = self.u_E
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        u_a = u_a.reshape([m * n])

        # Model
        model = gp.Model("matrix")
        # x = [u]
        var_dim = (m * n)
        l_bound = np.zeros(var_dim)
        u = model.addMVar(shape=var_dim, name="u", lb=l_bound)

        # setting the objective 
        # obj = np.hstack([np.zeros(m * n + m *n), epsilon, 1])
        model.setObjective( u @ (u - 2 * u_a), GRB.MINIMIZE)
        # Constraints
        # A^T u = p0
        A2 = flat_A_T
        b2 = p_0.reshape([n, 1])
        model.addMConstr(A2,  u ,'=', b2, name="C0")
        # # import ipdb; ipdb.set_trace()
        model.Params.OutputFlag = 0
        # Solve
        model.optimize()
        model.write("../../files/" + method + ".lp")
        # Checking the correctness of constraints
        # presolve = model.presolve()
        # model.computeIIS()
        # presolve.optimize()
        # presolve.write("../../files/presolve.lp")
        # print("slack", model.slack)
        u_vec = np.zeros((m,n))
        v = model.getVars()
        i = 0
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                if (i < m*n):
                    x0 = int(i/n)
                    y0 = int(i%n)
                    u_vec[x0,y0] = v.x
                    i = i + 1
            opt_val = model.objVal
            assert((np.sum(u_vec) - 1/(1 - gamma) < 10**-2))
            if DEBUG == 1:
                # print(x.X)
                self.print_solution(model, method)
        else:
            print("No solution for " + method)
            opt_val = None
        # print(model.x)
        return u_vec

    def gurobi_to_numpy(self, model, count, method=""):
        """Splits the gurobi variables and returns them in a form of ndarrays
        
        Args:
            model (gurobi object): Gurobi model object corresponds to Abbeel's method
        
        Returns:
            int, ndarray: t, w
        """
        w = np.zeros(count)
        if model.status == GRB.Status.OPTIMAL:
            v = model.getVars()
            for i in range(count):
                w[i] = v[i].x
        else:
            print('No solution for ' + method)
            w[0] = 1
        return w

    def Abbeel_policy_selector(self, mu_array):
        mu_E = self.mu_E
        k_num = self.k
        pol_num = mu_array.shape[0]
        model = gp.Model("QP")
        delta = model.addVars(pol_num, name='delta', lb=0)
        model.setObjective(gp.quicksum((mu_E[i] / pol_num - delta[j] *
                            mu_array[j, i]) * (mu_E[i] / pol_num - delta[j] *
                            mu_array[j, i])
                            for i in range(k_num)
                            for j in range(pol_num)), 
                            GRB.MINIMIZE)
        model.addConstr(gp.quicksum(delta[j] for j in range(pol_num)) == 1,
                            name="C0")
        model.write("../../files/Abbeel-policy-selector.lp")
        model.Params.OutputFlag = 0
        model.optimize()
        t1 = model.objVal
        delta_vec = self.gurobi_to_numpy(model, pol_num)
        return delta_vec

    def Abbeel_dual_vi(self):
        """Abbeels method, based on the dual formulation presented in page 3,
         equations 10-12. Abbeel's method solves an IRL problem, where 
         <S, A, P, gamma, D, phi> are assumed given. We iterate through all 
         available policies and calculate t_i = max_w min_j  w^T(mu(E) -
         mu(pi_j)).  We stop when t_i < eps.
        
        :returns: optimal policy, rewards

        """
        method = "Abbeel dual vi"
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + method + "="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        # P = self.transition
        # features matrix phi(m*(n*k))
        phi = self.phi
        gamma = self.gamma
        u = np.zeros((m, n))
        u[0,:] = np.ones(n)
        u_array = np.array([u])
        mu = self.multi_phi(u).flatten()
        mu_array = np.array([mu])
        u_E = self.u_E
        mu_E = self.multi_phi(u_E).flatten()
        rewards = np.zeros((m,n))
        i = 0

        # for i in range(1, num_iter):
        while(True):
            i += 1
            model = gp.Model("mdp")
            w = model.addVars(k_num, name="w", lb=-1)
            t = model.addVar(name="t", lb=-GRB.INFINITY)
            model.setObjective(t, GRB.MAXIMIZE)
            model.addConstrs((t <= gp.quicksum(w[k] * (mu_E[k] - mu_array[j][k]) 
                                for k in range(k_num))
                                for j in range(mu_array.shape[0]) ),
                                name="C0")
            model.addConstr((   gp.quicksum(w[k] * w[k]
                                for k in range(k_num)) <= 1),
                                name="C1")

            model.write("../../files/" + method + ".lp")
            model.Params.OutputFlag = 0
            model.optimize()
            t1 = model.objVal
            w_vec = self.gurobi_to_numpy(model, k_num)
            rewards = phi @ w_vec
            if DEBUG == 1:
                print("=" * 10, i, "=" * 10)
                print('opt_val', round(t1, 3))
                print("u\n", u)
                print('mu^pi', [round(num, 2) for num in mu.tolist()] )
                print('mu_E-mu^pi:', [round(num, 2) for num in 
                                            (mu_E - mu).tolist()])
                print('w', w_vec)
                print('r_', rewards.shape, [round(num, 2) for num in 
                                            rewards.flatten()])

            policy,_ = self.value_iteration(rewards)
            u = self.policy_to_u(policy) 
            mu = self.multi_phi(u).flatten()
            assert(mu.shape == mu_E.shape)
            assert(np.sum(policy, axis=0).shape == (n,))
            assert(np.sum(policy) == n)
            assert((np.sum(u) - 1/(1 - gamma) < 10**-3))

            if self.is_row_in_array(mu, mu_array) != True:
                mu_array = np.append(mu_array, [mu], axis=0) 
                u_array = np.append(u_array, [u], axis=0)
            else:
                break
        delta_vec = self.Abbeel_policy_selector(mu_array[1:,:])
        # print('delta', delta_vec)
        mixed_u = np.transpose(u_array[1:,:,:],[1,2,0]) @ delta_vec 
        return mixed_u, rewards, t1

    def Abbeel_dual_lp(self):
        """Abbeels method, based on the dual formulation presented in page 3,
         equations 10-12. Abbeel's method solves an IRL problem, where 
         <S, A, P, gamma, D, phi> are assumed given. We iterate through all 
         available policies and calculate t_i = max_w min_j  w^T(mu(E) - mu(pi_j)). 
         We stop when t_i < eps.
        
        :returns: optimal policy, rewards

        """
        method = "Abbeel dual LP"
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + method + "="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        # features matrix phi(m*(n*k))
        phi = self.phi
        u = np.zeros((m, n))
        u[0,:] = np.ones(n)
        u_array = np.array([u])
        mu = self.multi_phi(u).flatten()
        mu_array = np.array([mu])
        u_E = self.u_E
        mu_E = self.multi_phi(u_E).flatten()
        if DEBUG == 1:
            print('mu_E:', mu_E)
        rewards = np.zeros((m,n))
        t1 = 1
        i = 0

        while(True):
            i += 1
            model = gp.Model("mdp")
            w = model.addVars(k_num, name="w", lb=-1)
            t = model.addVar(name="t", lb=-GRB.INFINITY)
            model.setObjective(t, GRB.MAXIMIZE)
            model.addConstrs((t <= gp.quicksum(w[k] * (mu_E[k] - mu_array[j][k]) 
                                for k in range(k_num))
                                for j in range(mu_array.shape[0]) ),
                                name="C0")
            model.addConstr((   gp.quicksum(w[k] * w[k]
                                for k in range(k_num)) <= 1),
                                name="C1")

            model.write("../../files/" + method + ".lp")
            model.Params.OutputFlag = 0
            model.optimize()
            t1 = model.objVal
            w_vec = self.gurobi_to_numpy(model, k_num)
            rewards = phi @ w_vec
            if DEBUG == 1:
                print("=" * 10, i, "=" * 10)
                print('opt_val', round(t1, 3))
                print("u\n", u)
                print('mu', [round(num, 2) for num in mu.tolist()] )
                print('mu_E-mu^pi:', [round(num, 2) for num in (mu_E - mu).tolist()])
                print('w', w_vec)
                print('r_', rewards.shape, [round(num, 2) for num in rewards[0]])

            u,_ = self.dual_lp(rewards)
            mu = self.multi_phi(u).flatten()
            # if mu not in mu_array:
            if self.is_row_in_array(mu, mu_array) != True:
                mu_array = np.append(mu_array, [mu], axis=0) 
                u_array = np.append(u_array, [u], axis=0) 
            else:
                break
        # print("FINAL u\n", u)
        delta_vec = self.Abbeel_policy_selector(mu_array[1:,:])
        mixed_u = np.transpose(u_array[1:,:,:],[1,2,0]) @ delta_vec 
        return mixed_u, rewards, t1

    def Abbeel_dual_lp_mat(self):
        """Abbeels method, based on the dual formulation presented in page 3,
         equations 10-12. Abbeel's method solves an IRL problem, where 
         <S, A, P, gamma, D, phi> are assumed given. We iterate through all 
         available policies and calculate t_i = max_w min_j  w^T(mu(E) - mu(pi_j)). 
         We stop when t_i < eps.
        
        :returns: optimal policy, rewards

        """
        method = "Abbeel dual LP mat"
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + method + "="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        # features matrix phi(m*(n*k))
        phi = self.phi
        u = np.zeros((m, n))
        u[0,:] = np.ones(n)
        u_array = np.array([u])
        mu = self.multi_phi(u).flatten()
        mu_array = np.array([mu])
        u_E = self.u_E
        flat_phi = phi.reshape([m * n, k_num]).T
        flat_u_E = u_E.reshape(m * n)
        mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])
        mu_E_array = np.array(mu_E.T)
        if DEBUG == 1:
            print('mu_E:', mu_E)
        rewards = np.zeros((m,n))
        t1 = 1
        i = 0

        while(True):
            i += 1
            # Model
            model = gp.Model("mdp")
            var_dim = k_num + 1
            x = model.addMVar(shape=var_dim, name="x", lb=-GRB.INFINITY)
            # setting the objective 
            c = np.hstack([np.zeros(k_num), 1])
            model.setObjective(c @ x, GRB.MAXIMIZE)
            # Constraints
            # import ipdb; ipdb.set_trace()
            A0 = np.hstack([mu_array - mu_E_array, np.ones([i, 1])])
            b0 = np.zeros([i,1])
            model.addMConstr(A0, x, '<=', b0, name="C0")
            # [ w t ] [1_k*k  0_k*1] [w]
            #         [0_1*k  0_1*1] [t]
            A1 = np.hstack([np.vstack([np.eye(k_num), np.zeros([1,k_num])]),
                            np.zeros([k_num + 1, 1])])
            b1 = 1
            # import ipdb; ipdb.set_trace()
            model.addQConstr(x @ A1 @ x <= b1, name="C1")
            # model.setObjective(t, GRB.MAXIMIZE)

            model.write("../../files/" + method + ".lp")
            model.Params.OutputFlag = 0
            model.optimize()
            t1 = model.objVal
            w_vec = self.gurobi_to_numpy(model, k_num)
            # rewards = phi @ w_vec
            rewards = (w_vec.T @ flat_phi).reshape([m, n])
            if DEBUG == 1:
                print("=" * 10, i, "=" * 10)
                print('opt_val', round(t1, 3))
                print("u\n", u)
                print('mu', [round(num, 2) for num in mu.tolist()] )
                print('mu_E-mu^pi:', [round(num, 2) for num in (mu_E - mu).tolist()])
                print('w', w_vec)
                print('r_', rewards.shape, [round(num, 2) for num in rewards[0]])

            u,_ = self.dual_lp(rewards)
            mu = self.multi_phi(u).flatten()
            # if mu not in mu_array:
            if self.is_row_in_array(mu, mu_array) != True:
                mu_array = np.append(mu_array, [mu], axis=0) 
                mu_E_array = np.append(mu_E_array, mu_E.T, axis=0) 
                u_array = np.append(u_array, [u], axis=0) 
            else:
                break
        # print("FINAL u\n", u)
        delta_vec = self.Abbeel_policy_selector(mu_array[1:,:])
        mixed_u = np.transpose(u_array[1:,:,:],[1,2,0]) @ delta_vec 
        return mixed_u, rewards, t1
    
    def Abbeel_prob_simplex(self):
        """This is an implementation is based on probability simplex
        formulation, driven by Marek in one of the meetings. 
        Args:
            episodes (2d array): An array of trajectories. A trajectory consists
            of consequence state-action pairs.
        
        Returns:
            policy: The optimal policy obtained by the reward r = w * phi
        """
        n = self.n
        m = self.m
        k_num = self.k
        # features matrix phi(m*(n*k))
        phi = self.phi
        u = np.zeros((m, n))
        u[0,:] = np.ones(n)
        u_array = np.array([u])
        mu = self.multi_phi(u).flatten()
        mu_array = np.array([mu])
        u_E = self.u_E
        mu_E = self.multi_phi(u_E).flatten()
        rewards = np.zeros((m,n))
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + "Abbeel Probabilty Simplex" + "="*20)
            print("/\\"*32)
        t1 = 1
        i = 0

        while(True):
            i += 1
            # Model
            model = gp.Model("mdp")
            delta = model.addVars(mu_array.shape[0], name="alpha", lb=0)
            # Objective
            model.setObjective( gp.quicksum(((mu_E[p] - mu_array[j][p]) * 
                                (mu_E[p] - mu_array[k][p])) * (delta[j] * delta[k])
                                for p in range(k_num)
                                for k in range(mu_array.shape[0])
                                for j in range(mu_array.shape[0])),
                                GRB.MINIMIZE)
            # Constraints  
            model.addConstr((   gp.quicksum(delta[k] 
                                for k in range(mu_array.shape[0])) == 1),
                                name="C1")

            model.write("../../files/Abbeel-prob-simplex.lp")
            # Solve
            model.Params.OutputFlag = 0
            model.optimize()
            t1 = model.objVal
            alpha_vec = self.gurobi_to_numpy(model, mu_array.shape[0])

            model_inner = gp.Model("mdp_inner")
            w = model_inner.addVars(k_num, name="w", lb=-GRB.INFINITY)
            # Objective
            model_inner.setObjective( gp.quicksum(w[k] * (mu_E[k] - mu_array[j][k]) * 
                                alpha_vec[j]
                                for k in range(k_num) 
                                for j in range(mu_array.shape[0])),
                                GRB.MAXIMIZE)
            # Constraints  
            model_inner.addConstr(( gp.quicksum(w[k] * w[k]
                                for k in range(k_num)) <= 1),
                                name="C11")
            model_inner.write("../../files/Abbeel-inner.lp")
            model_inner.Params.OutputFlag = 0
            model_inner.optimize()
            w_vec = self.gurobi_to_numpy(model_inner, k_num)
            rewards = phi @ w_vec
            if DEBUG == 1:
                print("=" * 10, i, "=" * 10)
                print('opt val', round(t1, 3))
                print('alpha', alpha_vec)
                print('w', w_vec)
                print('r_', rewards.shape, [round(num, 2) for num in rewards.flatten()])
                print("u\n", u)
                print('mu', [round(num, 2) for num in mu.tolist()] )

            u,_ = self.dual_lp(rewards)
            mu = self.multi_phi(u).flatten()
            # if mu not in mu_array:
            if self.is_row_in_array(mu, mu_array) != True:
                mu_array = np.append(mu_array, [mu], axis=0) 
                u_array = np.append(u_array, [u], axis=0) 
            else:
                break
        mixed_u = np.transpose(u_array[1:,:,:],[1,2,0]) @ alpha_vec[1:]
        return mixed_u, rewards, t1

    def Abbeel_consistent_ru(self):
        if DEBUG == 0:
            print("/\\"*32)
            print("="*20 + "Abbeel RU" + "="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        epsilon = 0.1
        # features matrix phi(m*(n*k))
        phi = self.phi
        p_0 = self.p_0
        I = self.I
        gamma = self.gamma
        P = self.transition
        u = np.zeros((m, n))
        u[0,:] = np.ones(n)
        u_array = np.array([u])
        mu = self.multi_phi(u).flatten()
        mu_array = np.array([mu])
        u_E = self.u_E
        mu_E = self.multi_phi(u_E).flatten()
        if DEBUG == 1:
            print('mu_E:', mu_E)
        rewards = np.zeros((m,n))
        t1 = 1
        i = 0

        while(True):
            i += 1
            model = gp.Model("mdp")
            w = model.addVars(k_num, name="w", lb=-GRB.INFINITY)
            v = model.addVars(n, name="v", lb=-GRB.INFINITY)
            t = model.addVar(name="t", lb=-GRB.INFINITY)
            model.setObjective(t, GRB.MAXIMIZE)
            model.addConstrs((  t <= gp.quicksum(w[k] * (mu_E[k] - mu_array[j][k]) 
                                for k in range(k_num))
                                for j in range(mu_array.shape[0]) ),
                                name="C0")
            model.addConstr((   gp.quicksum(w[k] * w[k]
                                for k in range(k_num)) <= 1),
                                name="C1")
            model.addConstr((   gp.quicksum( - w[k] * phi[j, i, k] * u[j, i]
                                for k in range(k_num)
                                for j in range(m)
                                for i in range(n)) - epsilon + 
                                gp.quicksum(p_0[i] * v[i]
                                for i in range(n)) <= 0),
                                name="C2")
            model.addConstrs((  gp.quicksum(phi[j, i, k] * w[k]
                                for k in range(k_num) ) - 
                                gp.quicksum((I[j] - gamma * P[j])[i,l] * v[l]
                                for l in range(n))
                                <= 0 
                                for j in range(m) 
                                for i in range(n)),
                                name="C3")

            model.write("../../files/Abbeel-dual.lp")
            model.Params.OutputFlag = 0
            model.optimize()
            t1 = model.objVal
            w_vec = self.gurobi_to_numpy(model, k_num)
            rewards = phi @ w_vec
            if DEBUG == 0:
                print("=" * 10, i, "=" * 10)
                print('opt_val', round(t1, 3))
                print("u\n", u)
                print('mu', [round(num, 2) for num in mu.tolist()] )
                print('mu_E-mu^pi:', [round(num, 2) for num in (mu_E - mu).tolist()])
                print('w', w_vec)
                print('r_', rewards.shape, [round(num, 2) for num in rewards.flatten()])

            u,_ = self.dual_lp(rewards)
            mu = self.multi_phi(u).flatten()
            # if mu not in mu_array:
            if self.is_row_in_array(mu, mu_array) != True:
                mu_array = np.append(mu_array, [mu], axis=0) 
                u_array = np.append(u_array, [u], axis=0) 
            else:
                break
        # print("FINAL u\n", u)
        delta_vec = self.Abbeel_policy_selector(mu_array[1:,:])
        mixed_u = np.transpose(u_array[1:,:,:],[1,2,0]) @ delta_vec 
        return mixed_u, rewards, t1

    def Syed(self):
        """This method is based on Syed's LP based apprenticeship learning paper.
        
        Args:
            episodes (ndarray): An experiment, consist of a number of episods of runs
        
        Returns:
            ndarray: Optimal policy
        """
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + "Syed" + "="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        gamma = self.gamma
        I = self.I
        P = self.transition
        p_0 = self.p_0
        # features matrix phi(m*(n*k))
        phi = self.phi
        u_E = self.u_E
        mu_E = self.multi_phi(u_E)

        # Model
        model = gp.Model("mdp")
        u = model.addVars(m, n, name="u", lb=0.0)
        B = model.addVar(name="B", lb= - GRB.INFINITY)
        # The objective funciton
        model.setObjective(B, GRB.MAXIMIZE)
        # Constraints

        model.addConstrs((  B <= (gp.quicksum(phi[j, i, k] * u[j,i]
                            for j in range(m)
                            for i in range(n))) - mu_E[k]
                            for k in range(k_num) ), 
                            name="C0")
        model.addConstrs((  gp.quicksum((I[j] - gamma*P[j])[k,i] * u[j,k]
                            for k in range(n)
                            for j in range(m) )
                            == p_0[i]
                            for i in range(n) ),
                            name="C1")

        # Checking the correctness of constraints
        model.write("../../files/Syed.lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        if DEBUG == 1:
            self.print_solution(model, 'Syed')
        occ_freq = np.zeros((m,n))
        v = model.getVars()
        i = 0

        # if model.SolCount > 0:
        if model.status == GRB.Status.OPTIMAL:
            opt_val = model.objVal
            for v in model.getVars():
                if (i < m*n):
                    x = int(i/n)
                    y = int(i%n)
                    occ_freq[x,y] = v.x
                    i = i + 1
            # print(u)
        else:
            print("No solution for ", method)
        return occ_freq, -opt_val

    def Syed_mat(self):
        """This method is based on Syed's LP based apprenticeship learning
        paper.
        
        Args:
            episodes (ndarray): An experiment, consist of a number of episods of
            runs
        
        Returns:
            ndarray: Optimal policy
        """
        method = "Syed mat"
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + method + "="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        gamma = self.gamma
        I = self.I
        P = self.transition
        p_0 = self.p_0
        # features matrix phi(m*(n*k))
        phi = self.phi
        u_E = self.u_E
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T
        flat_u_E = u_E.reshape(m * n)
        mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])

        # Model
        model = gp.Model("mdp")
        var_dim = (m * n) + 1
        l_bound = np.hstack([np.zeros(var_dim - 1), -GRB.INFINITY])
        x = model.addMVar(shape=var_dim, name="x", lb=l_bound)
        # setting the objective 
        obj = np.hstack([np.zeros(m * n), 1])
        model.setObjective( obj @ x, GRB.MAXIMIZE)
        # Constraints
        A0 = np.hstack( [-flat_phi, np.ones([k_num, 1])] )
        b0 = -mu_E
        model.addMConstr(A0 , x, '<=', b0, name="C0") 
        A1 = np.hstack([flat_A_T, np.zeros([n, 1])])
        b1 = p_0.reshape([n, 1])
        model.addMConstr(A1, x, '=', b1, name="C1")

        # Checking the correctness of constraints
        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        if DEBUG == 1:
            self.print_solution(model, method)
        occ_freq = np.zeros((m,n))
        opt_val = -999
        v = model.getVars()
        i = 0

        # if model.SolCount > 0:
        if model.status == GRB.Status.OPTIMAL:
            opt_val = model.objVal
            for v in model.getVars():
                if (i < m*n):
                    x = int(i/n)
                    y = int(i%n)
                    occ_freq[x,y] = v.x
                    i = i + 1
            # print(u)
        else:
            print("No solution for " + method)
        return occ_freq, opt_val

    def Huang_l1(self, epsilon):
        """this function based on our method.
        Args:
            episodes (ndarray): A whole experiment, consist of a number of episodes of run
        
        Returns:
            ndarray: Optimal policy
        """
        method = "Huang-l1" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , method, '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        phi = self.phi
        # epsilon = 10.1
        I = self.I
        P = self.transition
        gamma = self.gamma
        p_0 = self.p_0
        u_E = self.u_E
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T
        flat_u_E = u_E.reshape(m * n)
        # mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])
        mu_E = np.zeros([k_num, 1])

        # Model
        model = gp.Model("matrix")
        # x = [u, lambda_1, lambda_2, t]
        var_dim = (m * n) + (m * n) + 1 + 1
        l_bound = np.hstack([np.zeros(var_dim - 1), -GRB.INFINITY])
        x = model.addMVar(shape=var_dim, name="x", lb=l_bound)
        # setting the objective 
        obj = np.hstack([np.zeros(m * n + m *n), epsilon, 1])
        model.setObjective( obj @ x, GRB.MINIMIZE)
        # Constraints
        # t >= phi(u_E - u + lambda_1 - lambda_2 * u_E)[i]
        # Phi u  -  Phi lambda_1  +  lambda_2 mu_E  +  t 1_(kx1)  >=  mu_E
        A0 = np.hstack([flat_phi, -flat_phi, mu_E, np.ones([k_num, 1])])
        b0 = mu_E
        model.addMConstr(A0 , x, '>=', b0, name="C0") 
        # A^T lambda_1 = lambda_2 p0
        A1 = np.hstack([np.zeros([n, m * n]), flat_A_T, -p_0.reshape([n,1]),
                        np.zeros([n, 1])])
        b1 = np.zeros(n)
        model.addMConstr(A1, x, '=', b1, name="C1")
        # A^T u = p0
        A2 = np.hstack([flat_A_T, np.zeros([n, m * n]), np.zeros([n, 1]), 
                        np.zeros([n, 1])])
        b2 = p_0.reshape([n, 1])
        model.addMConstr(A2,  x ,'=', b2, name="C2")
        # A3 = np.hstack([np.zeros([n * m, n * m]), np.ones([n * m, n * m]),
        #                            np.zeros([n * m, 2])])
        # b3 = np.zeros(n * m)
        # model.addMConstr(A3,  x ,'>', b3, name="C3")
        # A4 = np.hstack([np.zeros([2, 2 * n * m]), np.ones([2, 1]),
        #                 np.zeros([2, 1])])
        # b4 = np.zeros(2)
        # # import ipdb; ipdb.set_trace()
        # model.addMConstr(A4,  x ,'>', b4, name="C4")
        model.Params.OutputFlag = 0
        # Solve
        model.optimize()
        model.write("../../files/" + method + ".lp")
        # Checking the correctness of constraints
        # presolve = model.presolve()
        # model.computeIIS()
        # presolve.optimize()
        # presolve.write("../../files/presolve.lp")
        # print("slack", model.slack)
        u_flat = self.gurobi_to_numpy(model, m * n, method)
        u_vec = u_flat.reshape([m, n])
        # u_vec = np.zeros((m,n))
        # v = model.getVars()
        # i = 0
        # if model.status == GRB.Status.OPTIMAL:
        #     for v in model.getVars():
        #         if (i < m*n):
        #             x0 = int(i/n)
        #             y0 = int(i%n)
        #             u_vec[x0,y0] = v.x
        #             i = i + 1
        #     opt_val = model.objVal
        #     assert((np.sum(u_vec) - 1/(1 - gamma) < 10**-2))
        #     if DEBUG == 1:
        #         # print(x.X)
        #         self.print_solution(model, method)
        # else:
        #     print("No solution for " + method)
        #     opt_val = None
        # print(model.x)
        return u_vec

    def Huang_l2(self, epsilon):
        """this function based on our method.
        Args:
            episodes (ndarray): A whole experiment, consist of a number of 
            episodes of run
        
        Returns:
            ndarray: Optimal policy
        """
        method = "Consistent R l2 mat" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , method, '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        phi = self.phi
        # epsilon = 0.41
        I = self.I
        P = self.transition
        p_0 = self.p_0
        gamma = self.gamma
        # u_E = self.u_E
        u_E = np.zeros([m, n])
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T
        flat_u_E = u_E.reshape(m * n)
        # flat_u_E = np.zeros(m * n)
        # mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])

        # Model
        model = gp.Model("matrix")
        # x = [u, lambda_1, lambda_2]
        var_dim = (m * n) + (m * n) + 1
        l_bound = np.hstack([np.zeros(var_dim)])
        x = model.addMVar(shape=var_dim, name="x", lb=l_bound)
        # setting the objective 
        Lambda = flat_phi.T @ flat_phi 
        Z = np.hstack([-np.eye(m * n), np.eye(m * n), - u_E.reshape([m * n, 1])])
        Q = Z.T @ Lambda @ Z
        E = np.hstack([np.zeros(2 * m * n), epsilon])
        B = E + 2 * flat_u_E.T @ Lambda @ Z
        c = flat_u_E.T @ Lambda @ flat_u_E 
        # import ipdb; ipdb.set_trace()
        model.setMObjective(Q, B, c, None, None, None, GRB.MINIMIZE)
        # B = np.hstack([-flat_phi, flat_phi, -mu_E])
        # Q = B.T @ B
        # c = mu_E
        # eps_vec = np.hstack([np.zeros(var_dim - 1), epsilon])
        # model.setMObjective(Q, 2 * ((c.T @ B).flatten() + eps_vec), 
        #                     (c.T @ c).flatten(),
        #                     None, None, None,
        #                     GRB.MINIMIZE)
        # Constraints
        # A^T lambda_1 = lambda_2 p0
        A1 = np.hstack([np.zeros([n, m * n]), flat_A_T, -p_0.reshape([n,1])])
        b1 = np.zeros(n)
        model.addMConstr(A1, x, '=', b1, name="C1")
        # A^T u = p0
        A2 = np.hstack([flat_A_T, np.zeros([n, m * n]), np.zeros([n, 1])])
        b2 = p_0.reshape([n, 1])
        model.addMConstr(A2,  x ,'=', b2, name="C2")

        # Checking the correctness of constraints
        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        # model.printStats()
        if DEBUG == 1:
            self.print_solution(model, method)
        # model.computeIIS()
        # turn the gurobi variable to numpy
        u_flat = self.gurobi_to_numpy(model, m * n, method)
        u_vec = u_flat.reshape([m, n])
        # u_vec = np.zeros((m,n))
        # v = model.getVars()
        # i = 0
        # if model.status == GRB.Status.OPTIMAL:
        #     for v in model.getVars():
        #         if (i < m*n):
        #             x = int(i/n)
        #             y = int(i%n)
        #             u_vec[x,y] = v.x
        #             i = i + 1
        #     opt_val = model.objVal
        #     assert((np.sum(u_vec) - 1/(1 - gamma) < 10**-2))
        #     # print("slack", model.slack)
        # else:
        #     print("No solution for " + method)
        #     opt_val = None
        # print(model.x)
        return u_vec

    def Cons_u_optimal_u(self):
        """Abbeel's method with consistent set u (A^T * u = p_0).
        Returns: TODO

        """
        method =  "consistent U" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + method +"="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        gamma = self.gamma
        P = self.transition
        p_0 = self.p_0
        I = self.I
        # features matrix phi(m*(n*k))
        phi = self.phi
        u_E = self.u_E
        # f_phi = phi.reshape([m * n, k_num])

        model = gp.Model("mdp")
        u = model.addVars(m, n, name="u", lb=0.0)
        model.setObjective(gp.quicksum((u[i, j] - u_E[i, j]) * phi[i, j, k] *
                            phi[h, l, k] * (u[h, l] - u_E[h, l])
                            for i in range(m)
                            for h in range(m)
                            for k in range(k_num)
                            for l in range(n)
                            for j in range(n)),
                            GRB.MINIMIZE)
        model.addConstrs((  gp.quicksum((I[j] - gamma*P[j])[k,i] * u[j,k]
                            for k in range(n)
                            for j in range(m) )
                            == p_0[i]
                            for i in range(n) ),
                            name="C2")

        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        t = model.objVal

        if DEBUG == 1:
            self.print_solution(model, 'Abbeels consistent')
            print("opt_val", t)
        u_vec = np.zeros((m,n))
        v = model.getVars()
        i = 0
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                if (i < m*n):
                    x = int(i/n)
                    y = int(i%n)
                    u_vec[x,y] = v.x
                    i = i + 1
            opt_val = model.objVal
            assert((np.sum(u_vec) - 1/(1 - gamma) < 10**-3))
        else:
            print("No solution " + method)
            opt_val = None
        return u_vec, opt_val

    def Cons_u_optimal_u_mat(self):
        """Abbeel's method with consistent set u (A^T * u = p_0).
        Returns: TODO

        """
        method =  "consistent U mat" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + method +"="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        gamma = self.gamma
        P = self.transition
        p_0 = self.p_0
        I = self.I
        # features matrix phi(m*(n*k))
        phi = self.phi
        u_E = self.u_E
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T
        flat_u_E = u_E.reshape(m * n)
        mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])
        # Model
        model = gp.Model("mdp")
        var_dim = (m * n) 
        l_bound = np.zeros(var_dim)
        x = model.addMVar(shape=var_dim, name="x", lb=l_bound)
        # setting the objective 
        Q = flat_phi.T @ flat_phi
        c = flat_u_E.T @ Q @ flat_u_E
        b = -2 * flat_u_E.T @ Q
        model.setMObjective(Q, b, c, None, None, None, GRB.MINIMIZE)
        # Constraints
        A0 = flat_A_T
        b0 = p_0.reshape([n, 1])
        model.addMConstr(A0, x, '=', b0, name="C0")

        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()

        u_vec = np.zeros((m,n))
        v = model.getVars()
        i = 0
        if model.status == GRB.Status.OPTIMAL:
            t = model.objVal
            if DEBUG == 1:
                self.print_solution(model, 'Abbeels consistent')
                print("opt_val", t)
            for v in model.getVars():
                if (i < m*n):
                    x = int(i/n)
                    y = int(i%n)
                    u_vec[x,y] = v.x
                    i = i + 1
            opt_val = model.objVal
            assert((np.sum(u_vec) - 1/(1 - gamma) < 10 ** -2))
        else:
            print("No solution for " + method)
            opt_val = None
        return u_vec, opt_val

    def Cons_r_optimal_u_l1(self):
        """this function based on our method.
        Args:
            episodes (ndarray): A whole experiment, consist of a number of episodes of run
        
        Returns:
            ndarray: Optimal policy
        """
        method = "Consistent R" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , method, '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        phi = self.phi
        epsilon = 149
        I = self.I
        P = self.transition
        p_0 = self.p_0
        gamma = self.gamma
        u_E = self.u_E

        # Model
        model = gp.Model("mdp")
        u = model.addVars(m,n, name="u", lb=0)
        lambda_1 = model.addVars(m, n, name="lambda_1", lb=0)
        lambda_2 = model.addVar(name="lambda_2", lb=0)
        t = model.addVar(name="t", lb=-GRB.INFINITY)
        # setting the objective 
        model.setObjective(t + lambda_2 * epsilon, GRB.MINIMIZE)
        # Constraints
        model.addConstrs((t >= gp.quicksum(phi[i, j, k] * ((1 + lambda_2) * 
                            u_E[i, j] - u[i, j] - lambda_1[i,j])
                            for i in range(m)
                            for j in range(n))
                            for k in range(k_num)),
                            name="C0")
        model.addConstrs((gp.quicksum((I[j] - gamma*P[j])[k,i] * lambda_1[j,k]
                            for k in range(n)
                            for j in range(m))
                            == lambda_2 * p_0[i]
                            for i in range(n)),
                            name="C1")
        model.addConstrs((  gp.quicksum((I[j] - gamma * P[j])[k,i] * u[j,k]
                            for k in range(n)
                            for j in range(m))
                            == p_0[i]
                            for i in range(n)),
                            name="C2")

        # Checking the correctness of constraints
        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        if DEBUG == 1:
            self.print_solution(model, method)
        # model.computeIIS()
        # model.write("../../files/RIRL_IIS.ilp")
        u_vec = np.zeros((m,n))
        v = model.getVars()
        i = 0
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                if (i < m*n):
                    x = int(i/n)
                    y = int(i%n)
                    u_vec[x,y] = v.x
                    i = i + 1
            opt_val = model.objVal
            assert((np.sum(u_vec) - 1/(1 - gamma) < 10**-3))
        else:
            print("No solution " + method)
            opt_val = None
        return u_vec, opt_val

    def Cons_r_optimal_u_l1_mat(self, epsilon):
        """this function based on our method.
        Args:
            episodes (ndarray): A whole experiment, consist of a number of episodes of run
        
        Returns:
            ndarray: Optimal policy
        """
        method = "Consistent R l1 mat " 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , method, '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        phi = self.phi
        # epsilon = 10.1
        I = self.I
        P = self.transition
        gamma = self.gamma
        p_0 = self.p_0
        u_E = self.u_E
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T
        flat_u_E = u_E.reshape(m * n)
        mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])

        # Model
        model = gp.Model("matrix")
        # x = [u, lambda_1, lambda_2, t]
        var_dim = (m * n) + (m * n) + 1 + 1
        l_bound = np.hstack([np.zeros(var_dim - 1), -GRB.INFINITY])
        x = model.addMVar(shape=var_dim, name="x", lb=l_bound)
        # setting the objective 
        obj = np.hstack([np.zeros(m * n + m *n), epsilon, 1])
        model.setObjective( obj @ x, GRB.MINIMIZE)
        # Constraints
        # t >= phi(u_E - u + lambda_1 - lambda_2 * u_E)[i]
        # Phi u  -  Phi lambda_1  +  lambda_2 mu_E  +  t 1_(kx1)  >=  mu_E
        A0 = np.hstack([flat_phi, -flat_phi, mu_E, np.ones([k_num, 1])])
        b0 = mu_E
        model.addMConstr(A0 , x, '>=', b0, name="C0") 
        # A^T lambda_1 = lambda_2 p0
        A1 = np.hstack([np.zeros([n, m * n]), flat_A_T, -p_0.reshape([n,1]),
                        np.zeros([n, 1])])
        b1 = np.zeros(n)
        model.addMConstr(A1, x, '=', b1, name="C1")
        # A^T u = p0
        A2 = np.hstack([flat_A_T, np.zeros([n, m * n]), np.zeros([n, 1]), 
                        np.zeros([n, 1])])
        b2 = p_0.reshape([n, 1])
        model.addMConstr(A2,  x ,'=', b2, name="C2")
        # A3 = np.hstack([np.zeros([n * m, n * m]), np.ones([n * m, n * m]),
        #                            np.zeros([n * m, 2])])
        # b3 = np.zeros(n * m)
        # model.addMConstr(A3,  x ,'>', b3, name="C3")
        # A4 = np.hstack([np.zeros([2, 2 * n * m]), np.ones([2, 1]),
        #                 np.zeros([2, 1])])
        # b4 = np.zeros(2)
        # # import ipdb; ipdb.set_trace()
        # model.addMConstr(A4,  x ,'>', b4, name="C4")
        model.Params.OutputFlag = 0
        # Solve
        model.optimize()
        model.write("../../files/" + method + ".lp")
        # Checking the correctness of constraints
        # presolve = model.presolve()
        # model.computeIIS()
        # presolve.optimize()
        # presolve.write("../../files/presolve.lp")
        # print("slack", model.slack)
        u_flat = self.gurobi_to_numpy(model, m * n, method)
        u_vec = u_flat.reshape([m, n])
        # u_vec = np.zeros((m,n))
        # v = model.getVars()
        # i = 0
        # if model.status == GRB.Status.OPTIMAL:
        #     for v in model.getVars():
        #         if (i < m*n):
        #             x0 = int(i/n)
        #             y0 = int(i%n)
        #             u_vec[x0,y0] = v.x
        #             i = i + 1
        #     opt_val = model.objVal
        #     assert((np.sum(u_vec) - 1/(1 - gamma) < 10**-2))
        #     if DEBUG == 1:
        #         # print(x.X)
        #         self.print_solution(model, method)
        # else:
        #     print("No solution for " + method)
        #     opt_val = None
        # print(model.x)
        return u_vec

    def Cons_r_l1_modified_uE(self, epsilon):
        """this function based on our method.
        Args:
            episodes (ndarray): A whole experiment, consist of a number of episodes of run
        
        Returns:
            ndarray: Optimal policy
        """
        method = "Consistent R l1 modifed" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , method, '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        phi = self.phi
        # epsilon = 10.1
        I = self.I
        P = self.transition
        gamma = self.gamma
        p_0 = self.p_0
        u_E = self.u_E
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T
        flat_u_E = u_E.reshape(m * n)
        mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])
        u_E_modified = self.get_u_E_consistent()
        mu_E_modified = (flat_phi @ u_E_modified.reshape(m * n)).reshape([k_num, 1])

        # Model
        model = gp.Model("matrix")
        # x = [u, lambda_1, lambda_2, t]
        var_dim = (m * n) + (m * n) + 1 + 1
        l_bound = np.hstack([np.zeros(var_dim - 1), -GRB.INFINITY])
        x = model.addMVar(shape=var_dim, name="x", lb=l_bound)
        # setting the objective 
        obj = np.hstack([np.zeros(m * n + m * n), epsilon, 1])
        model.setObjective( obj @ x, GRB.MINIMIZE)
        # Constraints
        # t >= phi(u_E - u + lambda_1 - lambda_2 * u_E)[i]
        # Phi u  -  Phi lambda_1  +  lambda_2 mu_E  +  t 1_(kx1)  >=  mu_E
        A0 = np.hstack([flat_phi, -flat_phi, mu_E_modified, np.ones([k_num, 1])])
        b0 = mu_E
        model.addMConstr(A0 , x, '>=', b0, name="C0") 
        # A^T lambda_1 = lambda_2 p0
        A1 = np.hstack([np.zeros([n, m * n]), flat_A_T, -p_0.reshape([n,1]),
                        np.zeros([n, 1])])
        b1 = np.zeros(n)
        model.addMConstr(A1, x, '=', b1, name="C1")
        # A^T u = p0
        A2 = np.hstack([flat_A_T, np.zeros([n, m * n]), np.zeros([n, 1]), 
                        np.zeros([n, 1])])
        b2 = p_0.reshape([n, 1])
        model.addMConstr(A2,  x ,'=', b2, name="C2")
        # A3 = np.hstack([np.zeros([n * m, n * m]), np.ones([n * m, n * m]),
        #                            np.zeros([n * m, 2])])
        # b3 = np.zeros(n * m)
        # model.addMConstr(A3,  x ,'>', b3, name="C3")
        # A4 = np.hstack([np.zeros([2, 2 * n * m]), np.ones([2, 1]),
        #                 np.zeros([2, 1])])
        # b4 = np.zeros(2)
        # # import ipdb; ipdb.set_trace()
        # model.addMConstr(A4,  x ,'>', b4, name="C4")
        model.Params.OutputFlag = 0
        # Solve
        model.optimize()
        model.write("../../files/" + method + ".lp")
        u_flat = self.gurobi_to_numpy(model, m * n, method)
        u_vec = u_flat.reshape([m, n])
        return u_vec

    def Cons_r_optimal_u_l2(self):
        """this function based on our method.
        Args:
            episodes (ndarray): A whole experiment, consist of a number of episodes of run
        
        Returns:
            ndarray: Optimal policy
        """
        method = "Consistent R" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , method, '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        phi = self.phi
        epsilon = 140.9
        I = self.I
        P = self.transition
        p_0 = self.p_0
        gamma = self.gamma
        u_E = self.u_E

        # Model
        model = gp.Model("mdp")
        u = model.addVars(m,n, name="u", lb=0.0)
        lambda_1 = model.addVars(m, n, name="lambda_1", lb=0.0)
        lambda_2 = model.addVar(name="lambda_2", lb=0.0)
        # setting the objective 
        model.setObjective(gp.quicksum((- u[i, j] + (1 - lambda_2) * u_E[i, j] +
                            lambda_1[i, j])  * phi[i, j, k] * phi[h, l, k] *
                            ( -u[h, l] + (1 - lambda_2) * u_E[h, l] +
                            lambda_1[i, j])
                            for i in range(m)
                            for h in range(m)
                            for k in range(k_num)
                            for l in range(n)
                            for j in range(n)) + epsilon * lambda_2,
                            GRB.MINIMIZE)
        # Constraints
        model.addConstrs((gp.quicksum((I[j] - gamma*P[j])[k,i] * lambda_1[j,k]
                            for k in range(n)
                            for j in range(m))
                            == lambda_2 * p_0[i]
                            for i in range(n)),
                            name="C1")
        model.addConstrs((  gp.quicksum((I[j] - gamma * P[j])[k,i] * u[j,k]
                            for k in range(n)
                            for j in range(m))
                            == p_0[i]
                            for i in range(n)),
                            name="C2")

        # Checking the correctness of constraints
        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        if DEBUG == 1:
            self.print_solution(model, method)
        # model.computeIIS()
        # model.write("../../files/RIRL_IIS.ilp")
        u_vec = np.zeros((m,n))
        v = model.getVars()
        i = 0
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                if (i < m*n):
                    x = int(i/n)
                    y = int(i%n)
                    u_vec[x,y] = v.x
                    i = i + 1
            opt_val = model.objVal
            assert((np.sum(u_vec) - 1/(1 - gamma) < 10**-3))
        else:
            print("No solution " + method)
            opt_val = None
        return u_vec, opt_val

    def Cons_r_optimal_u_l2_mat(self, epsilon):
        """this function based on our method.
        Args:
            episodes (ndarray): A whole experiment, consist of a number of 
            episodes of run
        
        Returns:
            ndarray: Optimal policy
        """
        method = "Consistent R l2 mat" 
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , method, '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        phi = self.phi
        # epsilon = 0.41
        I = self.I
        P = self.transition
        p_0 = self.p_0
        gamma = self.gamma
        u_E = self.u_E
        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T
        flat_u_E = u_E.reshape(m * n)
        mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])

        # Model
        model = gp.Model("matrix")
        # x = [u, lambda_1, lambda_2]
        var_dim = (m * n) + (m * n) + 1
        l_bound = np.hstack([np.zeros(var_dim)])
        x = model.addMVar(shape=var_dim, name="x", lb=l_bound)
        # setting the objective 
        Lambda = flat_phi.T @ flat_phi 
        Z = np.hstack([-np.eye(m * n), np.eye(m * n), - u_E.reshape([m * n, 1])])
        Q = Z.T @ Lambda @ Z
        E = np.hstack([np.zeros(2 * m * n), epsilon])
        B = E + 2 * flat_u_E.T @ Lambda @ Z
        c = flat_u_E.T @ Lambda @ flat_u_E 
        # import ipdb; ipdb.set_trace()
        model.setMObjective(Q, B, c, None, None, None, GRB.MINIMIZE)
        # B = np.hstack([-flat_phi, flat_phi, -mu_E])
        # Q = B.T @ B
        # c = mu_E
        # eps_vec = np.hstack([np.zeros(var_dim - 1), epsilon])
        # model.setMObjective(Q, 2 * ((c.T @ B).flatten() + eps_vec), 
        #                     (c.T @ c).flatten(),
        #                     None, None, None,
        #                     GRB.MINIMIZE)
        # Constraints
        # A^T lambda_1 = lambda_2 p0
        A1 = np.hstack([np.zeros([n, m * n]), flat_A_T, -p_0.reshape([n,1])])
        b1 = np.zeros(n)
        model.addMConstr(A1, x, '=', b1, name="C1")
        # A^T u = p0
        A2 = np.hstack([flat_A_T, np.zeros([n, m * n]), np.zeros([n, 1])])
        b2 = p_0.reshape([n, 1])
        model.addMConstr(A2,  x ,'=', b2, name="C2")

        # Checking the correctness of constraints
        model.write("../../files/" + method + ".lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        # model.printStats()
        if DEBUG == 1:
            self.print_solution(model, method)
        # model.computeIIS()
        # turn the gurobi variable to numpy
        u_flat = self.gurobi_to_numpy(model, m * n, method)
        u_vec = u_flat.reshape([m, n])
        return u_vec

    def Cons_u_optimal_r(self):
        """Abbeel's method with consistent set u (A^T * u = p_0).
        Returns: TODO

        """
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 + "Abbeel consistent" + "="*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        gamma = self.gamma
        P = self.transition
        p_0 = self.p_0
        # I_(m*n*n)
        # P_(m*(n*n))
        I = self.I
        A = (I - gamma*P)
        # phi(m*(n*k))
        phi = self.phi
        u_E = self.u_E

        model = gp.Model("mdp")
        w = model.addMVar(shape=k_num, name="w", lb=-GRB.INFINITY)
        y = model.addMVar(shape=n, name="y", lb=-GRB.INFINITY)
        model.setObjective(self.multi_phi(u_E).T @ w - p_0.T @ y, GRB.MAXIMIZE)
        model.addConstr(phi.reshape([m * n, k_num]) @ w -
                                A.reshape([ m * n, n ]) @ y <= 0, name="C0") 
        model.addConstr(w @ w <= 1, name="C1")

        model.write("../../files/Abbeel_consistent_u_dual.lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        t = model.objVal

        if DEBUG == 1:
            self.print_solution(model, 'Abbeels consistent')
        w_vec = np.zeros(k_num)
        y_vec = np.zeros(n)
        v = model.getVars()
        i = 0
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                if (i < k_num):
                    # x = int(i/n)
                    # y = int(i%k_num)
                    w_vec[i] = v.x
                else:
                    y_vec[i - k_num] = v.x
                i = i + 1
        else:
            print("No solution")
        rewards = phi @ w_vec
        return y_vec, rewards, t 

    def Cons_r_optimal_r(self):
        """this function based on our method.
        Args:
            episodes (ndarray): A whole experiment, consist of a number of episodes of run
        
        Returns:
            ndarray: Optimal policy
        """
        if DEBUG == 1:
            print("/\\"*32)
            print("="*20 , 'Abbeel consistent R', '='*20)
            print("/\\"*32)
        n = self.n
        m = self.m
        k_num = self.k
        epsilon = 0.5
        I = self.I
        P = self.transition
        p_0 = self.p_0
        phi = self.phi
        gamma = self.gamma
        u_E = self.u_E
        mu_E = self.multi_phi(u_E).reshape(k_num)
        # phi(m*(n*k))
        phi = self.phi

        # Model
        model = gp.Model("mdp")
        w = model.addVars(k_num, name="w", lb=-GRB.INFINITY)
        lambda_1 = model.addVars(n, name="lambda_1", lb=-GRB.INFINITY)
        v = model.addVars(n, name="v", lb=0)
        # setting the objective 
        model.setObjective(gp.quicksum( w[k] * mu_E[k]
                            for k in range(k_num)) - gp.quicksum(lambda_1[j] * p_0[j]
                            for j in range(n)),
                            GRB.MAXIMIZE)
        # Constraints
        model.addConstrs((-gp.quicksum((I[j] - gamma*P[j])[i,k] * lambda_1[k]
                            for k in range(n)) + gp.quicksum(phi[j, i, k] * w[k]
                            for k in range(k_num)) <= 0
                            for j in range(m)
                            for i in range(n)),
                            name="C1")
        model.addConstrs((-gp.quicksum((I[j] - gamma*P[j])[i,k] * v[k]
                            for k in range(n)) +
                            gp.quicksum(phi[j, i, l] * w[l]
                            for l in range(k_num)) <= 0
                            for j in range(m)
                            for i in range(n)),
                            name="C2")
        model.addConstr((  gp.quicksum(p_0[i] * v[i]
                            for i in range(n)) -
                            gp.quicksum(mu_E[k] * w[k]
                            for k in range(k_num)) - epsilon <= 0),
                            name="C3")
        model.addConstr((   gp.quicksum(w[k] * w[k]
                            for k in range(k_num)) <= 1),
                            name="C4")

        # Checking the correctness of constraints
        model.write("../../files/Abbeel_consistent_r_r.lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        if DEBUG == 1:
            self.print_solution(model, 'Abbeel consistent R')
        w_vec = np.zeros(k_num)
        v = model.getVars()
        i = 0
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                if (i < k_num):
                    w_vec[i] = v.x
                    i = i + 1
        else:
            print("No solution")
        rewards = phi @ w_vec
        return rewards 


    def constraint_generation(self):
        """Constraint generation for solving 
            max_u min_hat{u} min_r r^T(u - \hat{u}) 

        :returns: optimal u

        """
        method = "constraint generation"
        n = self.n
        m = self.m
        k_num = self.k
        # P = self.transition
        p_0 = self.p_0
        I = self.I
        # features matrix phi(m*(n*k))
        phi = self.phi
        gamma = self.gamma
        P = self.transition
        u_E = self.u_E

        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T
        # flat_u_E = u_E.reshape(m * n)
        # mu_E = (flat_phi @ flat_u_E).reshape([k_num, 1])

        len_demo_set = 10
        mu_hat = [self.multi_phi(self.get_u_E_consistent()).flatten()
                 for i in range(len_demo_set)]

        # x = [u, lambda_1, lambda_2, t]
        var_dim = (m * n) + 1
        l_bound = np.hstack([np.zeros(var_dim - 1), -GRB.INFINITY])
        # Model
        model = gp.Model()
        x = model.addMVar(shape=var_dim, name="x", lb=l_bound)
        C = np.hstack([np.zeros(m * n), 1])
        model.setMObjective(None, C, 0.0, None, None, x, GRB.MINIMIZE)

        # -1 t <= Phi u - \hat{\mu}
        A1 = np.hstack([flat_phi, np.ones([k_num, 1])])
        # -1 t <= Phi u - \hat{\mu}
        A2 = np.hstack([flat_phi, -np.ones([k_num, 1])])
        # import ipdb; ipdb.set_trace()
        for i in range(len_demo_set):
            model.addMConstr(A1, x , ">=", mu_hat[i])
            model.addMConstr(A2, x, "<=", mu_hat[i])

        # A^T u = p0
        A3 = np.hstack([flat_A_T, np.zeros([n, 1])])
        b3 = p_0.reshape([n, 1])
        model.addMConstr(A3, x ,'=', b3, name="C3")

        # t >= || u - hat_u ||_inf \forall hat_u
        # model.addConstrs((t >= u[i, j] - U_hat[k][i, j]
        #                   for i in range(m) for j in range(n)
        #                   for k in range(len(U_hat))),
        #                   name="C0")

        # model.addConstrs((t <= -(u[i, j] - U_hat[k][i, j])
        #                   for i in range(m) for j in range(n)
        #                   for k in range(len(U_hat))),
        #                   name="C2")


        # model.addConstrs((  gp.quicksum((I[j] - gamma*P[j])[k,i] * u[j, k]
        #                     for k in range(n)
        #                     for j in range(m) )
        #                     == p_0[i]
        #                     for i in range(n) ),
        #                     name="C1")

        model.write("../../files/" + method + ".lp")
        model.Params.OutputFlag = 0
        model.optimize()
        t1 = model.ObjVal
        u_vec = self.gurobi_to_numpy(model, m * n)
        return u_vec

    def analytical_chebyshev(self):
        """L infinity norm solution to the Chebyshev center
            max_u min_hat{u} min_r r^T(u - \hat{u}) 

        :returns: optimal u

        """
        method = "Chebyshev"
        n = self.n
        m = self.m
        k_num = self.k
        # P = self.transition
        p_0 = self.p_0
        I = self.I
        # features matrix phi(m*(n*k))
        phi = self.phi
        gamma = self.gamma
        P = self.transition

        flat_A_T = (I - gamma * P).reshape([m * n, n]).T
        flat_phi = phi.reshape([m * n, k_num]).T

        S_D = set()
        C = []

        demo = self.demonstrations[0]
        for s, a in demo:
            if s not in S_D:
                S_D.add(s)
                c2_list = np.zeros(m * n)
                c2_list[n * a:n * (a + 1)] = np.ones(n)
                c2_list[n * a + s] = 0
                C.append(c2_list)

        C = np.array(C)
        n_D = len(S_D)

        # Model
        model = gp.Model("Chebyshev")
        
        # define variable x = [u, delta, theta, zeta]
        u = model.addMVar(shape=(m * n), name="u", lb=0.0)
        delta = model.addMVar(shape=1, name="delta")
        theta = model.addMVar(shape=(n, 2 * k_num), name="theta",
                              lb=-GRB.INFINITY)
        zeta = model.addMVar(shape=(n_D, 2 * k_num), name="zeta", 
                              lb=-GRB.INFINITY)
        # define objective
        C0 = np.array([1])
        model.setMObjective(None, C0, 0.0, None, None, delta, GRB.MINIMIZE)

        # constraints
        #  1_i Phi u - p_0^T theta_i <= delta
        one_i = np.zeros([k_num, 1])
        for i in range(k_num):
            one_i[i, 0] = 1
            model.addConstr((one_i.T @ flat_phi) @ u -
                            p_0.T @ theta[:, i] <= delta,
                            name="C0"+str(i))
            one_i[i, 0] = 0

        # -1_i Phi u - p_0^T theta_i <= delta
        for i in range(k_num):
            one_i[i, 0] = -1
            model.addConstr((one_i.T @ flat_phi) @ u -
                            p_0.T @ theta[:, k_num + i] <= delta,
                            name="C1"+str(i))
            one_i[i, 0] = 0

        # A^T theta_i - Phi 1_i + C zeta_i >= 0
        for i in range(k_num):
            one_i[i, 0] = 1
            # import ipdb;ipdb.set_trace()
            model.addConstr(flat_A_T.T @ theta[:, i] + C.T @ zeta[:, i] >=
                              (one_i.T @ flat_phi)[0], name="C2"+str(i))
            one_i[i, 0] = 0

        # A^T theta_i + Phi 1_i + C zeta_i >= 0
        for i in range(k_num):
            one_i[i, 0] = -1
            # import ipdb;ipdb.set_trace()
            model.addConstr(flat_A_T.T @ theta[:, k_num + i] +
                            C.T @ zeta[:, k_num + i] >= (one_i.T @ flat_phi)[0],
                            name="C3"+str(i))
            one_i[i, 0] = 0

        # A^T u = p_0
        model.addConstr(flat_A_T @ u == p_0, name="C4")

        model.write("../../files/" + method + ".lp")
        model.Params.OutputFlag = 0
        model.optimize()
        t1 = model.ObjVal
        u_vec = self.gurobi_to_numpy(model, m * n)
        return u_vec

    def feature_averages(self, trajectory_feature):
        gamma = self.gamma
        horizon = len(self.demonstrations[0])
        gamma_t = np.array([gamma**j for j in range(horizon)]).reshape(horizon,1)
        average = np.sum(np.multiply(trajectory_feature, gamma_t),axis=0) 
        return average

    def episode_to_trajectory_feature(self, episode):
        trajectory = []
        for sample in episode:
            s0, a0 = sample
            trajectory.append(self.phi[a0, s0, :])
        return trajectory
    
    def calculate_objective(self, nonoptimal_demos, policy_features):
        '''For the partition function Z($\theta$), we just sum over all the
        exponents of their rewards, similar to the equation above equation (6)
        in the original paper.'''
        objective = np.dot(self.mu_E, self.weights)
        for i in range(nonoptimal_demos.shape[0]):
            objective -= np.exp(np.dot(policy_features[i],self.weights))
        return objective

    def calculate_expert_feature(self):
        expert_feature = np.zeros_like(self.weights)
        # import ipdb;ipdb.set_trace()
        for episode in self.demonstrations:
            trajectory = self.episode_to_trajectory_feature(episode)
            expert_feature += self.feature_averages(trajectory)
        expert_feature /= len(self.demonstrations)
        return expert_feature

    def rel_max_ent_irl(self, nonoptimal_demos, step_size=1e-4,num_iters=50000,print_every=5000):
        expert_feature = self.calculate_expert_feature()
        policy_features = np.zeros((len(nonoptimal_demos),self.num_features))
        for i in range(len(nonoptimal_demos)):
            policy_features[i] = feature_averages(nonoptimal_demos[i])
            
        importance_sampling = np.zeros((len(nonoptimal_demos),))
        for i in range(num_iters):
            update = np.zeros_like(self.weights)
            for j in range(len(nonoptimal_demos)):
                importance_sampling[j] = np.exp(np.dot(policy_features[j],
                                                       self.weights))
            importance_sampling /= np.sum(importance_sampling,axis=0)
            weighted_sum = np.sum(np.multiply(np.array([importance_sampling,] *
                                            policy_features.shape[1]).T,\
                                            policy_features),axis=0)
            self.weights += step_size*(expert_feature - weighted_sum)
            # One weird trick to ensure that the weights don't blow up the objective.
            self.weights = self.weights / np.linalg.norm(self.weights,
                                                         keepdims=True)
            if i%print_every == 0:
                print("Value of objective is: " + str(self.calculate_objective(
                                                    nonoptimal_demos)))

    def normalize(self, vals):
        """
        normalize to (0, max_val)
        input:
            vals: 1d array
        """
        min_val = np.min(vals)
        max_val = np.max(vals)
        return (vals - min_val) / (max_val - min_val)


    def compute_state_visition_freq(self, policy, deterministic=True):
        """compute the expected states visition frequency p(s| theta, T) 
        using dynamic programming
        inputs:
            P_a     NxNxN_ACTIONS matrix - transition dynamics
            gamma   float - discount factor
            trajs   list of list of Steps - collected from expert
            policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

        returns:
            p       Nx1 vector - state visitation frequencies
        """
        N_STATES = self.n
        N_ACTIONS = self.m
        gamma = self.gamma
        feat_map = self.phi[0,:,:]
        trajs = self.demonstrations
        P = self.transition
        # P_a = self.transition[a,:,:]

        T = len(trajs[0])
        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros([N_STATES, T]) 

        for traj in trajs:
            mu[traj[0][0], 0] += 1
        mu[:,0] = mu[:,0]/len(trajs)

        # import ipdb;ipdb.set_trace()
        for s in range(N_STATES):
            for t in range(T-1):
                # if deterministic:
                #     mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])]
                #                   for pre_s in range(N_STATES)])
                # else:
                mu[s, t+1] = sum([sum([mu[pre_s, t] * P[a1, pre_s, s] * \
                                        policy[a1, pre_s]
                                        for a1 in range(N_ACTIONS)])
                                        for pre_s in range(N_STATES)])
        p = np.sum(mu, 1)
        return p
    def max_ent_irl(self, lr, n_iters):
        """
        Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)
        inputs:
            feat_map    NxD matrix - the features for each state
            P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the 
                        transition prob of landing at state s1 when taking action 
                        a at state s0
            gamma       float - RL discount factor
            trajs       a list of demonstrations
            lr          float - learning rate
            n_iters     int - number of optimization steps
        returns
            rewards     Nx1 vector - recoverred state rewards
        """
        N_STATES = self.n
        N_ACTIONS = self.m
        feat_map = self.phi[0,:,:]
        # init parameters
        theta = np.random.uniform(size=(feat_map.shape[1],))
        # calc feature expectations
        feat_exp = self.calculate_expert_feature()
        # training
        for iteration in range(n_iters):
            # if iteration % int(n_iters/10) == 0:
            #     print("iteration: {}/{}".format(iteration, n_iters))
            # compute reward function
            rewards = np.dot(feat_map, theta)
            # compute policy
            policy,_ = self.value_iteration(rewards)
            # compute state visition frequences
            svf = self.compute_state_visition_freq(policy, deterministic=False)
            # compute gradients
            grad = feat_exp - feat_map.T.dot(svf)
            # update params
            theta += lr * grad
        rewards = np.dot(feat_map, theta)
        # return sigmoid(normalize(rewards))
        return self.normalize(rewards)

