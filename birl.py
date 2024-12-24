import numpy as np
import scipy.misc
import copy
from scipy.optimize import linprog
# import mdp
# import utils
# import mdp_worlds
# from mdp import ChainMDP


class BayesianIRL:
    def __init__(self, mdp_env, beta, step_stdev, debug=False, mcmc_norm = None, likelihood="birl", prior=None):
        self.mdp_env = copy.deepcopy(mdp_env) #deep copy just in case so we don't mess up original 
        self.beta = beta
        self.step_stdev = step_stdev
        self.num_states = mdp_env.n
        self.num_actions = mdp_env.m
        self.num_features = mdp_env.k
        self.reward_dim = mdp_env.n * mdp_env.m
        self.debug = debug
        self.mcmc_norm = mcmc_norm  #l2, inf, or l1
        self.likelihood = likelihood  #"birl" or "uniform"
        self.prior = prior
        self.state_features = mdp_env.phi

    def logsumexp(self, x):
        max_x = np.max(x)
        sum_exp = 0.0
        for xi in x:
            sum_exp += np.exp(xi - max_x)
        return max(x) + np.log(sum_exp)

    def log_likelihood(self, reward_hypothesis, q_values, demonstrations):
        #input is reward weights, q_values as a list [q(s0,a0), q(s1,a0), ..., q(sn,am)]
        # and demonstrations = [(s0,a0), ..., (sm,am)] list of state-action pairs
        #if self.prior is None:
        log_sum = 0.0
        if self.prior == "non-pos":
            #check if weights are all non-pos
            for r in reward_hypothesis:
                if r > 0:
                    return -np.inf
        for s,a in demonstrations:
            #there are no counterfactuals in a terminal state
            # if s not in self.mdp_env.terminals and a is not None:
            if a is not None:
                if self.likelihood == "birl":
                    Z_exponents = []
                    for b in range(self.num_actions):
                        Z_exponents.append(self.beta *
                                           q_values[s + self.num_states * b])
                    log_sum += self.beta * q_values[s + self.num_states * a] -\
                                            self.logsumexp(Z_exponents)
                elif self.likelihood == "uniform":
                    hinge_losses = 0.0

                    for b in range(self.num_actions):
                        hinge_losses += max(q_values[s + self.num_states * b] -
                                            q_values[s + self.num_states * a],
                                            0.0)
                        # print(hinge_losses)
                    log_sum += -self.beta * hinge_losses
                else:
                    raise NotImplementedError
        return log_sum

    #going to sample from L2-sphere but with negative weights 
    def print_policy_from_occupancies(self, u_sa,mdp_env):
        cnt = 0
        policy = self.get_optimal_policy_from_usa(u_sa, mdp_env)
        for r in range(mdp_env.n):
            row_str = ""
            for c in range(mdp_env.m):
                # if cnt not in mdp_env.terminals:
                row_str += mdp_env.get_readable_actions(np.argmax(policy[cnt])) + "\t"
                # else:
                #     row_str += ".\t"  #denote terminal with .
                cnt += 1
            print(row_str)

    def print_as_grid(x, mdp_env):
        #print into a num_rows by num_cols grid
        cnt = 0
        for r in range(mdp_env.num_rows):
            row_str = ""
            for c in range(mdp_env.num_cols):
                row_str += "{:.2f}\t".format(x[cnt])
                cnt += 1
            print(row_str)

    def get_optimal_policy_from_usa(self, u_sa, mdp_env):
        num_states, num_actions = mdp_env.n, mdp_env.m
        opt_stoch_pi = np.zeros((num_states, num_actions))
        for s in range(num_states):
            #compute the total occupancy for that state across all actions
            s_tot_occupancy = np.sum(u_sa[s::num_states])
            for a in range(num_actions):
                opt_stoch_pi[s][a] = u_sa[s+a*num_states] / max(s_tot_occupancy, 0.00000001)
        return opt_stoch_pi

    def get_policy_rewards(self, stoch_pi, rewards_sa):
        num_states, num_actions = stoch_pi.shape
        policy_rewards = np.zeros(num_states)
        for s, a_probs in enumerate(stoch_pi):
            expected_reward = 0.0
            for a, prob in enumerate(a_probs):
                index = s + num_states * a
                expected_reward += prob * rewards_sa[index]
            policy_rewards[s] = expected_reward
        return policy_rewards


    def get_policy_transitions(self, stoch_pi, mdp_env):
        num_states, num_actions = mdp_env.n, mdp_env.m
        P_pi = np.zeros((num_states, num_states))
        #calculate expectations
        for s1 in range(num_states):
            for s2 in range(num_states):
                cum_prob = 0.0
                for a in range(num_actions):
                    cum_prob += stoch_pi[s1,a] * mdp_env.transition[a, s1, s2]
                P_pi[s1,s2] =  cum_prob
        return P_pi

    def transform_to_R_sa(self, reward_weights):
        #assumes that inputs are the reward feature weights or state rewards
        #returns the vectorized R_sa 
        
        #first get R_s
        if reward_weights.shape == (self.num_features, ):
            # R_s = np.dot(self.state_features, reward_weights)
            phi = self.state_features
            R_s = phi[0,:] @ reward_weights
        elif reward_weights.shape == (self.num_states, ):
            R_s = reward_weights
        else:
            exit("Error, reward weights should be features or state rewards")
            # phi[m, n, k]
            # R_s = np.dot(self.state_features[0,:,:], reward_weights)
            # R_s = reward_weights
        return np.tile(R_s, self.num_actions)

    def solve_mdp_lp(self, mdp_env, reward_sa=None, debug=False):
        ''' method that uses Linear programming to solve MDP if reward_sa is not
            None, then it uses reward_sa in place of mdp_env.r_sa
        '''
        I_s = np.eye(mdp_env.n)
        gamma = mdp_env.gamma
        I_minus_gamma_Ps = []
        for P_a in mdp_env.transition:
            I_minus_gamma_Ps.append(I_s - gamma * P_a.transpose())
        A_eq = np.concatenate(I_minus_gamma_Ps, axis=1)
        b_eq = mdp_env.p_0
        # we want to maximize r_sa^T c so make it negative since scipy minimizes
        # by default
        if reward_sa is not None:
            c = -1.0 * reward_sa  
        else:
            c = -1.0 * mdp_env.r_sa  

        sol = linprog(c, A_eq=A_eq, b_eq = b_eq)
        #minimize:
        #c @ x
        #such that:
        #A_ub @ x <= b_ub
        #A_eq @ x == b_eq
        #all variables are non-negative by default

        if debug: print("expeced value MDP LP", -sol['fun'])  #need to negate the value to get the maximum
        u_sa = sol['x'] 
        return u_sa

    def solve_optimal_policy(self, reward_weights):
        # returns occupancy_frequencies and q_values
        # (vectorized |S|*|A| by q(s0,a0), q(s1,a0), ....)
        # import ipdb;ipdb.set_trace()
        reward_sa = self.transform_to_R_sa(reward_weights)
        #use optional argument to replace standard rewards with sample
        # import ipdb;ipdb.set_trace()
        # solve the dual lp for u
        occupancy_frequencies = self.solve_mdp_lp(self.mdp_env,
                                                 reward_sa=reward_sa) 
        
        num_states, num_actions = self.mdp_env.n, self.mdp_env.m
        gamma = self.mdp_env.gamma
        stochastic_policy = self.get_optimal_policy_from_usa(
                                        occupancy_frequencies, self.mdp_env)
        reward_policy = self.get_policy_rewards(stochastic_policy, reward_sa)
        transitions_policy = self.get_policy_transitions(stochastic_policy,
                                                        self.mdp_env)
        # transitions_policy = self.mdp_env.get_transitions(stochastic_policy)
        A = np.eye(num_states) - gamma * transitions_policy 
        b = reward_policy
        
        state_values = np.linalg.solve(A, b)
        Ps = tuple(self.mdp_env.transition[i] for i in range(num_actions))
        P_column = np.concatenate(Ps, axis=0)
        #print(P_column)
        q_values = reward_sa + gamma * np.dot(P_column, state_values)
        #q_values = mdp.get_q_values(occupancy_frequencies, self.mdp_env)
        return occupancy_frequencies, q_values

    def sample_init_reward(self):
        norm = self.mcmc_norm
        if norm is None or norm == "inf":
        #L-inf ball
            weights = 2*np.random.rand(self.num_features) - 1
            
        elif norm == "l2":
        #L2-ball
            weights = np.random.normal(0, self.step_stdev, self.num_features)
            weights /= np.linalg.norm(weights)

        elif norm == "l1":
        #l1-ball, kind of 
            weights = np.random.normal(0, self.step_stdev, self.num_features)
            weights = - np.abs(weights)
            weights /= np.sum(np.abs(weights))
        
        if self.prior == "non-pos":
            return -np.sign(weights) * weights
        else:
            return weights
        

    #generate normalized weights ||w||_2 =1
    def generate_proposal_weights(self, weights):
        norm = self.mcmc_norm
        new_weights = weights.copy()
        new_weights += np.random.normal(0, self.step_stdev, weights.shape)
        if norm == "inf":
            #normalize L-inf
            new_weights[new_weights > 1.] = 1.0
            new_weights[new_weights < -1.] = -1.0

        elif norm == "l2":
        #normalize L2
            new_weights /= np.linalg.norm(new_weights)

        elif norm == "l1":
        #L1
            new_weights /= np.sum(np.abs(new_weights))
        return new_weights


    def sample_posterior(self, demonstrations, num_samples, print_map_updates=False):
        # Fix the format of the demonstrations
        demos_sa = []
        if type(demonstrations[0]) is tuple and len(demonstrations[0]) == 2:
            #each element in demonstrations is a state-action pair so no preprocessing needed
            demos_sa = demonstrations
        else:
            #assume we have a list of lists of state-action tuples
            for d in demonstrations:
                for s,a in d:
                    demos_sa.append((s,a))
        
        #only save reward functions and occupancy_frequencies 
        reward_samples = []
        occupancy_frequencies = []
        map_weights = None
        map_occupancy = None
        
        #sample random reward hypothesis to start
        curr_weights = self.sample_init_reward()
        #print(curr_weights)
        curr_occupancies, curr_q_values = self.solve_optimal_policy(curr_weights)
        #compute log likelihood over demonstrations
        curr_ll =  self.log_likelihood(curr_weights, curr_q_values, demos_sa)
        best_ll = -np.inf
        
        #run MCMC
        accept_cnt = 0
        for step in range(num_samples):
            if self.debug: print("\n------\nstep", step)
            #compute proposal and log likelihood
            proposal_weights = self.generate_proposal_weights(curr_weights)

            #compute q_values and occupancy frequencies
            proposal_occupancies, proposal_q_values = self.solve_optimal_policy(
                                                                proposal_weights)

            if self.debug: print("proposal reward", proposal_weights)
            if self.debug: print("proposal qvalues", proposal_q_values)
            # if self.debug: self.print_policy_from_occupancies(
                                    # proposal_occupancies, self.mdp_env)

            prop_ll = self.log_likelihood(proposal_weights, proposal_q_values,
                                          demos_sa)
            if self.debug: print("prop_ll", prop_ll, "curr_ll", curr_ll)
            prob_accept = min(1.0, np.exp(prop_ll - curr_ll))
            if self.debug: print("prob accept", prob_accept)
            rand_sample = np.random.rand()
            if self.debug: print("rand prob", rand_sample)
            if rand_sample < prob_accept:
                accept_cnt += 1
                if self.debug: print("accept")
                #accept and add to chain
                reward_samples.append(proposal_weights)
                occupancy_frequencies.append(proposal_occupancies)
                curr_ll = prop_ll
                curr_weights = proposal_weights
                curr_occupancies = proposal_occupancies
                #update MAP
                if prop_ll > best_ll:
                    if print_map_updates: print(step)
                    # if self.debug: self.print_policy_from_occupancies(
                    #                             proposal_occupancies, self.mdp_env)
                    if self.debug: print("Q(s,a)", proposal_q_values)
                    best_ll = prop_ll
                    map_weights = proposal_weights.copy()
                    map_occupancy = proposal_occupancies.copy()
                    if print_map_updates: print("w_map", map_weights,
                                                "loglik = {:.4f}".format(best_ll))
            else:
                if self.debug: print("reject")
                reward_samples.append(curr_weights)
                occupancy_frequencies.append(curr_occupancies)
            #print out last reward sampled
            #print(reward_samples[-1])
                
        if self.debug: print("w_map", map_weights, "loglik", best_ll)
        if self.debug: print("accepted/total = {}/{} = {}".format(accept_cnt, num_samples, accept_cnt / num_samples))
        # if best_ll < -10:
        #     input("Didn't seem to converge... Check likelihoods and demos... Continue?")
        return map_weights, map_occupancy, np.array(reward_samples), np.array(occupancy_frequencies)

if __name__=="__main__":
    #mdp_env = mdp_worlds.two_state_chain()
    #demonstrations = [(1,0), (0,0)]

    mdp_env = mdp_worlds.machine_teaching_toy_featurized()
    demonstrations = [(2,3),(5,0),(4,0),(3,2)]

    beta = 10.0
    step_stdev = 0.2
    birl = BayesianIRL(mdp_env, beta, step_stdev, True)
    
    map_w, map_u, r_chain, u_chain = birl.sample_posterior(demonstrations, 100)
    print("map_weights", map_w)
    map_r = np.dot(birl.state_features, map_w)
    self.print_as_grid(map_r, mdp_env)
    print("Map policy")
    # self.print_policy_from_occupancies(map_u, mdp_env)
    print("chain")
    print(np.argmin(r_chain[:,0]))
    #for r in r_chain:
    #    print(r)
