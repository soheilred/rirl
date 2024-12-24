	# simple example
	# def rewardSample(self):
	# 	r = np.array([[[0, 5, 1], [50, -2, 0]],[[0, 1, 2], [5, -1, 0]]])
	# 	return r


	# input:
	# 	P(m*(n*n)): transition probability
	# 	phi(m*(n*k)): feature vector 
	# 	p_0(n): initial distribution 

	# output:
	# 	optimal policy (u_E)
	def cvar(self):
		n = self.n
		m = self.m
		l = self.k
		alpha = 0.1
		I = self.I
		P = self.transition
		p_0 = self.p_0
		gamma = self.gamma
		# reward function r(s,a), with first incrementing over s then a
		# r = self.rewardSample()
		num_r_set = r.shape[0] # number of sets in available reward sets


		# Model
		model = Model("mdp")

		# Create u variable--the lagrange variable for value function, more specifically called occupancy frequency
		u = model.addVars(m,n, name="u", lb=0)
		y = model.addVars(l, name="y", lb=0) # -y is replaced with y
		t = model.addVar(name="t", lb=0)


		# The objective is to maximize the rewards
		# model.setObjective((quicksum(-d[i] * t[i] for i in range(l))), GRB.MAXIMIZE)
		model.setObjective(t-1/(alpha*num_r_set)*quicksum(y[i] for i in range(num_r_set)), GRB.MAXIMIZE)

		# Constraints
		model.addConstrs((quicksum((I[i] - gamma*P[i])[k,j] * u[i,k]
							for k in range(n))
							== p_0[j]
							for j in range(n)
							for i in range(m) ),
							name="C0")


		model.addConstrs((-y[i] <= (quicksum(r[i,j,k] * u[j,k]
							for j in range(m)
							for k in range(n)) - t) 
							for i in range(num_r_set)), 
							name="C1")

		# Checking the correctness of constraints
		model.write("files/cvar.lp")

		# Solve
		model.optimize()
		return model



	def rirl0(self, episodes):
		# features matrix phi(m*(n*k))
		n = self.n
		m = self.m

		epsilon = 0.5
		I = self.I
		P = self.transition
		p_0 = self.p_0
		gamma = self.gamma
		r_max = 5 * np.ones([m,n])
		r_min = -1 * np.ones([m,n])

		u_E = self.uExpert(episodes)
	
		# Model
		model = Model("mdp")

		u = model.addVars(m,n, name="u", lb=0)
		t0 = model.addVar(name="t0", lb=0)
		t1 = model.addVars(m,n, name="t1", lb=0)
		t2 = model.addVars(m,n, name="t2", lb=0)

		# setting the objective for Expert's policy
		model.setObjective( t0 * epsilon - 
							quicksum( t2[i, j] * r_min[i, j] - t1[i, j] * r_max[i, j] 
							for i in range(m)
							for j in range(n) ), 
							GRB.MAXIMIZE) 


		# Constraints
		model.addConstrs((	quicksum((I[j] - gamma * P[j])[k,i] * t2[j,k]
							for j in range(m)
							for k in range(n))
							== t1 * p_0[i]
							for i in range(n) ),
							name="C0")

		model.addConstrs((	quicksum((I[j] - gamma * P[j])[k,i] * u[j,k]
							for j in range(m)
							for k in range(n))
							== p_0[i]
							for i in range(n) ),
							name="C1")

		# Checking the correctness of constraints
		model.write("files/RIRL.lp")

		# Solve
		model.optimize()
		# self.printSolution(model)
		policy = self.model2Policy(model)
		return policy


	def rirl1(self, episodes):
		# features matrix phi(m*(n*k))
		n = self.n
		m = self.m

		epsilon = 0.5
		I = self.I
		P = self.transition
		p_0 = self.p_0
		gamma = self.gamma

		u_E = self.uExpert(episodes)
	
		# Model
		model = Model("mdp")
		
		u = model.addVars(m,n, name="u", lb=0)
		t0 = model.addVar(name="t0", lb=0)
		t1 = model.addVar(name="t1", lb=0)
		t2 = model.addVars(m,n, name="t2", lb=0)

		# setting the objective for Expert's policy
		model.setObjective( t0 + t1 * epsilon, GRB.MAXIMIZE) 

		# Constraints
		model.addConstrs((	t0 <= ( u[i, j] - (t1) * u_E[i, j] + t2[i, j] )
							for i in range(m)
							for j in range(n) ),
							name="C0")

		model.addConstrs((	quicksum((I[j] - gamma * P[j])[k,i] * t2[j,k]
							for j in range(m)
							for k in range(n))
							== t1 * p_0[i]
							for i in range(n) ),
							name="C1")

		model.addConstrs((	quicksum((I[j] - gamma * P[j])[k,i] * u[j,k]
							for j in range(m)
							for k in range(n))
							== p_0[i]
							for i in range(n) ),
							name="C2")

		# Checking the correctness of constraints
		model.write("files/RIRL.lp")

		# Solve
		model.optimize()
		# self.printSolution(model)
		policy = self.model2Policy(model)
		return policy


	# transition probability P_(m*(n*n))
	# initial state distribution p_0(n)
	# r0: reward(m*n)
	# features matrix phi(m*(n*k))
	# u(m, n): occupancy frequency 

	def rIsConsistent(self, r0, episodes):
		epsilon = 0.9
		m = self.m
		n = self.n
		l = self.k
		I = self.I
		phi = self.phi
		P = self.transition
		p_0 = self.p_0
		gamma = self.gamma

		u_E = self.uExpert(episodes)

		# Model
		model = Model("mdp")

		w = model.addVars(l, name="w")
		v = model.addVars(n, name="v")

		model.setObjective(quicksum(
							quicksum((r0[j, i] - phi[j, i, k] * w[k]) * 
							(r0[j, i] - phi[j, i, k] * w[k])
							for k in range(l)) 
							for i in range(n)
							for j in range(m)),
							GRB.MINIMIZE)

		# Constraints
		model.addConstr((	quicksum((u_E[j, i] * phi[j, i, k] * w[k])
							for k in range(l)
							for i in range(n)
							for j in range(m)) + epsilon
							>= quicksum(p_0[i] * v[i]
							for i in range(n))),
							name="C0")

		model.addConstrs((	quicksum((I[j] - gamma*P[j])[i,k] * v[k]
							for k in range(n))
							>= quicksum(phi[j, i, k] * w[k]
							for k in range(l))
							for i in range(n)
							for j in range(m) ),
							name="C1")

		model.addConstr((	quicksum((u_E[j, i] * phi[j, i, k] * w[k])
							for k in range(l)
							for i in range(n)
							for j in range(m)) == 1 ),
							name="C2")
							
		# Checking the correctness of constraints
		model.write("files/RConsistency.lp")

		# Solve
		model.optimize()
		self.printSolution(model)





		# features matrix phi(m*(n*k))
	def set_phi(self, phi):
		self.phi = phi

	# features matrix phi(m*(n*k))
	def get_phi(self, s, a):
		return self.phi[a,s]

	def get_phi_matrix(self):
		return self.phi


	# This method is based on Andrew Ng's IRL paper. They are trying to maximize sum_s (Q*(s,pi*) - max_a Q^pi(s,a)

	# input:
	# 	P(m*(n*n)): transition probability
	#	lambda: l1 norm coefficient to encourage simple reward function
	# 	p_0(n): initial distribution 

	# output:
	# 	optimal policy (u_E)
	def AndrewsMethod(self, episodes):
		pass
				
	# input:
	#	lambda: l1 norm coefficient to encourage simple reward function
	# 	p_0(n): initial distribution 

	# output:
	# 	optimal policy (u_E)
	# def AndrewsMethod(self, episodes):
	# 	pass


	# This method is based on Abbeel's apprenticeship learning paper. They are trying to max_w min_pi[i] w(mu_e - mu_i)

	# input:
	# 	P(m*(n*n)): transition probability
	# 	phi(m*(n*k)): feature vector 
	# 	p_0(n): initial distribution 
	# def return_cvar(returns, alpha):
# 	import pandas as pd
# 
# 	df = pd.DataFrame(returns, columns=['returns'])
# 	# df = df.sort_values('returns', inplace=True, ascending=True)
# 	var_alpha = df.quantile(alpha)[0]
# 	more_than_var = [b for b in returns if b > var_alpha]
# 	if len(more_than_var) == 0:
# 		print("no value greater than var")
# 		return var_alpha
# 	cvar = np.mean(more_than_var)
# 	return cvar
#
	def get_mu(self, u, pi=None):
		"""Computes feature expectation (mu) of a given occupancy frequency u
		
		Args:
		    u (m * n nparray): Occupancy frequency
		    pi (n * 1 nparray, optional): Policy
		
		Returns:
		    k * 1 nparray: feature expection
		"""
		k = self.k
		n = self.n
		# m = self.m
		mu = np.zeros(k)
		# features matrix phi(m*(n*k))
		phi = self.phi
		if (u.shape == (n,)):
			phi_pi = np.zeros((n,k))
			for s in range(n):
				phi_pi[s] = phi[pi[s], s, :]
			mu = phi_pi.T @ u
			return mu
		assert(u.shape == phi[:,:,0].shape)
		for i in range(k):
			flatten_phi = phi[:,:,i].flatten()
			flatten_u = u.flatten()
			mu[i] = np.dot(flatten_phi, flatten_u)
		# print(mu)
		return mu



#           t = model.addVar(name="t", lb=-1000)
#           model.addConstrs((  t >= - mu_E[k] + mus[j][k] 
#                               for k in range(l)
#                               for j in range(i) ),
#                               name="C0")
#           model.addConstrs((  t <= (-mu_E[k] + mus[j][k]) * (-mu_E[k] + mus[j][k])
#                               for k in range(l)
#                               for j in range(i)),
#                               name="C0")
#           model.addConstr((   quicksum(w[k] * w[k]
#                               for k in range(l)) <= 1),
#                               name="C1")


#           t = model.addVar(name="t", lb=-1000)
#           model.addConstrs((  t >= - mu_E[k] + mus[j][k] 
#                               for k in range(l)
#                               for j in range(i) ),
#                               name="C0")
#           model.addConstrs((  t <= (-mu_E[k] + mus[j][k]) * (-mu_E[k] + mus[j][k])
#                               for k in range(l)
#                               for j in range(i)),
#                               name="C0")
#           model.addConstr((   quicksum(w[k] * w[k]
#                               for k in range(l)) <= 1),
#                               name="C1")


#           t = model.addVar(name="t", lb=-1000)
#           model.addConstrs((  t >= - mu_E[k] + mus[j][k] 
#                               for k in range(l)
#                               for j in range(i) ),
#                               name="C0")
#           model.addConstrs((  t <= (-mu_E[k] + mus[j][k]) * (-mu_E[k] + mus[j][k])
#                               for k in range(l)
#                               for j in range(i)),
#                               name="C0")
#           model.addConstr((   quicksum(w[k] * w[k]
#                               for k in range(l)) <= 1),
#                               name="C1")


#           t = model.addVar(name="t", lb=-1000)
#           model.addConstrs((  t >= - mu_E[k] + mus[j][k] 
#                               for k in range(l)
#                               for j in range(i) ),
#                               name="C0")
#           model.addConstrs((  t <= (-mu_E[k] + mus[j][k]) * (-mu_E[k] + mus[j][k])
#                               for k in range(l)
#                               for j in range(i)),
#                               name="C0")
#           model.addConstr((   quicksum(w[k] * w[k]
#                               for k in range(l)) <= 1),
#                               name="C1")


#           t = model.addVar(name="t", lb=-1000)
#           model.addConstrs((  t >= - mu_E[k] + mus[j][k] 
#                               for k in range(l)
#                               for j in range(i) ),
#                               name="C0")
#           model.addConstrs((  t <= (-mu_E[k] + mus[j][k]) * (-mu_E[k] + mus[j][k])
#                               for k in range(l)
#                               for j in range(i)),
#                               name="C0")
#           model.addConstr((   quicksum(w[k] * w[k]
#                               for k in range(l)) <= 1),
#                               name="C1")


          t = model.addVar(name="t", lb=-1000)
          model.addConstrs((  t >= - mu_E[k] + mus[j][k] 
                              for k in range(l)
                              for j in range(i) ),
                              name="C0")
          model.addConstrs((  t <= (-mu_E[k] + mus[j][k]) * (-mu_E[k] + mus[j][k])
                              for k in range(l)
                              for j in range(i)),
                              name="C0")
          model.addConstr((   quicksum(w[k] * w[k]
                              for k in range(l)) <= 1),
                              name="C1")


            # Model
            model = Model("mdp")
            alpha = model.addVars(i, name="alpha", lb=0)
            # Objective
            model.setObjective( quicksum(alpha[j] * 
                                (-mu_E[k] + mus[j][k]) * (-mu_E[k] + mus[j][k])
                                for k in range(l)
                                for j in range(i)),
                                GRB.MINIMIZE)
            # Constraints  
            model.addConstr((   quicksum(alpha[k] 
                                for k in range(i)) == 1),
                                name="C1")

            model.write("../files/Abbeel.lp")
            # Solve
            model.Params.OutputFlag = 0
            model.optimize()
            t = model.objVal
            # import ipdb; ipdb.set_trace()
            alpha_vec = self.gurobi_to_numpy(model, i)
            w = np.zeros(l)
            w[np.argmax(mu_E - mus[np.argmax(alpha_vec)])] = 1
 

    def Abbeel_with_consistent_u_tmp(self):
        """Abbeel's method with consistent set u (A^T * u = p_0).
        Returns: TODO

        """
        print("/\\"*32)
        print("="*32 + "Abbeel consistent" + "="*32)
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

        model = Model("mdp")
        u = model.addVars(m, n, name="u", lb=0.0)
        w = model.addVars(k_num, name="w", lb=-10)
        t = model.addVar(name="t", lb=-1000)
        model.setObjective(t, GRB.MAXIMIZE)
        model.addConstr((  t <= quicksum(w[k] * phi[i, j, k] *
                            (u_E[i, j] - u[i, j]) 
                            for k in range(k_num)
                            for j in range(n)
                            for i in range(m) )),
                            name="C0")
        model.addConstr((   quicksum(w[k] * w[k]
                            for k in range(k_num)) <= 1),
                            name="C1")
        model.addConstrs((  quicksum((I[j] - gamma*P[j])[k,i] * u[j,k]
                            for k in range(n)
                            for j in range(m) )
                            == p_0[i]
                            for i in range(n) ),
                            name="C2")

        model.write("../files/Abbeel_with_consistent_u.lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        t = model.objVal

        self.print_solution(model, 'Abbeels consistent')
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
        else:
            print("No solution")
        return u_vec




    def Abbeel_with_consistent_u_tmp(self):
        """Abbeel's method with consistent set u (A^T * u = p_0).
        Returns: TODO

        """
        print("/\\"*32)
        print("="*32 + "Abbeel consistent" + "="*32)
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

        model = Model("mdp")
        u = model.addVars(m, n, name="u", lb=0.0)
        model.setObjective(quicksum(phi[i, j, k] * (u_E[i, j] - u[i, j])
                            for k in range(k_num)
                            for j in range(n)
                            for i in range(m)),
                            GRB.MINIMIZE)
        model.addConstrs((  quicksum((I[j] - gamma*P[j])[k,i] * u[j,k]
                            for k in range(n)
                            for j in range(m) )
                            == p_0[i]
                            for i in range(n) ),
                            name="C2")

        model.write("../files/Abbeel_with_consistent_u.lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()

        self.print_solution(model, 'Abbeels consistent')
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
        else:
            print("No solution")
        return u_vec


        # if model.SolCount> 0:
        #   model.printAttr('x')
        # else:
        #
        model = Model("mdp")
        u = model.addVars(m, n, name="u", lb=0.0)
        w = model.addVars(k_num, name="w", lb=-10)
        t = model.addVar(name="t", lb=-1000)
        model.setObjective(t, GRB.MAXIMIZE)
        model.addConstr((  t <= quicksum(w[k] * phi[i, j, k] *
                            (u_E[i, j] - u[i, j]) 
                            for k in range(k_num)
                            for j in range(n)
                            for i in range(m) )),
                            name="C0")
        model.addConstr((   quicksum(w[k] * w[k]
                            for k in range(k_num)) <= 1),
                            name="C1")
        model.addConstrs((  quicksum((I[j] - gamma*P[j])[k,i] * u[j,k]
                            for k in range(n)
                            for j in range(m) )
                            == p_0[i]
                            for i in range(n) ),
                            name="C2")

        model.write("../files/Abbeel_with_consistent_u.lp")
        # Solve
        model.Params.OutputFlag = 0
        model.optimize()
        t = model.objVal

        self.print_solution(model, 'Abbeels consistent')
        u_vec = np.zeros((m,n))
        w_vec = np.zeros(k_num)
        v = model.getVars()
        i = 0
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                if (i < m*n):
                    x = int(i/n)
                    y = int(i%n)
                    u_vec[x,y] = v.x
                elif (i < m * n + k_num):
                    w_vec[i - m * n] = v.x
                i = i + 1
        else:
            print("No solution")
        return u_vec, self.multi_phi(w_vec)

   print('No solution')



        w = model.addVars(k_num, name="w", lb=-10)
        y = model.addVars(n, name="y", lb=0.0)
        model.setObjective(quicksum(w[k] * phi[i,j,k] * u_E[i,j]
                            for i in range(m)
                            for j in range(n)
                            for k in range(k_num)) - 
                            quicksum(p_0[j] * y[j]
                            for j in range(n)),
                            GRB.MAXIMIZE)
        model.addConstr((   quicksum(w[k] * w[k]
                            for k in range(k_num)) <= 1),
                            name="C0")
        model.addConstrs((  quicksum((I[j] - gamma*P[j])[i,l] * y[l] -
                            phi[j, i, k] * w[k]
                            for l in range(n) 
                            for k in range(k_num)) >= 0
                            for j in range(m) 
                            for i in range(n) ),
                            name="C1")

        u = model.addMVar(shape=m * n, name="u", lb=0.0)
        model.setObjective((u - u_E) @ f_phi @\
                            f_phi.transpose([1, 0]) @ (u - u_E),
                            GRB.MAXIMIZE)
        model.addConstr(phi.reshape([m * n, k_num]) @ w -
                                A.reshape([ m * n, n ]) @ y <= 0, name="C0") 
        model.addConstr(w @ w <= 1, name="C1")
        
        # flat_u_E = u_E.reshape(m * n)
        # reduced_phi = phi.reshape([m * n, k_num])
        # trans_phi = reduced_phi.transpose([1, 0])
        # model.setObjective( (flat_u_E - u + lambda_1) @ trans_phi @
                            # reduced_phi @ ((1 - lambda_2) * flat_u_E - u + lambda_1)
                            # , GRB.MINIMIZE)

    def model_to_policy(self, model):
        """Turns a gurobi solution to a policy.
        
        Args:
            model (gurobi solution): This is a solution of the linear program
        
        Returns:
            ndarray: A policy correspond to the solution
        """
        n = self.n
        m = self.m
        policy = np.zeros(n)
        u = np.zeros((m,n))
        v = model.getVars()
        i = 0

        # if model.SolCount > 0:
        if model.status == GRB.Status.OPTIMAL:
            for v in model.getVars():
                if (i < m*n):
                    x = int(i/n)
                    y = int(i%n)
                    u[x,y] = v.x
                    i = i + 1
            # print(u)
        else:
            print("No solution")
        # for j in range(n):
        #   policy[j] = np.argmax([u[i*n + j].x for i in range(m)])
        # print(len(v))
        # for v in model.getVars():
        #   print(v)
            # print('%s %g' % (v.varName, v.x))
        policy = np.argmax(u, axis=0)
        # print(policy)
        return policy
        
    # ax.plot(my_list, zero_one_returns[:, 0], label='Optimal', color='r',
            # marker="v")
    # ax.plot(my_list, zero_one_returns[:, 1], label='Abbeel VI', color='b',
            # marker="o")
    # ax.plot(my_list, zero_one_returns[:, 2], label='Syed', color='g',
            # marker="^")
    # ax.plot(my_list, zero_one_returns[:, 3], label='Consistent U',color='y',
            # marker="s")
    # ax.plot(my_list, zero_one_returns[:, 4],
            # label='Consistent R',color='pink', marker="D")


    cum_data = {
        'OG': {'cu': [],'syed': [],'l1c': [], 'l2c': []},
        'AC': {'cu': [],'syed': [],'l1c': [], 'l2c': []},
        'GC': {'cu': [],'syed': [],'l1c': [], 'l2c': []},
        'LSTD': {'cu': [],'syed': [],'l1c': [], 'l2c': []},
    }
    l1_closeness = {'OG': [], 'AC': [], 'GC': [], 'LSTD': []}
    l2_closeness = {'OG': [], 'AC': [], 'GC': [], 'LSTD': []}
    optimals_r = []
    for i in range(num_iterations):
        print(f'{i} iterations have been completed!!!!!!!!!!!!!!!!')
        # episodes_opt is the experiment that the optimal policy is obtained from
        episodes_opt, returns_opt = create_samples(num_actions, num_states,
                                                   transition, rewards, p_hat, gamma,
                                            opt_policy, num_episodes, episodes_len)
        optimals_r.append(returns_opt)
        # passing the samples to the mdp object
        mdp.get_u_E(episodes_opt)
        # test_policy = 1/2 * np.ones([num_actions, num_states])
        # test_policy = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
        # u_test = mdp.policy_to_u(test_policy)
        u_E = mdp.u_E
        data, closeness = various_estimators(mdp, rewards, u_E, opt_u, episodes_opt)
        for u_method in ['OG', 'AC', 'GC', 'LSTD']:
            for irl in ['cu', 'syed', 'l1c', 'l2c']:
                cum_data[u_method][irl].append(data[u_method][irl])

        l1_closeness["OG"].append(closeness["OG"][0])
        l1_closeness["AC"].append(closeness["AC"][0])
        l1_closeness["GC"].append(closeness["GC"][0])
        l1_closeness["LSTD"].append(closeness["LSTD"][0])
        l2_closeness["OG"].append(closeness["OG"][1])
        l2_closeness["AC"].append(closeness["AC"][1])
        l2_closeness["GC"].append(closeness["GC"][1])
        l2_closeness["LSTD"].append(closeness["LSTD"][1])

    print('OPTIMAL:', np.mean(optimals_r), np.var(optimals_r))
    df = {
        'method': ['OG', 'AC', 'GC', 'LSTD'],
        'Consistent-U': [f'{np.mean(cum_data["OG"]["cu"])}, {np.var(cum_data["OG"]["cu"])}',
                         f'{np.mean(cum_data["AC"]["cu"])}, {np.var(cum_data["AC"]["cu"])}',
                         f'{np.mean(cum_data["GC"]["cu"])}, {np.var(cum_data["GC"]["cu"])}',
                         f'{np.mean(cum_data["LSTD"]["cu"])}, {np.var(cum_data["LSTD"]["cu"])}'],
        'Syed': [f'{np.mean(cum_data["OG"]["syed"])}, {np.var(cum_data["OG"]["syed"])}',
                         f'{np.mean(cum_data["AC"]["syed"])}, {np.var(cum_data["AC"]["syed"])}',
                         f'{np.mean(cum_data["GC"]["syed"])}, {np.var(cum_data["GC"]["syed"])}',
                         f'{np.mean(cum_data["LSTD"]["syed"])}, {np.var(cum_data["LSTD"]["syed"])}'],
        'l1c': [f'{np.mean(cum_data["OG"]["l1c"])}, {np.var(cum_data["OG"]["l1c"])}',
                         f'{np.mean(cum_data["AC"]["l1c"])}, {np.var(cum_data["AC"]["l1c"])}',
                         f'{np.mean(cum_data["GC"]["l1c"])}, {np.var(cum_data["GC"]["l1c"])}',
                         f'{np.mean(cum_data["LSTD"]["l1c"])}, {np.var(cum_data["LSTD"]["l1c"])}'],
        'l2c': [f'{np.mean(cum_data["OG"]["l2c"])}, {np.var(cum_data["OG"]["l2c"])}',
                         f'{np.mean(cum_data["AC"]["l2c"])}, {np.var(cum_data["AC"]["l2c"])}',
                         f'{np.mean(cum_data["GC"]["l2c"])}, {np.var(cum_data["GC"]["l2c"])}',
                         f'{np.mean(cum_data["LSTD"]["l2c"])}, {np.var(cum_data["LSTD"]["l2c"])}']
    }
    pd.DataFrame.from_dict(df).to_csv('estimation_results.csv')
    for meth in ["OG", "AC", "GC", "LSTD"]:
        print(f"{meth} AVERAGE CLOSENESS (L1): {np.mean(l1_closeness[meth])} +- {np.var(l1_closeness[meth])}")
        print(f"{meth} AVERAGE CLOSENESS (L2): {np.mean(l2_closeness[meth])} +- {np.var(l2_closeness[meth])}")


def gerry_print_resutls(mdp, rewards, u_E, methodname):
    num_actions = 2
    mdp.set_u_expert(u_E)
    # print(u_test)
    # mdp.u_E = u_test
    # uncomment this line to set u^E to the optimal u
    # mdp.u_E = opt_occ_freq
    # max_ent_reward = mdp.max_ent_irl(0.9, 100)
    # print(max_ent_reward)
    # max_ent_reward = np.array([max_ent_reward for a in range(num_actions)])
    # max_end_policy, _ = mdp.dual_lp(max_ent_reward)
    # print(max_end_policy)
    # opt_occ_freq, opt_val = mdp.dual_lp(rewards)
    # using the dual of minimization problem
    # primal_LP = mdp.primal_lp(rewards)
    # Abbeels_vi_u, Abbeel_vi_reward, Abbeel_vi_opt_val = mdp.Abbeel_dual_vi()
    Abbeels_lp_u_mat, Abbeel_reward_mat, Abbeel_lp_opt_val_mat = mdp.\
                                            Abbeel_dual_lp_mat()
    # Abbeels_prob_simplex, Abbeel_prob_reward, simplex_opt_val =
    # mdp.Abbeel_prob_simplex()
    Syeds_u_mat, Syed_opt_val_mat = mdp.Syed_mat()
    consistent_u_u_mat, const_u_u_opt_val_mat = mdp.Cons_u_optimal_u_mat()
    # consistent_r_l1_mat, const_r_l1_opt_val_mat = mdp.Cons_r_optimal_u_l1_mat(epsilon)
    # consistent_r_l2_mat, const_r_l2_opt_val_mat = mdp.Cons_r_optimal_u_l2_mat(epsilon)
    # Abbeels_lp_u, Abbeel_reward, Abbeel_lp_opt_val =
    # mdp.Abbeel_consistent_ru()

    # print("="*40)
    # print("=" * 11 + "optimal values" + "=" * 11)
    # print("="*40)
    # print("Abbeel lp\t", round(Abbeel_lp_opt_val_mat, 3))
    # print("cons u\t\t", round(const_u_u_opt_val_mat, 3))
    # print("cons r l2\t", round(const_r_l2_opt_val_mat, 3))
    # print("Syed\t\t", round(Syed_opt_val_mat, 3))
    # print("cons r l1\t", round(const_r_l1_opt_val_mat, 3))
    # print("="*40)
    # print("=" * 11 + "optimal solutions" + "=" * 11)
    # print("="*40)
    # print("True Optimal u\n", opt_occ_freq)
    # print("Expert's u\n", u_E)
    # print("Primal\n", primal_LP)
    # print("Abbeel dual vi\n", Abbeels_vi_u)
    # print("Abbeel lp\n", Abbeels_lp_u_mat)
    # print("Abbeel probability simplex\n", Abbeels_prob_simplex)
    # print("cons U's u\n", consistent_u_u_mat)
    # print("Syed's u\n", Syeds_u_mat)
    # print("cons R's u l1\n", consistent_r_l1_mat)
    # print("="*40)

    # u_opt = mdp.policy_to_u(optimal_policy)
    # print("=" * 6, "True Return (u^T r_{True})", "=" * 6)
    # print("="*40)

    eps_list = [0, 0.1, 1.0, 10]
    estimation_paper = True
    if estimation_paper:
        consistent_r_u_l1_epsion0, _ = None, None
        consistent_r_u_l2_epsion0, _ = None, None
        consistent_r_u_l1_epsion1, _ = None, None
        consistent_r_u_l2_epsion1, _ = None, None
        consistent_r_u_l1_epsion2, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
        consistent_r_u_l2_epsion2, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
        consistent_r_u_l1_epsion3, _ = None, None
        consistent_r_u_l2_epsion3, _ = None, None
    else:
        consistent_r_u_l1_epsion0, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[0])
        consistent_r_u_l2_epsion0, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[0])
        consistent_r_u_l1_epsion1, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[1])
        consistent_r_u_l2_epsion1, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[1])
        consistent_r_u_l1_epsion2, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[2])
        consistent_r_u_l2_epsion2, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[2])
        consistent_r_u_l1_epsion3, _ = mdp.Cons_r_optimal_u_l1_mat(eps_list[3])
        consistent_r_u_l2_epsion3, _ = mdp.Cons_r_optimal_u_l2_mat(eps_list[3])

    # print("| Optimal u\t\t|", mdp.get_return(rewards, opt_occ_freq))
    # print("Abbeel dual vi\t\t\t",
                        # mdp.get_return(rewards, Abbeels_vi_u))
    # print("Abbeel dual lp\t\t\t",
                        # mdp.get_return(rewards, Abbeels_lp_u))
    # print("Abbeel prob\t\t\t",
                        # mdp.get_return(rewards, Abbeels_prob_simplex))
    # print("| Abbeel lp\t\t|", mdp.get_return(rewards, Abbeels_lp_u_mat))
    # print("| consistent R l2\t|", mdp.get_return(rewards, consistent_r_l2_mat))
    # print("consistent R l1\t\t", mdp.get_return(rewards, consistent_r_l1_mat))
    # print("| consistent R \t\t|  ")
    if estimation_paper:
        cu = mdp.get_return(rewards, consistent_u_u_mat)
        syed = mdp.get_return(rewards, Syeds_u_mat)

        l1_a = None
        l1_b = None
        l1_c = mdp.get_return(rewards, consistent_r_u_l1_epsion2)
        l1_d = None

        l2_a = None
        l2_b = None
        l2_c = mdp.get_return(rewards, consistent_r_u_l2_epsion2)
        l2_d = None
    else:
        cu = mdp.get_return(rewards, consistent_u_u_mat)
        print("| consistent U\t\t|", cu)
        syed = mdp.get_return(rewards, Syeds_u_mat)
        print("| Syed\t\t\t|", syed)

        l1_a = mdp.get_return(rewards, consistent_r_u_l1_epsion0)
        print("| l1, eps=", eps_list[0], "\t\t|", l1_a)
        l1_b = mdp.get_return(rewards, consistent_r_u_l1_epsion1)
        print("| l1, eps=", eps_list[1], "\t\t|", l1_b)
        l1_c = mdp.get_return(rewards, consistent_r_u_l1_epsion2)
        print("| l1, eps=", eps_list[2],"\t\t|",  l1_c)
        l1_d = mdp.get_return(rewards, consistent_r_u_l1_epsion3)
        print("| l1, eps=", eps_list[3],"\t\t|",  l1_d)

        l2_a = mdp.get_return(rewards, consistent_r_u_l2_epsion0)
        print("| l2, eps=", eps_list[0], "\t\t|", l2_a)
        l2_b = mdp.get_return(rewards, consistent_r_u_l2_epsion1)
        print("| l2, eps=", eps_list[1],"\t\t|",  l2_b)
        l2_c = mdp.get_return(rewards, consistent_r_u_l2_epsion2)
        print("| l2, eps=", eps_list[2],"\t\t|",  l2_c)
        l2_d = mdp.get_return(rewards, consistent_r_u_l2_epsion3)
        print("| l2, eps=", eps_list[3],"\t\t|",  l2_d)
    return cu, syed, l1_a, l1_b, l1_c, l1_d,  l2_a, l2_b, l2_c, l2_d

def various_estimators(mdp, rewards, u_E, opt_u, episodes_opt):
    gamma = mdp.gamma
    # AC_uE, _ = mdp.FVAC(episodes=episodes_opt, alphW=False, alphR=False)
    # GC_uE, _ = mdp.FVGC(episodes=episodes_opt, alphW=False, alphR=False)

    ACAW_uE, _ = mdp.FVAC(episodes=episodes_opt, alphW=True, alphR=False)
    GCAW_uE, _ = mdp.FVGC(episodes=episodes_opt, alphW=True, alphR=False)
    LSTD_mu = mdp.LSTD_muEstimator(episodes=episodes_opt)
    # pprint.pprint(ACAW_uE)
    # pprint.pprint(GCAW_uE)
    # pprint.pprint(LSTD_mu)
    # exit()

    # ACAR_uE, _ = mdp.FVAC(episodes=episodes_opt, alphW=False, alphR=True)
    # GCAR_uE, _ = mdp.FVGC(episodes=episodes_opt, alphW=False, alphR=True)

    method_and_result = [
                ("OG", u_E * (1/(1-gamma)) / u_E.sum()),
                # ("AC_uE", AC_uE),("GC_uE", GC_uE),
                ("AC", ACAW_uE),("GC", GCAW_uE), ("LSTD", LSTD_mu)
                # ("ACAR_uE", ACAR_uE),("GCAR_uE", GCAR_uE)
    ]
    # data = {
    #     'method': [],
    #     'Consistent-U': [],
    #     'Syed': [],
    #     'l1a': [],
    #     'l1b': [],
    #     'l1c': [],
    #     'l1d': [],
    #     'l2a': [],
    #     'l2b': [],
    #     'l2c': [],
    #     'l2d': [],
    # }
    data = {
        'OG': {},
        'AC': {},
        'GC': {},
        'LSTD': {},
    }
    # print("OPT\n", opt_u)
    # for method in method_and_result:
    #     print(f"{method[0]}\n", method[1])
    for u in method_and_result:
        # print(f"\n\nmethod: {u[0]} __________________________________")
        cu, syed, l1a, l1b, l1c, l1d,  l2a, l2b, l2c, l2d = gerry_print_resutls(mdp,
                                            rewards, u[1], u[0]) #  * (1/(1-gamma)) / u[1].sum()
        data[u[0]] = {
            "cu": cu,
            "syed": syed,
            "l1c": l1c,
            "l2c": l2c,
        }
    #     data['method'].append(u[0])
    #     data['Consistent-U'].append(cu)
    #     data['l1a'].append(l1a)
    #     data['l1b'].append(l1b)
    #     data['l1c'].append(l1c)
    #     data['l1d'].append(l1d)
    #     data['l2a'].append(l2a)
    #     data['l2b'].append(l2b)
    #     data['l2c'].append(l2c)
    #     data['l2d'].append(l2d)
    #     data['Syed'].append(syed)
    # pd.DataFrame.from_dict(data).to_csv('irl_results.csv')

    # in form: {'method': {"irl_key": result, ...} ...}
    closeness = {
        'OG': (np.linalg.norm(u_E - opt_u, ord=1), np.linalg.norm(u_E - opt_u, ord=2)),
        'AC': (np.linalg.norm(ACAW_uE - opt_u, ord=1), np.linalg.norm(ACAW_uE - opt_u, ord=2)),
        'GC': (np.linalg.norm(GCAW_uE - opt_u, ord=1), np.linalg.norm(GCAW_uE - opt_u, ord=2)),
        'LSTD': (np.linalg.norm(LSTD_mu - opt_u, ord=1), np.linalg.norm(LSTD_mu - opt_u, ord=2)),
    }
    return data, closeness


   
def irl_various_estimators(mdp, episodes_opt, norm):
    gamma = mdp.gamma
    u_E = mdp.u_E
    ACAW_uE, _ = mdp.FVAC(episodes=episodes_opt, alphW=True, alphR=False)
    GCAW_uE, _ = mdp.FVGC(episodes=episodes_opt, alphW=True, alphR=False)
    # print(u_E)
    # print(ACAW_uE)
    # print(GCAW_uE)
    method_and_result = [
                ("OG", u_E),
                ("AC", ACAW_uE), #  * (1/(1-gamma)) / ACAW_uE.sum()
                ("GC", GCAW_uE) #  * (1/(1-gamma)) / GCAW_uE.sum()
    ]
    # data = {
    #     'OG': [],
    #     'AC': [],
    #     'GC': [],
    # }
    data = []
    for u in method_and_result:
        error = print_resutls(mdp, u[1] * (1/(1-gamma)) / u[1].sum(), norm)
        # cu, syed, l1a, l1b, l1c, l1d,  l2a, l2b, l2c, l2d = error
        # data[u[0]]= {
        #     "cu": cu, "syed": syed,
        #     "l1a": l1a, "l2a": l2a,
        #     "l1b": l1b, "l2b": l2b,
        #     "l1c": l1c, "l2c": l2c,
        #     "l1d": l1d, "l2d": l2d,
        # }
        data.append(error)
    return data
   
def opt_various_estimators(mdp, opt_u, episodes_opt):
    gamma = mdp.gamma
    u_E = mdp.u_E
    ACAW_uE, _ = mdp.FVAC(episodes=episodes_opt, alphW=True, alphR=False)
    GCAW_uE, _ = mdp.FVGC(episodes=episodes_opt, alphW=True, alphR=False)
    ACAW_uE = ACAW_uE* (1/(1-gamma)) / ACAW_uE.sum()
    GCAW_uE = GCAW_uE* (1/(1-gamma)) / GCAW_uE.sum()
    closeness = [
        [np.linalg.norm(u_E - opt_u, ord=1), np.linalg.norm(u_E - opt_u, ord=2)],
        [np.linalg.norm(ACAW_uE - opt_u, ord=1), np.linalg.norm(ACAW_uE - opt_u, ord=2)],
        [np.linalg.norm(GCAW_uE - opt_u, ord=1), np.linalg.norm(GCAW_uE - opt_u, ord=2)]
    ]
    return closeness
   

    def alpha_hat(self, episodes):
        freq = np.zeros(self.n)
        for episode in episodes:
            for sample in episode:
                (S, A, R) = sample
                if self.p_0[S] > 0:
                    freq[S] += 1
        return (freq / freq.sum())

    def FVAC(self, episodes, alphW=False, alphR=False):
        assert not (alphW and alphR)
        # variable setup
        gamma = self.gamma
        num_actions = self.m
        num_states = self.n
        ep_length = len(episodes[0])
        num_eps = len(episodes)
        u = np.zeros((num_actions, num_states))
        chunk_count = 0
        alpha_prob = self.p_0
        # start_test = if alpha_prob[][0, 1, 2]

        # alpha correction count if needed
        if alphR:
            freq = self.alpha_hat(episodes)
        def correction(startstate):
            if alphR:
                return self.p_0[startstate] / (freq[startstate])
            elif alphW:
                return alpha_prob[startstate]
            else:
                return 1

        # sampling algorithm
        for k, tau in enumerate(episodes):
            if k == num_eps:
                break
            for i, pair in enumerate(tau):
                if i == ep_length:
                    break
                a, init_s = pair[1], pair[0] #check this out
                if alpha_prob[init_s]:
                    chunk_count += 1
                    # print(f'{i}:YES: {k, states}')
                    for g, pair_prime in enumerate(tau[i:]):
                        #check this out
                        action, states = int(pair_prime[1]), pair_prime[0]
                        # print(f"mu[{action}, {states}] += {alpha_prob[init_s] * gamma**g}") 
                        u[action, states] += correction(init_s) * (gamma**g)
        return u / chunk_count, chunk_count

    def FVGC(self, episodes, alphW=False, alphR=False):
        assert not (alphW and alphR)
        # variable setup
        gamma = self.gamma
        num_actions = self.m
        num_states = self.n
        mu = np.zeros((num_actions, num_states))
        chunk_count = 0
        alpha_prob = self.p_0

        # alpha correction count if needed
        if alphR:
            freq = self.alpha_hat(episodes)
        def correction(startstate):
            if alphR:
                return self.p_0[startstate] / (freq[startstate])
            elif alphW:
                return alpha_prob[startstate]
            else:
                return 1

        # sampling algorithm
        for k,tau in enumerate(episodes):
            for i, pair in enumerate(tau):
                if alpha_prob[pair[0]]:
                    chunk_count += 1
                    init_s = pair[0]
                    k = i
                    while gamma > np.random.random():
                        if k >= len(tau):
                            break
                        pair = tau[k]
                        state, action = pair[0], pair[1]
                        mu[action, state] += correction(init_s) * 1
                        k += 1
        return mu / chunk_count, chunk_count

    def LSTD_muEstimator(self, episodes: list):
        # initialize matrices for least squares computation
        A_matrix = np.zeros((self.k, self.k), dtype='float64')
        b = np.zeros(self.k, dtype='float64')

        # repeat for episodes
        for episode in episodes:
                (S, A, R) = episode[0]
                for sample in episode[1:]:
                    # grab the next state/action
                    (S_prime, A_prime, R) = sample

                    # add to A_matrix and b for least squares computation
                    # dont change this is correct for my python version
                    A_matrix = A_matrix + np.outer(self.phi[A, S], self.phi[A,S] - self.gamma * self.phi[A_prime, S_prime])
                    b = b + np.outer(self.phi[A_prime, S_prime], self.phi[A, S])

                    # b = b + np.outer(self.phi[A, S], self.phi[A_prime, S_prime])

                    # initialize values at time t for next iteration
                    S = S_prime
                    A = A_prime

        A_matrix = A_matrix.astype(np.float64) + .001 * np.eye(A_matrix.shape[0])
        w = np.linalg.solve(A_matrix, b)
        mu = (self.phi.reshape(-1, self.phi.shape[-1]) @ w).mean(axis=0)
        mu = mu.reshape(10, 6)
        # pprint.pprint(mu)
        # exit()
        return mu

def main():
    n = 2
    gamma = 0.9 # discount factor
    m = 2 # number of actions
    # transition probability P_(m*(n*n))
    transition = np.array([[[.8, .2], [0.5, 0.5]], [[.1, .9], [0.5, .5]]])
    # initial state distribution p_0(n)
    p_0 = np.zeros(n)
    p_0[0] = 1
    # rewards = get_random_reward(m, n) # doesn't matter at this point
    rewards = np.array([[0.1, 0.92],[1, 0.02]])
    # features matrix phi(m*(n*k))
    k = 4
    phi = np.zeros([m, n, k])
    phi[0, 0] = np.array([1, 0, 0, 0])
    phi[0, 1] = np.array([0, 1, 0, 0])
    phi[1, 0] = np.array([0, 0, 1, 0])
    phi[1, 1] = np.array([0, 0, 0, 1])
    # phi[0, 0] = np.array([1, 0])
    # phi[0, 1] = np.array([0, 1])
    # phi[1, 0] = np.array([1, 0])
    # phi[1, 1] = np.array([0, 1])
    # phi = np.repeat(rewards[:, :, np.newaxis], k, axis=2)

    mdp = MDP(n, m, k, transition, phi, p_0, gamma)
    # using value iteration
    # optimal_policy = mdp.value_iteration(rewards)
    opt_occ_freq, opt_val = mdp.dual_lp(rewards)
    opt_policy = mdp.u_to_policy(opt_occ_freq)
    opt_u = mdp.policy_to_u(opt_policy)
    num_episodes = 100
    episodes_len = 10
    epsilon = 10.1
    # demonstrations is the experiment that the optimal policy is obtained from
    demonstrations, returns_opt = create_samples(m, n, transition, rewards, gamma,
                                opt_policy, num_episodes, episodes_len)
    # passing the samples to the mdp object
    mdp.get_u_E(demonstrations)
    # print(mdp.FVAC(episodes=demonstrations))
    u_E = mdp.u_E
    # uncomment this line to set u^E to the optimal u
    # mdp.u_E = opt_occ_freq
    # import ipdb;ipdb.set_trace()
    print(mdp.max_ent_irl(0.9, 100))

    # using the dual of minimization problem
    # primal_LP = mdp.primal_lp(rewards)
    # Abbeels_vi_u, Abbeel_vi_reward, Abbeel_vi_opt_val = mdp.Abbeel_dual_vi()
    Abbeels_lp_u_mat, Abbeel_reward_mat, Abbeel_lp_opt_val_mat = mdp.\
                                            Abbeel_dual_lp_mat()
    # Abbeels_prob_simplex, Abbeel_prob_reward, simplex_opt_val =
    # mdp.Abbeel_prob_simplex()
    Syeds_u_mat, Syed_opt_val_mat = mdp.Syed_mat()
    consistent_u_u_mat, const_u_u_opt_val_mat = mdp.Cons_u_optimal_u_mat()
    consistent_r_l1_mat, const_r_l1_opt_val_mat = mdp.Cons_r_optimal_u_l1_mat(epsilon)
    consistent_r_l2_mat, const_r_l2_opt_val_mat = mdp.Cons_r_optimal_u_l2_mat(epsilon)
    # Abbeels_lp_u, Abbeel_reward, Abbeel_lp_opt_val =
    # mdp.Abbeel_consistent_ru()

    print("="*40)
    print("=" * 11 + "optimal values" + "=" * 11)
    print("="*40)
    print("Abbeel lp\t", round(Abbeel_lp_opt_val_mat, 3))
    print("cons u\t\t", round(const_u_u_opt_val_mat, 3))
    print("cons r l2\t", round(const_r_l2_opt_val_mat, 3))
    print("Syed\t\t", round(Syed_opt_val_mat, 3))
    print("cons r l1\t", round(const_r_l1_opt_val_mat, 3))
    print("="*40)
    print("=" * 11 + "optimal solutions" + "=" * 11)
    print("="*40)
    print("True Optimal u\n", opt_occ_freq)
    print("Expert's u\n", u_E)
    # print("Primal\n", primal_LP)
    # print("Abbeel dual vi\n", Abbeels_vi_u)
    print("Abbeel lp\n", Abbeels_lp_u_mat)
    # print("Abbeel probability simplex\n", Abbeels_prob_simplex)
    print("cons U's u\n", consistent_u_u_mat)
    print("cons R's u l2\n", consistent_r_l2_mat)
    print("Syed's u\n", Syeds_u_mat)
    print("cons R's u l1\n", consistent_r_l1_mat)
    print("="*40)

    # u_opt = mdp.policy_to_u(optimal_policy)
    print("=" * 6, "True Return (u^T r_{True})", "=" * 6)
    print("="*40)
    print("optimal u\t\t", mdp.get_return(rewards, opt_occ_freq))
    # print("Abbeel dual vi\t\t\t", 
                        # mdp.get_return(rewards, Abbeels_vi_u))
    # print("Abbeel dual lp\t\t\t", 
                        # mdp.get_return(rewards, Abbeels_lp_u))
    # print("Abbeel prob\t\t\t", 
                        # mdp.get_return(rewards, Abbeels_prob_simplex))
    print("Abbeel lp\t\t", mdp.get_return(rewards, Abbeels_lp_u_mat))
    print("consistent U\t\t", mdp.get_return(rewards, consistent_u_u_mat))
    print("consistent R l2\t\t", mdp.get_return(rewards, consistent_r_l2_mat))
    print("Syed\t\t\t", mdp.get_return(rewards, Syeds_u_mat))
    print("consistent R l1\t\t", mdp.get_return(rewards, consistent_r_l1_mat))

    opt_policy, optimal_value = mdp.value_iteration(rewards)
    # plot_area(n, m, k, gamma, transition, p_0, phi, mdp.u_E, mdp.mu_E, optimal_value)
    # solve_inqualities(n, m, k, gamma, transition, p_0, phi, mdp.u_E, mdp.mu_E)


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
        U_hat = [np.zeros(m * n)]
        counter = 0

        # for i in range(1, num_iter):
        while(True):
            counter += 1
            model = gp.Model("mdp")
            u = model.addVars(m * n, name="u", lb=0)
            t = model.addVar(name="t", lb=-GRB.INFINITY)
            model.setObjective(t, GRB.MINIMIZE)

            # t >= || u - hat_u ||_inf \forall hat_u
            model.addConstrs((t >= u[i] - U_hat[j][i]
                                for i in range(m * n))
                                for j in range(U_hat.shape[0])),
                                name="C0")

            # A^T u = p_0
            model.addConstrs((  quicksum((I[j] - gamma*P[j])[k,i] * u[j,k]
                            for k in range(n)
                            for j in range(m) )
                            == p_0[i]
                            for i in range(n) ),
                            name="C1")

            model.write("../../files/" + method + ".lp")
            model.Params.OutputFlag = 0
            model.optimize()
            t1 = model.objVal
            u_vec = self.gurobi_to_numpy(model, m * n)
            
        return u_vec
