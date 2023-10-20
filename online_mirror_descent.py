from dataclasses import dataclass

import gym
import numpy as np
from gym import spaces
from scipy.special import kl_div

_EPSILON = 10**(-25)


class MirrorDescent:
    """
    Class to compute Online Mirror Descent with changing transition costs
    """

    def __init__(self, env, n_agents=100, lr= 0.01, reward_type='entropy_max'):
        self.env = env
        self.lr = lr
        
        # useful as a shortcut
        self.S = env.size * env.size
        self.A = env.action_space.n
        self.N_steps = env.max_steps
        
        # number of agents to observe
        self.n_agents = n_agents

        # state action counts
        self.n_counts = np.zeros((self.S, self.A))
        self.m_counts = np.zeros((self.S, self.A, self.S))

        # initial probability transition
        self.P = np.ones((self.S, self.A, self.S))/self.S
        # COMMENT UPPER LINE AND UNCOMMENT BOTTOM LINE FOR TEST
        self.trueP = self.env.P(self.env.p)

        # initial policy
        self.policy = np.ones((self.N_steps, self.S, self.A))/self.A

        # initial state distribution sequence
        self.mu = np.zeros((self.N_steps, self.S))

        # initial state-action value function
        self.Q = np.zeros((self.N_steps, self.S, self.A))

        # inital algorithm count step
        self.count_step = 0

        # set target
        self.target = self.S - 1

        # count number of couples (s,a) visited
        self.n_state_action_visited = []

        # reward type
        self.reward_type = reward_type

        # target matching rho
        if self.reward_type == 'marginal_match':
            self.target_rho = self.env.grid.target_marginal()

        # multiple objectives
        if self.reward_type == 'multi_objectives':
            self.multi_objectives = [(self.env.size-1) * self.env.size, self.env.size-1, (self.env.size-1) * self.env.size + self.env.size-1]

        # noise parameters initialization
        self.noise_params = np.zeros(5)


    def nu0(self):
        return self.env.initial_state_dist()

    def reward(self, mu, x):
        """
        mu = vector of size N (\mu(x) for all n \in [N])
        """
        if self.reward_type == 'entropy_max':
            r_mu = - np.log(mu + _EPSILON)
        elif self.reward_type == 'marginal_match':
            r_mu = np.log(self.target_rho[x] + _EPSILON) - np.log(mu + _EPSILON) - 1
        elif self.reward_type == 'multi_objectives':
            if x in self.multi_objectives:
                r_mu = 2*(1-mu)
            else:
                r_mu = 0
        return r_mu

    def objective_function(self, policy, P_model, step):
        # mu_dist = self.mu_induced(policy, self.trueP)
        mu_dist = self.mu_induced(policy, P_model)

        if self.reward_type == 'entropy_max':
            mu = mu_dist[step,:]
            return 1+np.dot(mu, np.log(mu + _EPSILON))/np.log(self.S - len(self.env.grid.wall_cells)+1)

        elif self.reward_type == 'marginal_match':
            obj = 0
            for x in range(self.S):
                obj += -kl_div(mu_dist[step,x], self.target_rho[x] + _EPSILON)
            return 1-obj

        elif self.reward_type == 'multi_objectives':
            obj = 0
            for x in self.multi_objectives:
                obj -= (1-mu_dist[step,x])**2
            return -obj

    def softmax(self, y, pi):
        """softmax function
        Args:
          y: vector of len |A|
          pi: vector of len |A|
        """
        max_y = max(y)
        exp_y = [np.exp(self.lr * (y[a] - max_y)) for a in range(y.shape[0])]
        norm_exp = sum(exp_y)
        return [l / norm_exp for l in exp_y]

    def policy_from_logit(self, Q, prev_policy):
        """Compute policy from Q function
        """
        policy = np.zeros((self.N_steps, self.S, self.A))
        for n in range(self.N_steps):
            for x in range(self.S):
                policy[n,x,:] = self.softmax(Q[n,x,:], prev_policy[n,x,:])
                # assert np.sum(policy[n,x,:]) == 1,  'policy should sum to 1'
        
        return policy

    def state_action_value(self, mu, policy):
        """
        Computes the state-action value function
        (without updating pi)
        """
        Q = np.zeros((self.N_steps, self.S, self.A))

        reward = np.zeros((self.N_steps, self.S))
        for x in range(self.S):
            reward[:,x] = self.reward(mu[:,x],x)
            Q[self.N_steps-1,x,:] = reward[self.N_steps-1,x]

        for n in range(self.N_steps - 1, 0, -1):
            for x in range(self.S):
                for a in range(self.A):
                    Q[n-1,x,a] = reward[n-1,x] 
                    for x_next in range(self.S):
                        Q[n-1,x,a] += self.P[x,a,x_next] * np.dot(policy[n, x_next,:], Q[n,x_next,:])

        return Q
    
    def state_action_value_and_policy(self, mu, policy):
        """
        Computes the regularized state-action value function
        while updating the policy
        """    
        Q = np.zeros((self.N_steps, self.S, self.A))
        pi = np.zeros((self.N_steps, self.S, self.A))

        reward = np.zeros((self.N_steps, self.S))
        for x in range(self.S):
            reward[:,x] = self.reward(mu[:,x], x)
            Q[self.N_steps-1,x,:] = reward[self.N_steps-1,x]
            self.Q[self.N_steps-1,x,:] += Q[self.N_steps-1,x,:]
            pi[self.N_steps-1,x,:] = self.softmax(self.Q[self.N_steps-1,x,:], policy[self.N_steps-1,x,:])
        
        for n in range(self.N_steps - 1, 0, -1):
            for x in range(self.S):
                for a in range(self.A):
                    Q[n-1,x,a] = reward[n-1,x] 
                    for x_next in range(self.S):
                        
                        Q[n-1,x,a] += self.P[x,a,x_next] * np.dot(pi[n, x_next,:], \
                        -1/self.lr * (np.log(pi[n,x_next,:]+_EPSILON) - np.log(policy[n,x_next,:]+_EPSILON) ) + Q[n,x_next,:])
                # Compute new policy for time step n-1
                self.Q[n-1,x,:] += Q[n-1,x,:]
                pi[n-1,x,:] = self.softmax(self.Q[n-1,x,:], policy[n-1,x,:])
        return pi


    def mu_induced(self, policy, P):
        """
        Computes the state distribution induced by a policy
        """
        mu = np.zeros((self.N_steps, self.S))
        mu[0,:] = self.nu0()
        for n in range(1,self.N_steps):
            for x in range(self.S):
                for x_prev in range(self.S):
                    mu[n, x] += mu[n-1, x_prev] * np.dot(policy[n-1, x_prev, :], P[x_prev, :, x])   

        # np.testing.assert_array_equal(np.sum(mu, axis=1), np.ones(self.N_steps), 'proba density should sum to 1')
        return mu 

    
    def iteration(self, n_iterations, true_model, true_noise, P_model, algo):
        """
        Computes one iteration of OMD with changing transitions
        true_model = if the dynamics are known but not necessarily the noise
        true_noise = if both dynamics and noise are known, therefore P is entirely known
        """
        if self.count_step == 0:
            self.error_steplast = []
            for n in range(self.N_steps):
                self.mu[n,:] = self.nu0() 
                self.sum_Q = self.Q
            if true_noise == True:
                self.P = self.trueP
            elif true_model == True:
                self.P = self.env.P(self.noise_params)
            
        for iter in range(n_iterations):
            print('iteration', iter)
            
            self.count_step += 1
            if algo == 'MD-CURL':
                # 1,2a) Update state-value function and compute policy
                self.policy = self.state_action_value_and_policy(self.mu, self.policy)
            elif algo == 'OMD-MFG':
                # 1b) Update the state-value function
                self.Q = self.state_action_value(self.mu, self.policy)
                # 2b) Compute the policy associated
                self.sum_Q += self.Q
                self.policy = self.policy_from_logit(self.sum_Q, self.policy)
            # 3) Update the probability transitions if needed
            if true_noise == False:
                if true_model == False:
                    self.P = self.estimate_transition()
                else:
                    self.P = self.estimate_noise(self.count_step)
            # 4) Update the state-action distribution
            self.mu = self.mu_induced(self.policy, self.P)
            # 5) Compute objective function value at the last time step
            self.error_steplast.append(self.objective_function(self.policy, P_model, -1))


    def sample_policy(self, n, state):
        return np.random.choice(self.A, p=self.policy[n, state,:])

    def estimate_transition(self):
        """
        Estimate transitions using a policy.
        n_agents = n_steps/max_steps
        """
        n_steps = self.n_agents * self.env.max_steps

        P = np.zeros((self.S, self.A, self.S))

        observation = self.env.reset()
        state = self.env.obs_to_state(observation)
        for n in range(n_steps):
            # 1. Sample an action using the policy
            time_step = n % self.env.max_steps
            action = self.sample_policy(time_step, state)
            # 2. Step in the env using this random action
            observation, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.env.obs_to_state(observation)
            # 3. Update state-action counts
            self.n_counts[state, action] += 1
            self.m_counts[state, action, next_state] += 1
            state = next_state.copy()

            if terminated or truncated:
                observation = self.env.reset()
                state = self.env.obs_to_state(observation)

        for s in range(self.S):
            P[:,:,s] = self.m_counts[:,:,s]/np.maximum(1, self.n_counts)

        n_a_s = np.argwhere(np.sum(P, axis =2) == 0)
        for i in range(len(n_a_s)):
            P[n_a_s[i][0], n_a_s[i][1],:] =  1/self.S 
        self.n_state_action_visited.append(self.S*self.A - len(n_a_s))
        return P
        # np.testing.assert_array_equal(np.sum(self.P, axis=2), np.ones((self.S, self.A)), 'proba kernel should sum to 1')

    def estimate_noise(self, t):
        """
        Estimate noise categorical distribution parameters when we suppose the physical dynamics are known (function g)
        t = episode
        """
        n_steps = self.env.max_steps
        noise_traj = np.zeros(5)

        observation = self.env.reset()
        state = self.env.obs_to_state(observation)
        for n in range(n_steps):
            # 1. Sample an action using the policy
            time_step = n % self.env.max_steps
            action = self.sample_policy(time_step, state)
            # 2. Step in the env using this random action
            observation, reward, terminated, truncated, epsilon = self.env.step(action)
            next_state = self.env.obs_to_state(observation)
            # 3. Append noise
            noise_traj[epsilon] += 1
            # 4. Update state
            state = next_state.copy()

            if terminated or truncated:
                observation = self.env.reset()
                state = self.env.obs_to_state(observation)

        # 4. Update noise parameters
        self.noise_params = (n_steps * t * self.noise_params + noise_traj)/(n_steps * (t+1))

        # 5. Update probability transition kernel
        P = self.env.P(self.noise_params)
        return P


        
