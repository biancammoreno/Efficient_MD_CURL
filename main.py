from FourRooms import FourRoomsEnv
from online_mirror_descent import MirrorDescent
import matplotlib.pyplot as plt
import tikzplotlib
import os
import numpy as np


def color_walls_white(env, mu):
    for coor in env.grid.wall_cells:
        x = coor[1] * env.size + coor[0]
        mu[x] = -1000
    mu = mu.reshape((env.size,env.size))
    mu_masked = np.ma.masked_less(mu, 0)

    return mu_masked

def show_final_distribution(env, mu, max_steps, reward_type, noise_prob, true_model, true_noise, iter, algo, learn):
    # Print final distribution
    for step in [max_steps-1]:
    	mu_to_print = color_walls_white(env, mu[step,:])
    	np.savetxt('mu_values.txt', mu_to_print)	
    	plt.imshow(mu_to_print)
    	plt.colorbar()
    	plt.title('mu')
    	plt.savefig('results/' + reward_type + '_mu_dist_iter_' + str(iter) + '_noise_' + str(noise_prob) + '_step_' + str(step) + '_model_' + str(true_model) + '_noise_' + str(true_noise) + '_algo_' + str(algo) + '_learn_' + str(learn) + '.png')
    	plt.close()

def main(max_steps, noise_prob, noise_type, n_agents, true_model, true_noise, reward_type, n_iterations, algo, learn=True):

    # Create directory to add results if it does not exist
    isExist = os.path.exists('results')
    if not isExist:
        os.makedirs('results')

    # define environment 
    env = FourRoomsEnv(max_steps=max_steps, noise_prob=noise_prob, noise_type=noise_type)
    obs = env.reset()
    model = MirrorDescent(env, n_agents=n_agents, reward_type=reward_type)
    P_model = env.P(env.p) # true probability transition kernel

    if learn == False:
        # learner do not want to learn the external noise: learns the policy as if the model had noise_prob=0, but policy is played in noisy env
        fake_env = FourRoomsEnv(max_steps=max_steps, noise_prob=0)
        obs = fake_env.reset()
        model = MirrorDescent(fake_env, n_agents=n_agents, reward_type=reward_type)


    if n_iterations >= 100:
        # save mu after 10 iterations
        model.iteration(n_iterations=10, true_model=true_model, true_noise=true_noise, P_model=P_model, algo=algo)
        mu = model.mu_induced(model.policy, P_model)
        show_final_distribution(env, mu, max_steps, reward_type, noise_prob, true_model, true_noise, model.count_step, algo, learn)

        # after 50 iterations
        model.iteration(n_iterations=40, true_model=true_model, true_noise=true_noise, P_model=P_model, algo=algo)
        mu = model.mu_induced(model.policy, P_model)
        show_final_distribution(env, mu, max_steps, reward_type, noise_prob, true_model, true_noise, model.count_step, algo, learn)

        # after all iterations
        model.iteration(n_iterations=n_iterations-50, true_model=true_model, true_noise=true_noise, P_model=P_model, algo=algo)
        mu = model.mu_induced(model.policy, P_model)
        show_final_distribution(env, mu, max_steps, reward_type, noise_prob, true_model, true_noise, model.count_step, algo, learn)
    else:
        model.iteration(n_iterations=n_iterations, true_model=true_model, true_noise=true_noise, P_model=P_model, algo=algo)
        mu = model.mu_induced(model.policy, P_model)
        show_final_distribution(env, mu, max_steps, reward_type, noise_prob, true_model, true_noise, model.count_step, algo, learn)


     # Plot the objective function at the last time step per iteration
    plt.loglog(model.error_steplast, label='train noise', c='b')
    plt.xlabel('Iteration')
    plt.title('log regret per iteration')
    plt.legend()
    plt.savefig('results/' + reward_type + '_objective_function_noise_' + str(noise_prob) + '_step_' + str(max_steps) + '_model_' + str(true_model) + '_noise_' + str(true_noise)  + '_algo_' + str(algo)+ '_learn_' + str(learn) + '.png')
    plt.close()



if __name__ == '__main__':
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_steps', type=int, required=True, help='N: length of an episode')
    parser.add_argument('--noise_prob', type=float, required=True, help='noise probability')
    parser.add_argument('--noise_type', type=str, required=True, help='type of noise: central or up')
    parser.add_argument('--n_agents', type=int, required=True, help='number of agents')
    parser.add_argument('--true_model', type=ast.literal_eval, required=True, help='True or False: if dynamics g_n are known by agent')
    parser.add_argument('--true_noise', type=ast.literal_eval, required=True, help='True or False: if external noise distributions h_n are known by agent')
    parser.add_argument('--reward_type', type=str, required=True, help='type of reward: entropy_max or multi_objectives')
    parser.add_argument('--n_iterations', type=int, required=True, help='number of iterations')
    parser.add_argument('--algo', type=str, required=True, help='algorithm to run: OMD-MFG or MD-CURL')
    parser.add_argument('--learn', type=ast.literal_eval, required=False, help='True or False: if learner will learn the external noise or not')
    args = parser.parse_args()

    main(max_steps=args.max_steps, noise_prob=args.noise_prob, noise_type=args.noise_type, n_agents=args.n_agents, true_model=args.true_model, true_noise=args.true_noise, 
    reward_type=args.reward_type, n_iterations=args.n_iterations, algo=args.algo, learn=args.learn)
