# Efficient Model-Based Concave Utility Reinforcement Learning through Greedy Mirror Descent
Code for running experiments on the article: Efficient Model-Based Concave Utility Reinforcement Learning through Greedy Mirror Descent.

To run an experiment save all files to a folder, create a virtual environment, install the requirements in `requirements.txt` and execute

`python main.py --max_steps 40 --noise_prob 0.1 --noise_type 'central' --n_agents 10 --true_model True --true_noise True --reward_type 'entropy_max' --n_iterations 10 --algo 'MD-CURL'`

For an explanation of the parameters execute

`python main.py -h` 

A new directory `results` is created to save images and graphs
