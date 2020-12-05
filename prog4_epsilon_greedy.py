
# Created by John C.S. Lui     Date: April 10, 2020

# fork of prog1_random_selection.py
# Homework 4
# Baseline Epsilon Epsilon-Greedy Arm Selection


import numpy as np
from scipy.stats import bernoulli  # import bernoulli
import matplotlib.pyplot as plt

Num_of_Arms = 4      # number of arms

#  input parameters 
winning_parameters = np.array([0.2, 0.3, 0.85, 0.9], dtype=float)
max_prob = 0.9				      # record the highest probability of winning for all arms
optimal_arm = 3					      # index for the optimal arm
T = 1000					      # number of rounds to simulate
total_iteration = 200				      # number of iterations to the MAB simulation

reward_round_iteration = np.zeros((T), dtype=float)     # reward in each round average by # of iteration
accumulated_reward_iteration = np.zeros((T), dtype=float)
arm_selected_iteration = np.zeros([Num_of_Arms,T], dtype=int)
upper_bound_regret_iteration = np.zeros((T), dtype=float)

# Go through T rounds
for iteration_count in range(total_iteration):
    emperical_estimates = np.zeros([Num_of_Arms], dtype=float)
    accumu_t = 0
    opt_arm = 0
    for round in range(T):
        t = round + 1
        c = np.random.random_sample()
        epsilon = 2
        if (c < epsilon):
            select_arm = np.mod(round+1, Num_of_Arms+1)-1
        else:
            select_arm = np.argmax(emperical_estimates)
            best_arm = select_arm
        
        arm_selected_iteration[select_arm][round] += 1
        reward = bernoulli.rvs(winning_parameters[select_arm]) 
        accumu_t += reward
        reward_round_iteration[round] += reward
        upper_bound_regret_iteration[round] += np.power(t, 2/3)*np.power(Num_of_Arms*np.log(t), 1/3)
        emperical_estimates[select_arm] += reward
        accumulated_reward_iteration[round] += accumu_t


# compute average reward for each round

average_reward_in_each_round = np.zeros(T, dtype=float)

for round in range(T):
   average_reward_in_each_round[round] = float(reward_round_iteration[round])/float(total_iteration)
   total_arm_iter = 0.0
   for arm in range(Num_of_Arms):
       total_arm_iter += arm_selected_iteration[arm][round]
   for arm in range(Num_of_Arms):
       arm_selected_iteration[arm][round] /= total_arm_iter

upper_bound_regret_iteration /= total_iteration
accumulated_reward_iteration /= total_iteration
# Let generate X and Y data points to plot it out
# Let Z be upper-bound of regret to plot it out

cumulative_optimal_reward = 0.0
cumulative_reward = 0.0

X = np.zeros (T, dtype=int)
Y = np.zeros (T, dtype=float)
Z = np.zeros (T, dtype=float)
for round in range(T):
	X[round] = round
	cumulative_optimal_reward += max_prob
	cumulative_reward += average_reward_in_each_round[round]
	Y[round] = cumulative_optimal_reward - cumulative_reward

fig, axs = plt.subplots(4)
fig.suptitle('Performance of Baseline Epsilon-Greedy Arm Selection')
fig.subplots_adjust(hspace=0.5)
fig.set_figheight(30)
axs[0].plot(X,Y, color = 'red', label='Regret of RSP')
axs[0].set(xlabel='round number', ylabel='Regret')
axs[0].grid(True)
axs[0].legend(loc='upper left')
axs[0].set_xlim(0,T)
axs[0].set_ylim(0,1.1*(cumulative_optimal_reward - cumulative_reward))
axs[1].plot(X, accumulated_reward_iteration, color = 'black', label='accumulated average reward')
axs[1].set(xlabel='round number', ylabel='Average Reward per round')
axs[1].grid(True)
axs[1].legend(loc='upper left')
axs[1].set_xlim(0,T)
axs[1].set_ylim(0,T)
axs[2].plot(X, upper_bound_regret_iteration, color = 'red', label='upper bound')
axs[2].set(xlabel='round number', ylabel='Upper bound')
axs[2].grid(True)
axs[2].legend(loc='upper left')
axs[2].set_xlim(0,T)
for arm in range(Num_of_Arms):
    axs[3].plot(X, arm_selected_iteration[arm], label='arm '+str(arm+1))
axs[3].set(xlabel='round number', ylabel='fraction of selected index')
axs[3].grid(True)
axs[3].legend(loc='upper left')
axs[3].set_xlim(0,T)
axs[3].set_ylim(0, 1.1)
plt.savefig("prog5_figure.png")
plt.show()
