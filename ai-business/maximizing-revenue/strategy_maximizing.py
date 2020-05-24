# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:13:33 2020

@author: wyckliffe
"""


import numpy as np
import random
import matplotlib.pyplot as plt

N = 100000 # number of customers samples
D = 9 # number of strategies

# create simulations
conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
#conversion_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
X = np.array(np.zeros([N, D]))

for n in range(N):
    for d in range(D):
        if np.random.rand() <= conversion_rates[d]:
            X[n, d] = 1


strategies_selected_rs = [] # list of strategies selected by random sampling
strategies_selected_ts = [] # list of strategies selected by thombson sampling

total_reward_rs = 0
total_reward_ts = 0

numbers_of_rewards_got_1 = [0] * D
numbers_of_rewards_got_0 = [0] * D

for n in range(N):
    if n % 1000 == 0:
        print('{:03d}/{:03d} customers simulated'.format(n, N))

    # pick a random strategy
    strategy_rs = random.randrange(D)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs += reward_rs

    # pick a strategy using Thompson sampling
    max_random = 0
    strategy_ts = 0
    for d in range(0, D) :
        random_beta = random.betavariate(numbers_of_rewards_got_1[d] +1, numbers_of_rewards_got_0[d] +1)
        if random_beta > max_random :
            max_random = random_beta
            strategy_ts = d

    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_got_1[strategy_ts] = numbers_of_rewards_got_1[strategy_ts] + 1
    else:
        numbers_of_rewards_got_0[strategy_ts] = numbers_of_rewards_got_0[strategy_ts] + 1

    strategies_selected_ts.append(strategy_ts)
    total_reward_ts += reward_ts

# compute relative and absolute returns
absolute_return = (total_reward_ts - total_reward_rs ) * 100
relative_return =  (total_reward_ts - total_reward_rs ) / total_reward_rs * 100

print('Absolute return: ${:.0f}'.format(absolute_return))
print('Relative return: {:.0f}%'.format(relative_return))

# Plotting the Histogram of Selections
plt.hist(strategies_selected_ts)
plt.title("Histogram of Selections by Thombson Sampling")
plt.xlabel("Strategy")
plt.ylabel("Number of times the strategy was selected")
plt.show()

# Plotting the Histogram of Selections
plt.hist(strategies_selected_rs)
plt.title("Histogram of Selections by Random Sampling")
plt.xlabel("Strategy")
plt.ylabel("Number of times the strategy was selected")
plt.show()


rewards_strategies = [0] * D
for n in range(0, N):
    # Best Strategy
    for i in range(0, D):
        rewards_strategies[i] = rewards_strategies[i] + X[n, i]
    total_reward_bs = max(rewards_strategies)


# Regret of Thompson Sampling
strategies_selected_ts = []
total_reward_ts = 0
total_reward_bs = 0
numbers_of_rewards_1 = [0] * D
numbers_of_rewards_0 = [0] * D
rewards_strategies = [0] * D
regret = []

for n in range(0, N):
    # Thompson Sampling
    strategy_ts = 0
    max_random = 0
    for i in range(0, D):

        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                         numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = i
    reward_ts = X[n, strategy_ts]

    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] = numbers_of_rewards_1[strategy_ts] + 1
    else:
        numbers_of_rewards_0[strategy_ts] = numbers_of_rewards_0[strategy_ts] + 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts = total_reward_ts + reward_ts

    # Best Strategy
    for i in range(0, D):
        rewards_strategies[i] = rewards_strategies[i] + X[n, i]
    total_reward_bs = max(rewards_strategies)

    # Regret
    regret.append(total_reward_bs - total_reward_ts)

# Plotting the Regret Curve
plt.plot(regret)
plt.title('Regret Curve for Thombson sampling')
plt.xlabel('Round')
plt.ylabel('Regret')
plt.show()

# Regret of the Random Strategy
strategies_selected_rs = []
total_reward_rs = 0
total_reward_bs = 0
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
rewards_strategies = [0] * d
regret = []

for n in range(0, N):
    # Random Strategy

    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs = total_reward_rs + reward_rs
    # Best Strategy

    for i in range(0, d):
        rewards_strategies[i] = rewards_strategies[i] + X[n, i]
    total_reward_bs = max(rewards_strategies)

    # Regret
    regret.append(total_reward_bs - total_reward_rs)

# Plotting the Regret Curve
plt.plot(regret)
plt.title('Regret Curve For Random Strategy')
plt.xlabel('Round')
plt.ylabel('Regret')
plt.show()