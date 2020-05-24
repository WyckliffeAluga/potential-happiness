# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:13:33 2020

@author: wyckliffe
"""


import numpy as np
import random
import matplotlib.pyplot as plt

N = 10000 # number of customers sampled
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

    # pick a random strategy
    strategy_rs = random.randrange(D)
    strategies_selected_rs.append(strategies_selected_rs)
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
