# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:31:44 2020

@author: wyckliffe
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values , print_policy

# this is policy evaluation not optimization

# to compare our results, check the result of the other MC
from monte_carlo import play_game

lr = 0.001
all_actions = ('U', 'D', 'L', 'R')
e = 10e-4 # threshold
gamma = 0.9

def random_action(a, eps=0.1):

    # choose the given action with p = 0.5
    # choose some other a' != a with p = 0.5/3

    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        return np.random.choice(all_actions)



if __name__ == '__main__' :

    grid = standard_grid()

    print("Rewards: ")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
      }
    # initialize theta
    # the model is V^ = that.dot(x) where x = [i , j , i*j , 1] (the 1 is a bias term)

    theta = np.random.randn(4) / 2

    def s_to_x(s) :  # maps states into x's

        return np.array([s[0] - 1, s[1] - 1.5 , s[0]*s[1] - 3, 1])

    # repeat until convergences
    deltas = []
    t = 1.0

    for it in range(20000) :
        if it % 100 == 0 :
            t += 0.01

        alpha = lr / t

        # generate an episode using the policy
        biggest_change = 0
        states_returns = play_game(grid, policy)
        seen_states = set()

        for s , G in states_returns :

            if s not in seen_states :
                old_theta = theta.copy()

                x = s_to_x(s)
                V_hat = theta.dot(x)
                theta += alpha * (G - V_hat) * x  # update gradient descent

                biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
                seen_states.add(s)

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    # optain predicted values
    V = {}
    states = grid.all_states()

    for s in states:
        if s in grid.actions :
            V[s] = theta.dot(s_to_x(s))
        else:
            # this is the terminal state
            V[s] = 0


    print("Predicted Values")
    print_values(V, grid)

    print("Policy")
    print_policy(policy, grid)

