# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:49:03 2020

@author: wyckliffe
"""



import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid , negative_grid
from iterative_policy_evaluation import print_values, print_policy

e = 10e-4
gamma = 0.9
alpha = 0.1

all_actions = ('U', 'D' , 'L', 'R')

# this is only for policy evaluation and not optimization

def random_action(a, eps=0.1):

    # use epsilon-soft
    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        return np.random.choice(all_actions)


def play_game(grid, policy):

    s = (2, 0)
    grid.set_state(s)
    states_rewards = [(s,0)]

    while not grid.game_over() :
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_rewards.append((s, r))

    return states_rewards


if __name__ == "__main__" :

    grid = standard_grid()

    print("Rewards: ")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
      }

    V = {}
    states = grid.all_states()

    for s in states :
        V[s] = 0

    for it in range(1000) :

        # generate an episode
        states_rewards = play_game(grid, policy)

        # the first tuple (s , r) is the one we start in with 0 reward since there is no reward for starting a game
        # the last (s, r) tuple is the terminal state and final reward
        # the value of the terminal state is 0 by definaiotn

        for t in range(len(states_rewards) - 1):
            s, _ = states_rewards[t]
            s2, r = states_rewards[t + 1]

            # keep updating Vs as the episode continues
            V[s] = V[s] + alpha * r + gamma*(V[s2] - V[s])


    print("Values")
    print_values(V, grid)
    print("Policy")
    print_policy(policy, grid)