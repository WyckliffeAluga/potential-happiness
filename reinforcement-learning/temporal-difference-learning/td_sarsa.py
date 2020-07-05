# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:11:00 2020

@author: wyckliffe
"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid , negative_grid
from iterative_policy_evaluation import print_values, print_policy
from td0_prediction import random_action
from monte_carlo_optimize import max_dict

e = 10e-4
gamma = 0.9
alpha = 0.1

all_actions = ('U', 'D' , 'L', 'R')


if __name__ == '__main__':
     # fully online we cannot play the game then do the optimization of policy
     # we optimize the policy while playing the game

     # get the negative grid because since we are trying to find the best route and also
     # if we do not penalize we may just spend the rest of the time at one spot
     # grid = standard_grid()
      grid = negative_grid(step_cost=-1)

      # print rewards
      print("rewards:")
      print_values(grid.rewards, grid)

      # no policy initialization, derive  policy from most recent Q

      # initialize Q(s,a)

      Q = {}
      states = grid.all_states()

      for s in states:
        Q[s] = {}
        for a in all_actions:
          Q[s][a] = 0

      # keep track of how many times Q[s] has been updated
      update_counts = {}
      update_counts_sa = {}

      for s in states:
        update_counts_sa[s] = {}

        for a in all_actions:
          update_counts_sa[s][a] = 1.0

      # repeat until convergence
      t = 1.0
      deltas = []

      for it in range(10000):

        if it % 100 == 0:
          t += 1e-2

        if it % 500 == 0:
          print("it:", it)


        s = (2, 0) # start state
        grid.set_state(s)


        a = max_dict(Q[s])[0]
        a = random_action(a, eps=0.5/t)
        biggest_change = 0


        while not grid.game_over():

          r = grid.move(a)
          s2 = grid.current_state()

          # take the next action as episode progresses

          a2 = max_dict(Q[s2])[0]
          a2 = random_action(a2, eps=0.5/t) # epsilon-greedy

          #update Q(s,a)the episode progresses

          old_qsa = Q[s][a]
          Q[s][a] = Q[s][a] + alpha*(r + gamma*Q[s2][a2] - Q[s][a])
          biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

          # we would like to know how often Q(s) has been updated too
          update_counts[s] = update_counts.get(s,0) + 1

          # next state becomes current state
          s = s2
          a = a2

        deltas.append(biggest_change)

      plt.plot(deltas)
      plt.show()

      # determine the policy from Q*
      # find V* from Q*
      policy = {}
      V = {}

      for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

      # what's the proportion of time we spend updating each part of Q?
      print("update counts:")
      total = np.sum(list(update_counts.values()))

      for k, v in update_counts.items():
        update_counts[k] = float(v) / total
      print_values(update_counts, grid)

      print("values:")
      print_values(V, grid)
      print("policy:")
      print_policy(policy, grid)