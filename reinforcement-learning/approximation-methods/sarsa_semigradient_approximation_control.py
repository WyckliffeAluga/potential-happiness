# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:34:55 2020

@author: wyckliffe
"""


import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from monte_carlo_optimize import max_dict
from iterative_policy_evaluation import print_values , print_policy
from td0_prediction import random_action



e = 10e-4
gamma = 0.9
alpha = 0.1

all_actions = ('U', 'D' , 'L', 'R')

sa_2_idx = {}
idx = 0

# goal is to find the optimal policy

class Model :

    def __init__(self):

        self.theta = np.random.randn(25) / np.sqrt(25)


    def sa_to_x(self, s, a) :

        return np.array([
            s[0] - 1              if a == 'U' else 0,
            s[1] - 1.5            if a == 'U' else 0,
            (s[0]*s[1] - 3)/3     if a == 'U' else 0,
            (s[0]*s[0] - 2)/2     if a == 'U' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'U' else 0,
            1                     if a == 'U' else 0,
            s[0] - 1              if a == 'D' else 0,
            s[1] - 1.5            if a == 'D' else 0,
            (s[0]*s[1] - 3)/3     if a == 'D' else 0,
            (s[0]*s[0] - 2)/2     if a == 'D' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'D' else 0,
            1                     if a == 'D' else 0,
            s[0] - 1              if a == 'L' else 0,
            s[1] - 1.5            if a == 'L' else 0,
            (s[0]*s[1] - 3)/3     if a == 'L' else 0,
            (s[0]*s[0] - 2)/2     if a == 'L' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'L' else 0,
            1                     if a == 'L' else 0,
            s[0] - 1              if a == 'R' else 0,
            s[1] - 1.5            if a == 'R' else 0,
            (s[0]*s[1] - 3)/3     if a == 'R' else 0,
            (s[0]*s[0] - 2)/2     if a == 'R' else 0,
            (s[1]*s[1] - 4.5)/4.5 if a == 'R' else 0,
            1                     if a == 'R' else 0,
            1
                                                    ])

    def predict(self, s , a) :

        x = self.sa_to_x(s, a)
        return self.theta.dot(x)

    def gradient(self, s, a) :

        return self.sa_to_x(s, a)



def get_Q(model, s):

    Qs = {}

    for a in all_actions :

        q_sa = model.predict(s, a)
        Qs[a] = q_sa
    return Qs

if __name__ == '__main__':

  #  if we use the standard grid, there's a good chance we will end up with
  # suboptimal policies
  # instead, let's penalize each movement so the agent will find a shorter route.
  #
  # grid = standard_grid()
  grid = negative_grid(step_cost=-0.1)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # no policy initialization, derive  policy from most recent Q

  states = grid.all_states()

  for s in states:
    sa_2_idx[s] = {}
    for a in all_actions:
      sa_2_idx[s][a] = idx
      idx += 1

  # initialize model
  model = Model()

  # repeat until convergence
  t = 1.0
  t2 = 1.0
  deltas = []
  for it in range(20000):
    if it % 100 == 0:
      t += 0.01
      t2 += 0.01
    if it % 1000 == 0:
      print("it:", it)
    alpha = alpha / t2

    # instead of 'generating' an epsiode, PLAY
    # an episode within this loop
    s = (2, 0) # start state
    grid.set_state(s)

    # get Q(s) so we can choose the first action
    Qs = get_Q(model, s)

    a = max_dict(Qs)[0]
    a = random_action(a, eps=0.5/t) # epsilon-greedy
    biggest_change = 0

    while not grid.game_over():

      r = grid.move(a)
      s2 = grid.current_state()

      # we need the next action as well since Q(s,a) depends on Q(s',a')
      # if s2 not in policy then it's a terminal state, all Q are 0

      old_theta = model.theta.copy()

      if grid.is_terminal(s2):

        model.theta += alpha*(r - model.predict(s, a))*model.gradient(s, a)

      else:
        # not terminal
        Qs2 = get_Q(model, s2)
        a2 = max_dict(Qs2)[0]
        a2 = random_action(a2, eps=0.5/t) # epsilon-greedy

        # we will update Q(s,a) AS we experience the episode
        model.theta += alpha*(r + gamma*model.predict(s2, a2) - model.predict(s, a))*model.gradient(s, a)

        # next state becomes current state
        s = s2
        a = a2

      biggest_change = max(biggest_change, np.abs(model.theta - old_theta).sum())
    deltas.append(biggest_change)

  plt.plot(deltas)
  plt.show()

  # determine the policy from Q*
  # find V* from Q*
  policy = {}
  V = {}
  Q = {}
  for s in grid.actions.keys():
    Qs = get_Q(model, s)
    Q[s] = Qs
    a, max_q = max_dict(Qs)
    policy[s] = a
    V[s] = max_q

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)