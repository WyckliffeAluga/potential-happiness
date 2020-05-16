# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:08:09 2020

@author: wyckliffe
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values , print_policy
from td0_prediction import play_game, e, gamma, alpha, all_actions




# this is only policy evaluation and not optimization

class Model:

    def __init__(self) :

        self.theta = np.random.randn(4)

    def s_to_x(self, s) :

        return np.array([s[0] - 1, s[1] - 1.5 , s[0]*s[1] - 3, 1])

    def predict(self, s) :
        x = self.s_to_x(s)

        return self.theta.dot(x)

    def gradient(self, s) :

        return self.s_to_x(s)


if __name__ == '__main__':


  grid = standard_grid()

  # print rewards

  print("rewards")

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

  model = Model()
  deltas = []


  k = 1.0

  for it in range(20000):
    if it % 10 == 0:
      k += 0.01
    alpha = alpha/k
    biggest_change = 0

    states_rewards = play_game(grid, policy)

    for t in range(len(states_rewards) - 1):

      s, _ = states_rewards[t]
      s2, r = states_rewards[t+1]


      old_theta = model.theta.copy()

      if grid.is_terminal(s2):
        target = r
      else:
        target = r + gamma*model.predict(s2)

      model.theta += alpha*(target - model.predict(s))*model.gradient(s)

      biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())

    deltas.append(biggest_change)

  plt.plot(deltas)
  plt.show()

  # obtain predicted values
  V = {}
  states = grid.all_states()

  for s in states:
    if s in grid.actions:
      V[s] = model.predict(s)
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)