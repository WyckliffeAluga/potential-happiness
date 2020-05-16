# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:57:12 2020

@author: wyckliffe
"""


import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_optimize import max_dict

gamma = 0.9
all_actions = ('U', 'D', 'L', 'R')

# finding optimal policy and value function using


def random_action(a, eps=0.1):

    # choose the given action with p = 0.5
    # choose some other a' != a with p = 0.5/3

    p = np.random.random()

    if p < (1 - eps):
        return a
    else:
        return np.random.choice(all_actions)


def play_game(grid, policy):

  # returns a list of states and corresponding returns

  s = (2, 0)
  grid.set_state(s)
  a = random_action(policy[s])


  states_actions_rewards = [(s, a, 0)]
  while True:

    r = grid.move(a)
    s = grid.current_state()

    if grid.game_over():
      states_actions_rewards.append((s, None, r))
      break
    else:
      a = random_action(policy[s]) # the next state is stochastic
      states_actions_rewards.append((s, a, r))



  G = 0
  states_actions_returns = []
  first = True

  for s, a, r in reversed(states_actions_rewards):

    if first:
      first = False
    else:
      states_actions_returns.append((s, a, G))

    G = r + gamma*G

  states_actions_returns.reverse()

  return states_actions_returns


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  # grid = standard_grid()
  # try the negative grid too, to see if agent will learn to go past the "bad spot"
  # in order to minimize number of steps
  grid = negative_grid(step_cost=-0.1)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)


  policy = {}

  for s in grid.actions.keys():
    policy[s] = np.random.choice(all_actions)

  # initialize Q(s,a) and returns
  Q = {}
  returns = {}
  states = grid.all_states()

  for s in states:
    if s in grid.actions: # not a terminal state
      Q[s] = {}
      for a in all_actions:
        Q[s][a] = 0
        returns[(s,a)] = []
    else:
      # terminal state or state we can't otherwise get to
      pass

  # repeat until convergence
  deltas = []
  for t in range(5000):
    if t % 100 == 0:
      print(t)

    # generate an episode using pi
    biggest_change = 0
    states_actions_returns = play_game(grid, policy)

    # calculate Q(s,a)
    seen_state_action_pairs = set()

    for s, a, G in states_actions_returns:
      # check if we have already seen s
      sa = (s, a)

      if sa not in seen_state_action_pairs:
        old_q = Q[s][a]
        returns[sa].append(G)
        Q[s][a] = np.mean(returns[sa])
        biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
        seen_state_action_pairs.add(sa)
    deltas.append(biggest_change)

    # calculate new policy pi(s) = argmax[a]{ Q(s,a) }

    for s in policy.keys():
      a, _ = max_dict(Q[s])
      policy[s] = a

  plt.plot(deltas)
  plt.show()

  # find the optimal state-value function
  # V(s) = max[a]{ Q(s,a) }
  V = {}
  for s in policy.keys():
    V[s] = max_dict(Q[s])[1]

  print("final values:")
  print_values(V, grid)
  print("final policy:")
  print_policy(policy, grid)





