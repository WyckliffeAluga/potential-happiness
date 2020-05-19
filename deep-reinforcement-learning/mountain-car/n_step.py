# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:38:51 2020

@author: wyckl
"""


# Solve mountain car with  N-step method

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime


import mountain_car_v1_q_learning
from mountain_car_v1_q_learning import plot_cost_to_go, Transformer, Model, plot_running_average


class SGDRegressor:
  def __init__(self, **kwargs):
    self.w = None
    self.lr = 1e-2

  def partial_fit(self, X, Y):
    if self.w is None:
      D = X.shape[1]
      self.w = np.random.randn(D) / np.sqrt(D)
    self.w += self.lr*(Y - X.dot(self.w)).dot(X)

  def predict(self, X):
    return X.dot(self.w)

# replace SKLearn Regressor
mountain_car_v1_q_learning.SGDRegressor = SGDRegressor


def episode(model, eps, gamma, n=5):

  observation = env.reset()
  done = False
  total_reward = 0
  rewards = []
  states = []
  actions = []
  iterations = 0

  # array of [gamma^0, gamma^1, ..., gamma^(n-1)]
  multiplier = np.array([gamma]*n)**np.arange(n)
  # while not done and iters < 200:
  while not done and iterations < 10000:

    action = model.sample_action(observation, eps)

    states.append(observation)
    actions.append(action)

    prev_observation = observation
    observation, reward, done, info = env.step(action)

    rewards.append(reward)

    # update the model
    if len(rewards) >= n:

      return_up_to_prediction = multiplier.dot(rewards[-n:])
      G = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation)[0])
      model.update(states[-n], actions[-n], G)


    total_reward += reward
    iterations += 1

  # empty the cache
  if n == 1:
    rewards = []
    states = []
    actions = []
  else:
    rewards = rewards[-n+1:]
    states = states[-n+1:]
    actions = actions[-n+1:]


  if observation[0] >= 0.5:
    # we actually made it to the goal
    print("made it!")
    while len(rewards) > 0:
      G = multiplier[:len(rewards)].dot(rewards)
      model.update(states[0], actions[0], G)
      rewards.pop(0)
      states.pop(0)
      actions.pop(0)
  else:
    # we did not make it to the goal
    print("didn't make it...")
    while len(rewards) > 0:
      guess_rewards = rewards + [-1]*(n - len(rewards))
      G = multiplier.dot(guess_rewards)
      model.update(states[0], actions[0], G)
      rewards.pop(0)
      states.pop(0)
      actions.pop(0)

  return total_reward


if __name__ == '__main__':
  env = gym.make('MountainCar-v0')
  ft = Transformer(env)
  model = Model(env, ft, "constant")
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 300
  total_rewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    env.render()
    eps = 0.1*(0.97**n)
    total_reward = episode(model, eps, gamma)
    total_rewards[n] = total_reward
    print("episode:", n, "total reward:", total_reward)
  print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
  print("total steps:", -total_rewards.sum())

  plt.plot(total_rewards)
  plt.title("Rewards")
  plt.show()

  plot_running_average(total_rewards)

  # plot the optimal state-value function
  plot_cost_to_go(env, model)