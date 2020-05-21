# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:25:35 2020

@author: wyckliffe
"""


import gym
import os
import sys
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from cart_pole_v2_q_learning_with_bins import plot_running_average


class Layer:

  def __init__(self, m1, m2, f=T.tanh, use_bias=True):

    self.w = theano.shared(np.random.randn(m1, m2) * np.sqrt(2 / m1))
    self.params = [self.w]
    self.use_bias = use_bias

    if use_bias:
      self.b = theano.shared(np.zeros(m2))
      self.params += [self.b]

    self.f = f

  def forward(self, x):

    if self.use_bias:
      a = x.dot(self.w) + self.b
    else:
      a = x.dot(self.w)

    return self.f(a)


# approximates pi(a | s)
class Policy:

  def __init__(self, D, K, layer_sizes):

    # learning rate and other hyperparams
    lr = 1e-4

    # create the graph
    # K = number of actions

    self.layers = []
    m1 = D

    for M2 in layer_sizes:
      layer = Layer(m1, m2)
      self.layers.append(layer)
      m1 = m2

    # final layer
    layer = Layer(m1, K, lambda x: x, use_bias=False)

    self.layers.append(layer)

    # get all params for gradient later
    params = []
    for layer in self.layers:
      params += layer.params

    # inputs and targets
    x = T.matrix('x')
    actions = T.ivector('actions')
    advantages = T.vector('advantages')

    # calculate output and cost
    z = x

    for layer in self.layers:
      z = layer.forward(z)
    action_scores = z

    p_a_given_s = T.nnet.softmax(action_scores)

    selected_probabilities = T.log(p_a_given_s[T.arange(actions.shape[0]), actions])
    cost = -T.sum(advantages * selected_probabilities)

    # specify update rule
    grads = T.grad(cost, params)
    updates = [(p, p - lr*g) for p, g in zip(params, grads)]

    # compile functions
    self.train_ = theano.function(
      inputs=[x, actions, advantages],
      updates=updates,
      allow_input_downcast=True
    )
    self.predict_ = theano.function(
      inputs=[x],
      outputs=p_a_given_s,
      allow_input_downcast=True
    )

  def partial_fit(self, x, actions, advantages):

    x = np.atleast_2d(x)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)

    self.train_(x, actions, advantages)

  def predict(self, x):

    x = np.atleast_2d(x)
    return self.predict_(x)

  def sample_action(self, x):

    p = self.predict(x)[0]
    non_nans = np.all(~np.isnan(p))
    assert(non_nans)
    return np.random.choice(len(p), p=p)


# approximates V(s)
class Value:

  def __init__(self, D, layer_sizes):
    # constant learning rate is fine
    lr = 1e-4

    # create the graph
    self.layers = []
    m1 = D

    for m2 in layer_sizes:

      layer = Layer(m1, m2)
      self.layers.append(layer)
      m1 = m2

    # final layer
    layer = Layer(m1, 1, lambda x: x)
    self.layers.append(layer)


    params = []
    for layer in self.layers:
      params += layer.params

    # inputs and targets
    x = T.matrix('x')
    y = T.vector('y')

    # calculate output and cost
    z = x

    for layer in self.layers:
      z = layer.forward(z)
    y_ = T.flatten(z)
    cost = T.sum((y - y_)**2)

    # specify update rule
    grads = T.grad(cost, params)
    updates = [(p, p - lr*g) for p, g in zip(params, grads)]

    # compile functions
    self.train_ = theano.function(
      inputs=[x, y],
      updates=updates,
      allow_input_downcast=True
    )
    self.predict_ = theano.function(
      inputs=[x],
      outputs=y_,
      allow_input_downcast=True
    )

  def partial_fit(self, x, y):

    x = np.atleast_2d(x)
    y = np.atleast_1d(y)

    self.train_(x, y)

  def predict(self, x):

    x = np.atleast_2d(x)
    return self.predict_(x)


def td_episode(env, policy_model, value_model, gamma):

  observation = env.reset()
  done = False
  total_reward = 0
  iterations = 0

  while not done and iters < 200:

    action = policy_model.sample_action(observation)
    previous_observation = observation
    observation, reward, done, info = env.step(action)

    if done:
      reward = -200
      print("Fallen....!")

    # update the models
    next_value = value_model.predict(observation)
    G = reward + gamma*np.max(next_value)
    advantage = G - value_model.predict(previous_observation)
    policy_model.partial_fit(previous_observation, action, advantage)
    value_model.partial_fit(previous_observation, G)

    if reward == 1: # if we changed the reward to -200
      total_reward += reward
    iterations += 1

  return total_reward


def mc_episode(env, policy_model, value_model, gamma):

  observation = env.reset()
  done = False
  total_reward = 0
  iterations = 0

  states = []
  actions = []
  rewards = []
  #env.render()
  reward = 0
  while not done and iterations < 200:

    action = policy_model.sample_action(observation)

    states.append(observation)
    actions.append(action)
    rewards.append(reward)

    previous_observation = observation
    observation, reward, done, info = env.step(action)

    if done:
      reward = -200

    if reward == 1: # if we changed the reward to -200
      total_reward += reward
    iterations += 1

  # save the final (s,a,r) tuple
  action = policy_model.sample_action(observation)

  states.append(observation)
  actions.append(action)
  rewards.append(reward)

  returns = []
  advantages = []
  G = 0
  for s, r in zip(reversed(states), reversed(rewards)):
    returns.append(G)
    advantages.append(G - value_model.predict(s)[0])
    G = r + gamma*G
  returns.reverse()
  advantages.reverse()

  # update the models
  policy_model.partial_fit(states[1:], actions[1:], advantages[1:])
  value_model.partial_fit(states, returns)

  return total_reward


def main():

  env = gym.make('CartPole-v0')
  D = env.observation_space.shape[0]
  K = env.action_space.n

  policy = Policy(D, K, [])
  value = Value(D, [10])
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  N = 1000
  total_rewards = np.empty(N)
  costs = np.empty(N)

  for n in range(N):
    #env.render()
    total_reward = mc_episode(env, policy, value, gamma)
    total_rewards[n] = total_reward
    if n % 100 == 0:
      print("episode:", n, "total reward:", total_reward, "avg reward (last 100):", total_rewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", total_rewards[-100:].mean())
  print("total steps:", total_rewards.sum())

  plt.plot(total_rewards)
  plt.title("Rewards")
  plt.show()

  plot_running_average(total_rewards)


if __name__ == '__main__':
  main()