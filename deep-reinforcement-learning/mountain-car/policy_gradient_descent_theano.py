# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:07:12 2020

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

import mountain_car_v1_q_learning
from mountain_car_v1_q_learning import plot_cost_to_go, Transformer, Model, plot_running_average


# helper for adam optimizer

def adam(cost, params, lr0=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):

  grads = T.grad(cost, params)
  updates = []
  time = theano.shared(0)
  new_time = time + 1
  updates.append((time, new_time))

  lr = lr0*T.sqrt(1 - beta2**new_time) / (1 - beta1**new_time)
  for p, g in zip(params, grads):

    m = theano.shared(p.get_value() * 0.)
    v = theano.shared(p.get_value() * 0.)

    new_m = beta1*m + (1 - beta1)*g
    new_v = beta2*v + (1 - beta2)*g*g
    new_p = p - lr*new_m / (T.sqrt(new_v) + eps)

    updates.append((m, new_m))
    updates.append((v, new_v))
    updates.append((p, new_p))

  return updates


# so you can test different architectures
class Layer:

  def __init__(self, m1, m2, f=T.nnet.relu, use_bias=True, zeros=False):

    if zeros:
      self.w = theano.shared(np.zeros((m1, m2)))
    else:
      self.w = theano.shared(np.random.randn(m1, m2) * np.sqrt(2. / m1))

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

  def __init__(self, D, ft, layer_sizes=[]):

    self.ft = ft

    ##### hidden layers #####
    m1 = D
    self.layers = []
    for M2 in layer_sizes:

      layer = Layer(m1, m2)
      self.layers.append(layer)
      m1 = m2

    # final layer mean
    self.mean_layer = Layer(m1, 1, lambda x: x, use_bias=False, zeros=True)

    # final layer variance
    self.var_layer = Layer(m1, 1, T.nnet.softplus, use_bias=False, zeros=False)

    # get all params for gradient later
    params = self.mean_layer.params + self.var_layer.params
    for layer in self.layers:
      params += layer.params

    # inputs and targets
    x = T.matrix('x')
    actions = T.vector('actions')
    advantages = T.vector('advantages')
    target_value = T.vector('target_value')

    # get final hidden layer
    z = x
    for layer in self.layers:
      z = layer.forward(z)

    mean = self.mean_layer.forward(z).flatten()
    var = self.var_layer.forward(z).flatten() + 1e-5 # smoothing


    def log_pdf(actions, mean, var):

      k1 = T.log(2*np.pi*var)
      k2 = (actions - mean)**2 / var

      return -0.5*(k1 + k2)

    def entropy(var):

      return 0.5*T.log(2*np.pi*np.e*var)

    log_probabilities = log_pdf(actions, mean, var)
    cost = -T.sum(advantages * log_probabilities + 0.1*entropy(var))
    updates = adam(cost, params)

    # compile functions
    self.train_ = theano.function(
      inputs=[x, actions, advantages],
      updates=updates,
      allow_input_downcast=True
    )


    self.predict_ = theano.function(
      inputs=[x],
      outputs=[mean, var],
      allow_input_downcast=True
    )

  def partial_fit(self, x, actions, advantages):

    x = np.atleast_2d(x)
    x = self.ft.transform(x)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)

    self.train_(x, actions, advantages)

  def predict(self, x):

    x = np.atleast_2d(x)
    x = self.ft.transform(x)

    return self.predict_op(x)

  def sample_action(self, x):

    pred = self.predict(x)
    mu = pred[0][0]
    v = pred[1][0]
    a = np.random.randn()*np.sqrt(v) + mu

    return min(max(a, -1), 1)


# approximates V(s)
class Value:

  def __init__(self, D, ft, layer_sizes=[]):
    self.ft = ft

    # create the graph
    self.layers = []
    m1 = D
    for M2 in layer_sizes:

      layer = Layer(m1, m2)
      self.layers.append(layer)
      m1 = m2

    # final layer
    layer = Layer(m1, 1, lambda x: x)

    self.layers.append(layer)

    # get all params for gradient later
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
    y_ = T.flatten(y)
    cost = T.sum((y - y_)**2)

    # specify update rule
    updates = adam(cost, params, lr0=1e-1)

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
    x = self.ft.transform(x)
    y = np.atleast_1d(y)
    self.train_op(x, y)

  def predict(self, x):

    x = np.atleast_2d(x)
    x = self.ft.transform(x)

    return self.predict_op(x)


def td_episode(env, policy, value, gamma, train=True):

  observation = env.reset()
  done = False
  total_reward = 0
  iterations = 0

  while not done and iters < 2000:

    action = policy.sample_action(observation)
    previous_observation = observation
    observation, reward, done, info = env.step([action])

    total_reward += reward

    # update the models
    if train:
      next_value = value.predict(observation)
      G = reward + gamma*next_value
      advantage = G - value.predict(previous_observation)
      policy.partial_fit(previous_observation, action, advantage)
      value.partial_fit(previous_observation, G)

    iterations += 1

  return total_reward


def main():
  env = gym.make('MountainCarContinuous-v0')
  ft = Transformer(env, n_components=100)
  D = ft.dimensions
  policy = Policy(D, ft)
  value = Value(D, ft)
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  N = 50
  total_rewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    total_reward = td_episode(env, policy, value, gamma)
    total_rewards[n] = total_reward
    if n % 1 == 0:
      print("Episode:", n, "Total reward: %.1f" % total_reward, "avg reward (last 100): %.1f" % total_rewards[max(0, n-100):(n+1)].mean())

  print("Average reward for last 100 episodes:", total_rewards[-100:].mean())

  plt.plot(total_rewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(total_rewards)
  plot_cost_to_go(env, value)


if __name__ == '__main__':
  main()