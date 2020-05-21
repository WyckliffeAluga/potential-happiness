
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:25:35 2020

@author: wyckliffe
"""


import gym
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from cart_pole_v2_q_learning_with_bins import plot_running_average
tf.disable_v2_behavior()

#Layers
class Layer:

  def __init__(self, m1, m2, f=tf.nn.tanh, use_bias=True):

    self.w = tf.Variable(tf.random.normal(shape=(m1, m2)))
    self.use_bias = use_bias

    if use_bias:
      self.b = tf.Variable(np.zeros(m2).astype(np.float32))
    self.f = f

  def forward(self, x):

    if self.use_bias:
      a = tf.matmul(x, self.w) + self.b
    else:
      a = tf.matmul(x, self.w)
    return self.f(a)


# approximates pi(a | s)

class Policy:

  def __init__(self, D, K, layer_sizes):

    # create the graph
    # K = number of actions

    self.layers = []

    m1 = D
    for m2 in layer_sizes:

      layer = Layer(m1, m2)
      self.layers.append(layer)
      m1 = m2

    # final layer
    layer = Layer(m1, K, tf.nn.softmax, use_bias=False)
    self.layers.append(layer)

    # inputs and targets
    self.x = tf.placeholder(tf.float32, shape=(None, D), name='x')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

    # calculate output and cost
    z = self.x
    for layer in self.layers:
      z = layer.forward(z)
    p_a_given_s = z


    # self.action_scores = action_scores
    self.predict_ = p_a_given_s


    selected_probs = tf.log(
      tf.reduce_sum(
        p_a_given_s * tf.one_hot(self.actions, K),
        reduction_indices=[1]
      )
    )


    cost = -tf.reduce_sum(self.advantages * selected_probs)


    # self.train_ = tf.train.AdamOptimizer(1e-1).minimize(cost)
    self.train_ = tf.train.AdagradOptimizer(1e-1).minimize(cost)
    # self.train_ = tf.train.MomentumOptimizer(1e-4, momentum=0.9).minimize(cost)
    # self.train_ = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, x, actions, advantages):
    x = np.atleast_2d(x)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.session.run(
      self.train_,
      feed_dict={
        self.x: x,
        self.actions: actions,
        self.advantages: advantages,
      }
    )

  def predict(self, x):

    x = np.atleast_2d(x)
    return self.session.run(self.predict_, feed_dict={self.x: x})

  def sample_action(self, x):
    p = self.predict(x)[0]
    return np.random.choice(len(p), p=p)


# approximates V(s)

class Value:

  def __init__(self, D, layer_sizes):
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

    # inputs and targets
    self.x = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    # calculate output and cost
    z = self.x
    for layer in self.layers:
      z = layer.forward(z)
    y_ = tf.reshape(z, [-1]) # the output
    self.predict_ = y_

    cost = tf.reduce_sum(tf.square(self.y - y_))
    # self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(1e-2, momentum=0.9).minimize(cost)
    self.train_ = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, x, y):

    x = np.atleast_2d(x)
    y = np.atleast_1d(y)
    self.session.run(self.train_, feed_dict={self.x: x, self.y: y})

  def predict(self, x):

    x = np.atleast_2d(x)
    return self.session.run(self.predict_, feed_dict={self.x: x})


def td_episode(env, policy_model, value_model, gamma):

  observation = env.reset()
  done = False
  total_reward = 0
  iterations = 0

  while not done and iterations < 200:

    action = policy_model.sample_action(observation)
    previous_observation = observation
    observation, reward, done, info = env.step(action)


    # update the models
    next_value = value_model.predict(observation)[0]
    G = reward + gamma*next_value
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
  policy_model.partial_fit(states, actions, advantages)
  value_model.partial_fit(states, returns)

  return total_reward


def main():
  env = gym.make('CartPole-v0')
  D = env.observation_space.shape[0]
  K = env.action_space.n

  policy = Policy(D, K, [])
  value = Value(D, [10])

  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)

  policy.set_session(session)
  value.set_session(session)
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  N = 1000
  total_rewards = np.empty(N)
  costs = np.empty(N)

  for n in range(N):

    env.render()
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