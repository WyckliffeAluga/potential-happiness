# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:32:56 2020

@author: wyckliffe
"""


import gym
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

from cart_pole_v2_q_learning_with_bins import plot_running_average
tf.disable_v2_behavior()


class Layer :

    def __init__(self, m1, m2, f=tf.nn.tanh, use_bias=True) :

        self.w = tf.Variable(tf.random.normal(shape=(m1, m2)))
        self.params = [self.w]
        self.use_bias = use_bias

        if self.use_bias :
            self.b = tf.Variable(np.zeros(m2).astype(np.float32))
            self.params.append(self.b)

        self.f = f

    def forward(self, x) :

        if self.use_bias:
            a = tf.matmul(x, self.w) + self.b
        else:
            a = tf.matmul(x, self.w)

        return self.f(a)

class Network:

    def __init__(self, D, K, layer_sizes, gamma, max_experiences=1000, min_experiences=100, batch_size=32) :

        self.K = K

        # create the graph
        self.layers = []
        m1 = D

        for m2 in layer_sizes :
            layer = Layer(m1, m2)
            self.layers.append(layer)
            m1 = m2

        # final layer
        layer = Layer(m1, self.K, lambda x: x)
        self.layers.append(layer)

        # collect params
        self.params = []

        for layer in self.layers :
            self.params += layer.params

        # inputs and targets
        self.x = tf.placeholder(tf.float32, shape=(None, D), name='x')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32 , shape=(None,), name='actions')

        # calculate output and cost
        z = self.x

        for layer in self.layers :
            z = layer.forward(z)
        y_ = z
        self.predict_ = y_

        selected_action_values = tf.reduce_sum(y_ * tf.one_hot(self.actions, K),
                                               reduction_indices=[1])

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))

        self.train_ = tf.train.AdagradOptimizer(10e-3).minimize(cost)

        # replay memory
        self.experience = {'s':[], 'a':[], 'r':[], 's2':[]}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other) :

        # collect all the ops
        ops = []
        my_params = self.params
        other_params = other.params

        for p, q in zip(my_params, other_params) :
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def predict(self, x):

        x = np.atleast_2d(x)
        return self.session.run(self.predict_, feed_dict={self.x : x})

    def train(self, target_network) :

        # sample a random batch from buffer and fo an iteration
        if len(self.experience['s']) < self.min_experiences :

            # don't do anythin if we don't have enough experiences
            return

        # randomly select a batch
        idx = np.random.choice(len(self.experience['s']),
                               size=self.batch_size ,
                               replace=False)
        states = [self.experience['s'][i] for i in idx]
        actions= [self.experience['a'][i] for i in idx]
        rewards= [self.experience['r'][i] for i in idx]

        next_states = [self.experience['s2'][i] for i in idx]
        next_Q      = np.max(target_network.predict(next_states), axis=1)
        targets  = [r + self.gamma*next_q for r, next_q in  zip(rewards, next_Q)]

        # call optimizer
        self.session.run(
            self.train_,
            feed_dict={
                self.x: states,
                self.G: targets,
                self.actions: actions
                }
            )

    def add_experience(self, s, a, r, s2) :

        if len(self.experience['s']) >= self.max_experiences :
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)

        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)

    def sample_action(self, x, eps) :

        if np.random.random() < eps :
            return np.random.choice(self.K)
        else:
            x = np.atleast_2d(x)
            return np.argmax(self.predict(x)[0])


def episode(env, model, tmodel, eps, gamma, copy_period) :

    observation = env.reset()
    done = False
    total_reward = 0
    iterations = 0

    while not done and iterations < 2000 :

        action = model.sample_action(observation, eps)
        previous_observations =observation
        observation, reward, done, info = env.step(action)

        total_reward += reward

        if done :
           # print("Failed")
            reward =- 200

        # update the model
        model.add_experience(previous_observations, action, reward, observation)
        model.train(tmodel)

        iterations += 1

        if iterations % copy_period == 0 :
            tmodel.copy_from(model)

    return total_reward

def main():

    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_period = 50

    D = len(env.observation_space.sample())
    K = env.action_space.n
    sizes = [200,200]
    model = Network(D, K , sizes, gamma)
    tmodel= Network(D, K , sizes, gamma)

    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)

    model.set_session(session)
    tmodel.set_session(session)

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 500
    total_rewards = np.empty(N)
    costs = np.empty(N)

    for n in range(N) :
        eps = 1.0 / np.sqrt(n+1)
        total_reward = episode(env, model, tmodel, eps, gamma, copy_period)
        total_rewards[n] = total_reward

        if n % 100 == 0 :
            print('Episode:', n , "Total reward:", total_reward, 'eps:', eps, total_rewards[max(0, n-100):(n+1)].mean())

    print("Average reward for the last 100 episodes:", total_rewards[-100:].mean())
    print("Total steps:", total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_average(total_rewards)

if __name__ == '__main__':
  main()
