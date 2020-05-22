# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:30:29 2020

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

    def __init__(self, m1, m2, f=T.tanh, use_bias=True) :

        self.w = theano.shared(np.random.randn(m1, m2)/ np.sqrt(m1+m2))
        self.params = [self.w]
        self.use_bias = use_bias

        if use_bias :
            self.b = theano.shared(np.zeros(m2))
            self.params += [self.b]
        self.f = f

    def forward(self,x):

        if self.use_bias:
            a = x.dot(self.w) + self.b
        else:
            a = x.dot(self.w)

        return self.f(a)

class Network:

    def __init__(self, D, K , layer_sizes, gamma, max_experiences=1000, min_experiences=100, batch_size=32):

        self.K = K
        lr = 10e-3
        mu = 0.
        decay = 0.99

        # create the graph
        self.layers = []
        m1 = D

        for m2 in layer_sizes :
            layer = Layer(m1, m2)
            self.layers.append(layer)
            m1 = m2

        # final layer
        layer = Layer(m1, self.K, lambda x:x)
        self.layers.append(layer)

        # collect params for copy
        self.params = []

        for layer in self.layers :
            self.params += layer.params

        caches = [theano.shared(np.ones_like(p.get_value()) * 0.1) for p in self.params]
        velocities = [theano.shared(p.get_value()*0) for p in self.params]

        # inputs and targets
        x = T.matrix('x')
        G = T.vector('G')
        actions = T.ivector('actions')

        # calculate output and cost
        z = x

        for layer in self.layers :
            z = layer.forward(z)
        y_ = z

        selected_action_values =y_[T.arange(actions.shape[0]), actions]
        cost = T.sum((G-selected_action_values)**2)

        # train function
        grads = T.grad(cost, self.params)
        g_update = [(p, g, v) for p, v, g in zip(self.params, velocities, grads)]
        c_update = [(c, decay*c + (1 - decay)*g*g) for c, g in zip(caches, grads)]
        v_update = [(v, mu*v - lr*g / T.sqrt(c)) for v, c, g in zip(velocities, caches, grads )]

        updates = c_update + g_update + v_update

        # compile functions
        self.train_ = theano.function(
            inputs=[x, G, actions],
            updates=updates,
            allow_input_downcast=True,
            )

        self.predict_ = theano.function(
            inputs=[x],
            outputs=y_,
            allow_input_downcast=True,
            )

        # replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.gamma = gamma


    def copy_from(self, other):

        my_params = self.params
        other_params = other.params

        for p, q in zip(my_params, other_params) :
            actual = q.get_value()
            p.set_value(actual)

    def predict(self, x) :

        x = np.atleast_2d(x)
        return self.predict_(x)

    def train(self, target_network) :

        # sample a random batch from buffer

        if len(self.experience['s']) < self.min_experiences :

            return

        idx = np.random.choice(len(self.experience['s']), size=self.batch_size, replace=True)

        states = [self.experience['s'][i] for i in idx]
        actions= [self.experience['a'][i] for i in idx]
        rewards= [self.experience['r'][i] for i in idx]

        next_states = [self.experience['s2'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self.gamma * next_q for r, next_q in zip(rewards, next_Q)]

        # call optimizers
        self.train_(states, targets, actions)

    def add_experiences(self, s, a, r, s2) :

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
        previous_observation = observation
        observation, reward, done, info = env.step(action)

        total_reward += reward
        if done :
            reward = -200

        # updatte thhe model
        model.add_experiences(previous_observation, action, reward, observation)
        model.train(tmodel)

        iterations += 1

        if iterations % copy_period == 0 :
            tmodel.copy_from(model)

    return total_reward

def main() :

    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_period = 50

    D = len(env.observation_space.sample())
    K = env.action_space.n
    sizes = [200, 200]
    model = Network(D, K , sizes, gamma)
    tmodel = Network(D, K , sizes, gamma)

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 600
    total_rewards = np.empty(N)
    costs = np.empty(N)

    for n in range(N):
        eps = 1.0/ np.sqrt(n+1)
        total_reward = episode(env, model , tmodel, eps, gamma, copy_period)

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



