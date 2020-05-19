# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:37:10 2020

@author: wyckliffe
"""



# Solve mountain car with  TD(lambda) method

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime


import mountain_car_v1_q_learning
from mountain_car_v1_q_learning import plot_cost_to_go, Transformer, Model, plot_running_average



class Base:

    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, input_, target , eligibility, lr=10e-3) :

        self.w += lr * (target - input_.dot(self.w))*eligibility

    def predict(self,X) :
        X = np.array(X)
        return X.dot(self.w)

class Model:

    def __init__(self, env, transformer ) :

        self.env = env
        self.models = []
        self.transformer = transformer

        D = transformer.dimensions
        self.eligibility = np.zeros((env.action_space.n, D))

        for i in range(env.action_space.n) :
            model = Base(D)
            self.models.append(model)

    def predict(self, s) :

        X = self.transformer.transform([s])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G, gamma , lambda_) :

        X = self.transformer.transform([s])
        assert(len(X.shape) == 2)

        self.eligibility *= gamma * lambda_
        self.eligibility[a] += X[0]
        self.models[a].partial_fit(X[0] , G , self.eligibility[a])

    def sample_action(self, s, eps) :

        if np.random.random() < eps :
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def episode(model , eps, gamma, lambda_) :

    observation = env.reset()
    done = False
    total_reward = 0
    iterations = 0

    while not done and iterations < 10000 :

        action = model.sample_action(observation, eps)
        previous_observation = observation
        observation, reward, done ,info = env.step(action)

        # update model
        G = reward + gamma*np.max(model.predict(observation)[0])
        model.update(previous_observation, action, G, gamma, lambda_)

        total_reward += reward
        iterations += 1

    return total_reward

if __name__ == '__main__' :

    env = gym.make("MountainCar-v0")
    ft = Transformer(env)
    model = Model(env, ft)
    gamma = 0.99
    lambda_ = 0.7

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    total_rewards = np.empty(N)
    costs = np.empty(N)

    for n in range(N) :

        eps = 0.1 * (0.97**n)
        total_reward = episode(model, eps, gamma, lambda_)
        total_rewards[n] = total_reward
        print("Episode:", n, "Total reward:", total_reward)

    print("Average reward for last 100 episodes:", total_rewards[-100:].mean())
    print("total steps:", -total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_average(total_rewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)