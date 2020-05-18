# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:42:19 2020

@author: wyckliffe
"""


import gym
import os
import sys
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime



def state_builder(features) :

    # turns a list of integers into an int

    return int("".join(map(lambda feature : str(int(feature)), features)))

def bin_it(value, bins) :

    return np.digitize(x=[value], bins=bins)[0]


class  Transformer :

    def __init__(self):

        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2  , 2  , 9)
        self.pole_angle_bins    = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self, observation) :

        # returns an int
        cart_position, cart_velocity, pole_angle, pole_velocity = observation

        return state_builder([
            bin_it(cart_position, self.cart_position_bins),
            bin_it(cart_velocity, self.cart_velocity_bins),
            bin_it(pole_angle , self.pole_angle_bins),
            bin_it(pole_velocity, self.pole_velocity_bins)
            ])


class Model :

    def __init__(self, env, transformer) :

        self.env = env
        self.feature_transformer = transformer

        number_of_states = 10 ** env.observation_space.shape[0]
        number_of_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high=1, size=(number_of_states, number_of_actions))

    def predict(self, s) :

        x = self.feature_transformer.transform(s)
        return self.Q[x]

    def update(self, s, a , G) :

        x = self.feature_transformer.transform(s)
        self.Q[x,a] += 10e-3 * (G - self.Q[x,a]) # gradient descent

    def sample_action(self, s, eps) :

        if np.random.random() < eps : # choose a random action

            return self.env.action_space.sample()

        else:
            p = self.predict(s) # use the best posisible action
            return np.argmax(p)

def episode(model, eps, gamma) :

    observation = env.reset()
    done = False
    total_reward = 0
    iterations = 0

    while not done and iterations < 10000 :

        action = model.sample_action(observation, eps)
        previous_observation = observation
        observation , reward , done, info = env.step(action)

        total_reward += reward

        if done and iterations < 199 : # newer version of gym caps at 200 iterations
            reward = -300

        # update the model
        G = reward + gamma * np.amax(model.predict(observation)) # Q-learning equation
        model.update(previous_observation, action, G)

        iterations += 1

    return total_reward

def plot_running_average(total_reward):

    N = len(total_reward)
    running_averages = np.empty(N)

    for t in range(N) :
        running_averages[t] = total_reward[max(0 , t-100):(t + 1)].mean()

    plt.plot(running_averages)
    plt.title("Running Averages")
    plt.show()


if __name__ == '__main__' :

    env = gym.make('CartPole-v0')
    ft = Transformer()
    model = Model(env, ft)
    gamma = 0.9

    if 'monitor' in sys.argv :
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 10000
    total_rewards = np.empty(N)

    for n in range(N):
        #env.render()

        eps = 1.0 / np.sqrt(n+1)
        total_reward = episode(model, eps, gamma)
        total_rewards[n] = total_reward

        if n % 100 == 0 :
            print('episode:', n , 'total reward:', total_reward, 'eps:', eps)

    print("average reward for last 100 episodes:", total_rewards[-100:].mean())
    print('total steps:', total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")

    plot_running_average(total_rewards)

