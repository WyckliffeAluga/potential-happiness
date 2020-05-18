# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:13:17 2020

@author: wyckliffe
"""

import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from cart_pole_v2_q_learning_with_bins import plot_running_average


class SGDRegressor:

    def __init__(self, D) :

        self.w = np.random.randn(D)
        self.lr = 10e-2

    def partial_fit(self, X, Y) :

        self.w += self.lr * (Y - X.dot(self.w)).dot(X) # single gradient descent

    def predict(self, X) :

        return X.dot(self.w)


class Transformer:

    def __init__(self , env) :

        observation_examples = np.random.random((20000, 4))* 2 - 2
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0,  n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5,  n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1,  n_components=1000))
            ])
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations) :

        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

class Model:

    def __init__(self, env, feature_transformer) :

        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        for i  in range(env.action_space.n) :
            model = SGDRegressor(feature_transformer.dimensions)
            self.models.append(model)

    def predict(self, s) :

        X = self.feature_transformer.transform(np.atleast_2d(s))
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G) :

        X = self.feature_transformer.transform(np.atleast_2d(s))
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps) :

        if np.random.random() < eps :
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))


def episode(env, model, eps, gamma) :

    observation = env.reset()
    done = False
    total_reward = 0
    iterations = 0

    while not done and iterations < 2000 :

        action = model.sample_action(observation, eps)
        previous_observation = observation
        observation, reward, done , info = env.step(action)

        if done :
            reward = -200

        # update the model
        next_model = model.predict(observation)
        assert(len(next_model.shape) == 1)
        G = reward + gamma * np.max(next_model)
        model.update(previous_observation, action, G)

        if reward == 1:
            total_reward += reward
        iterations += 1

    return total_reward

def main() :

    env = gym.make('CartPole-v0')
    ft = Transformer(env)
    model = Model(env, ft)
    gamma = 0.99

    if 'monitor' in sys.argv :
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 1000
    total_rewards = np.empty(N)
    costs = np.empty(N)

    for n in range(N):

        #env.render()
        eps = 1.0 / np.sqrt(n+1)
        total_reward = episode(env, model, eps, gamma)
        total_rewards[n] = total_reward


        if n % 100 == 0 :

            print("Episode:", n, "Total reward:", total_reward, "eps:", eps, "Average reward (last 100):", total_rewards[max(0, n-100):(n+1)].mean())

    print("Average reward for last 100 episodes:", total_rewards[-100:].mean())
    print("Total steps:", total_rewards.sum())
    env.render()
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_average(total_rewards)

if __name__ == "__main__"  :
    main()

