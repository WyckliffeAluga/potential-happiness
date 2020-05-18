# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:18:14 2020

@author: wyckliffe
"""


import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import  SGDRegressor


class Transformer:

    def __init__(self, env , n_components=500):

        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # We use RBF kernels with different variances to cover different parts of the space

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
        feature_examples = featurizer.fit(scaler.transform(observation_examples))

        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations) :
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


class Model:

    def __init__(self, env, feature_transformer, learning_rate) :

        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()]), [0])
            self.models.append(model)


    def predict(self, s) :

        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G) :

        X = self.feature_transformer.transform([s])
        assert (len(X.shape) == 2)
        self.models[a].partial_fit(X, [G])

    def sample_action(self, s, eps) :

        if np.random.random() < eps:

            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))



# returns a list of states_and_rewards, and the total reward
def episode(model, env, eps, gamma):

  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  while not done and iters < 10000:

    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    # update the model
    next = model.predict(observation)
    # assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(next[0])
    model.update(prev_observation, action, G)

    totalreward += reward
    iters += 1

  return totalreward


def plot_cost_to_go(env, estimator, number_titles=20) :

      x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=number_titles)
      y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=number_titles)

      X, Y = np.meshgrid(x,y)

      Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

      fig = plt.figure(figsize=(10,5))
      ax = fig.add_subplot(111, projection='3d')
      surf = ax.plot_surface(X,Y,Z,
                             rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm,
                             vmin=-1.0, vmax=1.0)
      ax.set_xlabel('Position')
      ax.set_ylabel('Velocity')
      ax.set_zlabel('Cost-To-Go == -V(s)')
      ax.set_title('Cost-To-Go Function')
      fig.colorbar(surf)
      plt.show()

def plot_running_average(total_rewards) :

    N = len(total_rewards)
    running_average = np.empty(N)

    for t in range(N):
        running_average[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_average)
    plt.title("Running Average")
    plt.show()

if __name__ == '__main__' :

    env = gym.make('MountainCar-v0')
    ft = Transformer(env)
    model = Model(env, ft, "constant")

    gamma = 0.99

    if 'monitor' in sys.argv :
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N = 300
    total_rewards = np.empty(N)

    for n in range(N) :
        env.render()
        eps = 0.1 * (0.97 ** n)
        total_reward =episode(model, env, eps, gamma)
        total_rewards[n] = total_reward

        print('Episode:', n, 'Total reward:', total_reward)

    print("Average reward for last 100 episodes:", total_rewards[-100:].mean())
    print('Total steps:', -total_rewards.sum())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_average(total_rewards)
    plot_cost_to_go(env, model)


