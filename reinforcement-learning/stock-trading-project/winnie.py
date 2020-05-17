# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:55:50 2020

@author: wyckliffe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler


def get_data():

    # returns a T X 3 list of stock prices each row is a different stock
    # 0 = AAPL
    # 1 = MSI
    # 2 = SBUX

    df = pd.read_csv("aapl_msi_sbux.csv")
    return df.values


def get_scaler(env) :

    # returns scikit-learn scaler ojbect to scale the states
    states = []

    for _ in range(env.n_step):
        action = np.random.choice(env.action_space) # random action
        state, reward, done, info = env.step(action)
        states.append(state)

        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def makeDir(directory) :
    if not os.path.exists(directory):
        os.makedirs(directory)


class Model() :

    # a liner regression model

    def __init__(self, input_dim, n_action):

        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim) # weights
        self.b = np.zeros(n_action) # bias

        # momentum terms
        self.vW = 0
        self.vb = 0

        self.losses = []

    def predict(self, x):

        # makes sure X is in N x D
        assert(len(x.shape) == 2)

        return x.dot(self.W) + self.b

    def sgd(self, x, y, lr= 0.1, momentum=0.9) :

        # make sure x is in N x D
        assert(len(x.shape) == 2)

        # the loss values are 2 - D
        number_of_values = np.prod(y.shape)

        # do one step of gradient descent i.e d/dx (x^2) --> 2x
        y_ = self.predict(x)
        gW = 2 * x.T.dot(y_ - y) / number_of_values
        gb = 2 * (y_ - y).sum(axis=0) / number_of_values

        # update momentum terms
        self.vW = momentum * self.vW - lr * gW
        self.vb = momentum * self.vb - lr * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.mean((y_ - y )**2)
        self.losses.append(mse)


    def load(self, filepath) :
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save(self, filepath) :
        np.savez(filepath, W=self.W , b=self.b)


class Env :

    """# A 3 stock training environment

    State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (daily closing price)
    - price of stock 2 (daily closing price)
    - price of stock 3 (daily closing price)
    - cash owned (what can be used to purchase more stocks)

    Action: Categorical variable with 27 possibilities

    - for each stock you can
    - 0 = sell
    - 1 = hold
    - 2 = buy

    """

    def __init__(self, data, initial_investment=20000) :

        # data
        self.stock_price_history = data
        self.n_step , self.n_stock = self.stock_price_history.shape

        # instance attributes
        self.initial_investment = initial_investment
        self.current_step = None
        self.stock_owned = None
        self.stock_price = None

        self.action_space = np.arange(3 ** self.n_stock)

        # create action permutations
        # retuns a nested list with all the possible permutations

        self.action_list = list(map(list, itertools.product([0,1,2], repeat=self.n_stock)))

        # calculate the size of the sate
        self.state_dim = self.n_stock * 2 * 1

        self.reset()

    def reset(self):
        self.current_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.current_step]
        self.bank_balance = self.initial_investment

        return self._get_obs()

    def step(self, action) :

        assert(action in self.action_space)

        # get current value before perfoming an action
        previous_value = self._get_val()

        # update price, i.e go to the next day
        self.current_step += 1
        self.stock_price = self.stock_price_history[self.current_step]

        # perform the trade
        self._trade(action)

        # get the new value after taking the action
        current_value = self._get_val()

        # reward is the incrrease in porfolio value
        reward = current_value - previous_value

        # done if we run out of data
        done = self.current_step == self.n_step - 1

        # store the current value of the portfolio here
        info = {"current_value" : current_value}

        # conform to the API
        return self._get_obs(), reward , done , info

    def _get_obs(self) :
        # return observations

        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock : 2 * self.n_stock] = self.stock_price
        obs[-1] = self.bank_balance

        return obs

    def _get_val(self):

        # returns the current value of the portfolio

        return self.stock_owned.dot(self.stock_price) + self.bank_balance

    def _trade(self, action):

        # index the action we want to perform
        # 0 = sell
        # 1 = hold
        # 3 = buy

        action_vector  = self.action_list[action]

        # determine which stock to buy or sell
        sell_index = [] # stores index of stock to sell
        buy_index  = [] # stored index of stock to buy

        for i , a in enumerate(action_vector):
            if a == 0 :
                sell_index.append(i)
            elif a == 2 :
                buy_index.append(i)

        # sell any stocks we want to sell
        # buy any stocks we want to buy

        if sell_index :
            # when we sell ,we will sell all shares of that stock (to simplify the problem)

            for i in sell_index:
                self.bank_balance += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0

        if buy_index :

            # when buying we loop through each stock we want to buy and buy one share at a time until we run o=out of money
            # all or nothing
            can_buy = True

            while can_buy :
                for i in buy_index :
                    if self.bank_balance > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy one share
                        self.bank_balance -= self.stock_price[i]
                    else :
                        can_buy = False


class Agent(object) :

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = Model(state_size, action_size)


    def act(self, state):

        if np.random.rand() <= self.epsilon :
            return np.random.choice(self.action_size) # random choice

        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # greedy

    def train(self, state, action , reward, next_state, done):

        if done : # a little bit pendantic because stock market won't be ending soon but this is th end of our data
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state) # targets to the other actions
        target_full[0, action] = target #

        # run one training step
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min :
            self.epsilon += self.epsilon_decay

    def load(self, name) :
        self.model.load(name)

    def save(self, name) :
        self.model.save(name)


def episode(agent, env , is_train) :

    # after transforming states are already 1 x D
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done :

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])

        if is_train == 'train' :
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info['current_value']


if __name__ == '__main__':

  # configurations
  models_folder = 'winnie_models'
  rewards_folder = 'winnie_rewards'
  num_episodes = 2000
  batch_size = 32
  initial_investment = 20000


  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  args = parser.parse_args()

  makeDir(models_folder)
  makeDir(rewards_folder)

  data = get_data()
  n_timesteps, n_stocks = data.shape

  n_train = n_timesteps // 2

  train_data = data[:n_train]
  test_data = data[n_train:]

  env = Env(train_data, initial_investment)
  state_size = env.state_dim
  action_size = len(env.action_space)
  agent = Agent(state_size, action_size)
  scaler = get_scaler(env)

  # store the final value of the portfolio (end of episode)
  portfolio_value = []

  if args.mode == 'test':
    # then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # remake the env with test data
    env = Env(test_data, initial_investment)

    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0.01

    # load trained weights
    agent.load(f'{models_folder}/linear.npz')

  # play the game num_episodes times
  for e in range(num_episodes):
    t0 = datetime.now()
    val = episode(agent, env, args.mode)
    dt = datetime.now() - t0
    print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
    portfolio_value.append(val) # append episode end portfolio value

  # save the weights when we are done
  if args.mode == 'train':
    # save the DQN
    agent.save(f'{models_folder}/linear.npz')

    # save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)

    # plot losses
    plt.plot(agent.model.losses)
    plt.show()


  # save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)






