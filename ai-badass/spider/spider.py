# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:32:55 2020

@author: wyckliffe
"""


import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque


# Experience replay

class ReplayBuffer:

  def __init__(self, max_size=1e6):

    self.memory = []
    self.max_size = max_size
    self.index = 0

  def add(self, transition):

    if len(self.memory) == self.max_size:
      self.memory[int(self.index)] = transition
      self.index = (self.index + 1) % self.max_size
    else:
      self.memory.append(transition)

  def sample(self, batch_size):

    indexes = np.random.randint(0, len(self.memory), size=batch_size)

    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

    for i in indexes:

      state, next_state, action, reward, done = self.memory[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))

    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


# create the actor class
class Actor(nn.Module):

  def __init__(self, state_dim, action_dim, max_action):

    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):

    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))

    return x

# create the critic class
# this class has two neural networks that will be build simultenous
class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):

    super(Critic, self).__init__()
    # first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)

    # second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):

    xu = torch.cat([x, u], 1)

    # Forward-Propagation on the first Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)

    # Forward-Propagation on the second Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):

    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1
