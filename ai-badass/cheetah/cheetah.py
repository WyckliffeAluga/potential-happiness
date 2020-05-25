# -*- coding: utf-8 -*-
"""
Created on Sun May 24 20:38:01 2020

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

class ReplayBuffer :

  def __init__(self, max_size=1e6) :
    self.memory = []
    self.max_size = max_size
    self.index = 0

  def add_transition(self, transition):

    if len(self.memory) == self.max_size: # check whether the memory is full
      self.memory[int(self.index)] = transition
      self.index = (self.index + 1) % self.max_size
    else :
      self.memory.append(transition)

  def sample(self, batch_size) :

    indexes = np.random.randint(0, len(self.memory) + 1, batch_size)
    states_batch, next_states_batch, actions_batch, rewards_batch, dones_batch = [], [], [], [],[]
    for i in indexes :
      state, next_state, action, reward, done = self.memory[i]
      states_batch.append(np.array(state, copy=False))
      next_states_batch.append(np.array(next_state, copy=False))
      actions_batch.append(np.array(action, copy=False))
      rewards_batch.append(np.array(reward, copy=False))
      dones_batch.append(np.array(done, copy=False))

    return np.array(states_batch), np.array(next_states_batch), np.array(actions_batch), np.array(rewards_batch).reshape[-1,1], np.array(dones_batch).reshape(-1,1)


class Actor(nn.Module):

  def __init__(self, state_dim, action_dim, max_action):

    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400, bias=False)
    self.layer_2 = nn.Linear(400, 300, bias=False)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x) :

    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = torch.tanh(self.later_3(x)) * self.max_action
    return x


class Critic(nn.Module) :

  def __init__(self, state_dim, action_dim ) :
    super(Critic, self).__init__()

    # first critc NN
    self.layer_1 = nn.Linear(state_dim + action_dim, 400, bias=False)
    self.layer_2 = nn.Linear(400, 300, bias=False)
    self.layer_3 = nn.Linear(300, 1)

    # second critic NN
    self.layer_4 = nn.Linear(state_dim + action_dim, 400, bias=False)
    self.layer_5 = nn.Linear(400, 300, bias=False)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u) :

    xu = torch.cat([x, u], 1)
    # first critic NN
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # second critic NN
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)

    return x1, x2


