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
