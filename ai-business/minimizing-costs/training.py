# -*- coding: utf-8 -*-
"""
Created on Sat May 23 01:14:41 2020

@author: wyckl
"""


import os
import numpy as np
import random as rn
import env
import network
import q_learning


# set seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(1234)

epsilon = 0.3
number_actions = 5
direction_boundary = (number_actions -1) / 2
number_epochs = 1000
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# Build the enviroment
env = env.Env(temp_range=(10.0, 24.0), initial_month=0, initial_number_of_users=20, initial_rate_data=30)

# launch network
brain = network.Network(learning_rate=0.0001, number_actions=number_actions)

# launch q_learning
dqn = q_learning.DQN(max_memory=max_memory, discount=0.9)

# mode
train = True


# train the AI
env.train = train

model = brain.model
