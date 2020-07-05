# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:24:14 2020

@author: wyckliffe
"""


import numpy as np
import os
import random
import shutil
from statistics import mean
from base_model import BaseModel
from cnn import CNN

GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

class DDQNmodel(BaseModel):

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseModel.__init__(self, game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.ddqn = CNN(self.input_shape, action_space).model
        if os.path.isfile(self.model_path):
            self.ddqn.load_weights(self.model_path)

    def _save_model(self):
        self.ddqn.save_weights(self.model_path)
