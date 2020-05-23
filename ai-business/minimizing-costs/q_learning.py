# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:38:55 2020

@author: wyckliffe
"""


import numpy as np


class DQN:

    def __init__(self, max_memory=100, discount=0.9) :

        # memory of experience replay
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, transition, game_over) :

        self.memory.append([transition, game_over])

        if (len(self.memory) > self.max_memory) :
            del self.memory[0]

    def get_batch(self, model, batch_size=10):

        length_memory = len(self.memory)
        number_inputs = self.memory[0][0].shape[1]
        number_outputs = self.model.output_shape[-1]

        inputs = np.zeros((min(length_memory, batch_size), number_inputs))
        targets= np.zeros((min(length_memory, batch_size), number_outputs))

