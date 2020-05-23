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

    def
