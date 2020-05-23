# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:38:55 2020

@author: wyckliffe
"""


import numpy as np


class DQN:

    def __init__(self, max_memory=100, discount=0.9) :

        # memoery of experience replay
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount
