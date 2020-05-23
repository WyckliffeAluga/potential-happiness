# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:21:23 2020

@author: wyckliffe
"""


from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


class Network:

    def __init__(self, learning_rate=0.001, number_actions=5) :

        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        x = Dense(units=64, activation='sigmoid')(states)
        y = Dense(units=32, activation='sigmoid')(x)
        q_values = Dense(units=number_actions, activation='softmax')(y)
        self.model = Model(states, q_values)

        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
