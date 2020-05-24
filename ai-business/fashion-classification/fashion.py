# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:54:20 2020

@author: wyckliffe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Fashion :

    def __init__(self) :

        train = pd.read_csv('fashion-mnist_train.csv', sep=',')
        valid = pd.read_csv('fashion-mnist_test.csv' , sep=',')

        assert(train.shape[1] == valid.shape[1])

        self.train = np.array(train, dtype='float32')
        self.valid  = np.array(valid, dtype='float32')

    def show(self) :

        row = np.random.randint(0, len(self.train) + 1)
        plt.imshow(self.train[row, 1:].reshape(28,28), cmap=plt.get_cmap('gray'))


if __name__ == '__main__' :

    f = Fashion()

