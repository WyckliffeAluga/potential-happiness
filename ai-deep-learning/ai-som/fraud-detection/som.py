# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:34:03 2020

@author: wyckliffe
"""

import numpy as np
import pandas as pd
from pylab import bone, pcolor, colorbar, plot, show
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
#import the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# standardize the training dataset
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
# train the SOM
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)
# visualize
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
# find the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,1)], mappings[(5,2)]), axis = 0)
frauds = sc.inverse_transform(frauds)
