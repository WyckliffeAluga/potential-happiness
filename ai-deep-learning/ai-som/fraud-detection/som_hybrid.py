# -*- coding: utf-8 -*-
"""
Created on Fri May 29 21:10:36 2020

@author: wyckliffe
"""



# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense


# import the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# train the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# visualize the results
from pylab import bone, pcolor, colorbar, plot, show
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
frauds = np.concatenate((mappings[(8,9)], mappings[(6,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# customer matrix
customers =  dataset.iloc[:, 1:].values

is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0]  in frauds:
        is_fraud[i] = 1


# scale the features
sc = StandardScaler()
customers = sc.fit_transform(customers)

model = Sequential()
model.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(customers, is_fraud, batch_size = 1, epochs = 5)


# predict the Test set results
y_pred = model.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()]

