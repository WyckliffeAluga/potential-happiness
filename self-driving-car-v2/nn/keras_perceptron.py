# -*- coding: utf-8 -*-
"""
Created on Wed May  6 00:01:26 2020

@author: wyckliffe
"""


import numpy as np 
import tensorflow.keras 
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
import matplotlib.pyplot as plt

n_pts = 500

np.random.seed(0)

Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T
 
X = np.vstack((Xa, Xb))
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
 
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

model = Sequential()
model.add(Dense(units = 1, input_shape=(2,), activation="sigmoid"))
adam = Adam(lr=0.1)
model.compile(adam, loss="binary_crossentropy", metrics=["accuracy"])
h = model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs = 50, shuffle='true')  

#plt.plot(h.history["accuracy"])
#plt.plot(h.history["loss"])
#plt.plot("accuracy")
#plt.xlabel("epochs")   
#plt.legend(['accuracy', 'loss'])
#plt.show()    

def plot_boundary(X, y, model): 
    
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1, 50)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,0]) + 1, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    
    plt.contourf(xx, yy, z)
    
plot_boundary(X, Y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

x = 7.5
y = 5 

point = np.array([[x,y]])
predictions = model.predict(point)

plt.plot([x] , [y], marker="o", c="w", markersize=10)
print(predictions, "prediction")
plt.show()