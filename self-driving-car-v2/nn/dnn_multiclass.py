# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:08:56 2020

@author: wyckliffe 
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
from tensorflow.keras.layers import Dense
from keras.utils.np_utils import to_categorical


class DNN(): 
    
    def __init__(self, n):
        
        centers = [[-1,1], [-1, -1], [1 , -1]]
        X , y = datasets.make_blobs(n_samples=n, random_state=123, centers=centers, cluster_std=0.4)
        self.x = X 
        self.y = y
        
    def encode(self): 
        
        y_cat = to_categorical(self.y, 3)
        
        return y_cat
    
    def mason(self): 
        
        model = tf.keras.Sequential()
        model.add(Dense(units=3, input_shape=(2,), activation="sigmoid"))
        model.add(Dense(3, activation="softmax"))
        adam = tf.keras.optimizers.Adam
        model.compile(adam(lr=0.01), loss="categorical_crossentropy", metrics=['accuracy'])
        model.fit(x=self.x, y= self.encode(), verbose=1, batch_size=50 , epochs=100, shuffle="true")
        
        return model
    
    def progress(self):
        h =  self.mason()
        plt.plot(h.history["accuracy"])
        plt.xlabel('epoch')
        plt.title('accuracy')
        plt.show()
        
    def show(self):
        
        plt.scatter(self.x[self.y == 0, 0], self.x[self.y == 0, 1])
        plt.scatter(self.x[self.y == 1, 0], self.x[self.y == 1, 1])
        plt.scatter(self.x[self.y == 2, 0], self.x[self.y == 2, 1])
        plt.show()
    
    def plot_boundary(self, n): 
        centers = [[-1,1], [-1, -1], [1 , -1], [1 ,1], [0 , 0]]
        
        X , y = datasets.make_blobs(n_samples= n, random_state=123, centers=centers, cluster_std=0.4)
        y_cat = to_categorical(y, 5)
        
        model = tf.keras.Sequential()
        model.add(Dense(units=5, input_shape=(2,), activation="sigmoid"))
        model.add(Dense(5, activation="softmax"))
        adam = tf.keras.optimizers.Adam
        model.compile(adam(lr=0.01), loss="categorical_crossentropy", metrics=['accuracy'])
        model.fit(x=X, y= y_cat, verbose=1, batch_size=50 , epochs=100, shuffle="true")
    
        x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:,0]) + 0.25, 50)
        y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:,0]) + 0.25, 50)
        xx, yy = np.meshgrid(x_span, y_span)
        
        xx_, yy_ = xx.ravel(), yy.ravel()
        grid = np.c_[xx_, yy_]
        
        pred_func = model.predict_classes(grid)
        z = pred_func.reshape(xx.shape)
        
        plt.contourf(xx, yy, z)
        plt.scatter(X[y == 0, 0 ] , X[y == 0 , 1])
        plt.scatter(X[y == 1, 0 ] , X[y == 1 , 1])
        plt.scatter(X[y == 2, 0 ] , X[y == 2 , 1])
        plt.scatter(X[y == 3, 0 ] , X[y == 3 , 1])
        plt.scatter(X[y == 4, 0 ] , X[y == 4 , 1])
        
        x = 0.5 
        y = -1
        
        point = np.array([[x ,y]])
        
        prediction = model.predict_classes(point)
        plt.plot([x], [y], marker='o', markersize=10, c='r')
        
        print(prediction)
        plt.show()

d = DNN(500)