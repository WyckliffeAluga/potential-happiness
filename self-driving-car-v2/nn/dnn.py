# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:39:20 2020

@author: wyckliffe
"""


import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
from tensorflow.keras.layers import Dense



class dnn(): 
    
    def __init__(self, n): 
        self.n = n 
        X, y  = datasets.make_circles(n_samples=self.n, random_state=123, noise=0.1, factor=0.2)
        self.X = X
        self.y = y 
        
    
    def mason(self): 
        model = tf.keras.Sequential()
        model.add(Dense(4, input_shape=(2,), activation="sigmoid"))
        model.add(Dense(1, activation="sigmoid"))
        adam = tf.keras.optimizers.Adam
        model.compile(adam(lr=0.01), loss="binary_crossentropy", metrics=['accuracy'])
        h = model.fit(x=self.X, y= self.y, verbose=1, batch_size=20 , epochs=1, shuffle="true")
    
        return h
    
    def progress(self): 
        
       h =  self.mason()
       plt.plot(h.history["accuracy"])
       plt.xlabel('epoch')
       plt.title('accuracy')
       plt.show()
    
        
    def plot_boundary(self): 
        X, y  = datasets.make_circles(n_samples=self.n, random_state=123, noise=0.1, factor=0.2)
        
        model = tf.keras.Sequential()
        model.add(Dense(4, input_shape=(2,), activation="sigmoid"))
        model.add(Dense(1, activation="sigmoid"))
        adam = tf.keras.optimizers.Adam
        model.compile(adam(lr=0.01), loss="binary_crossentropy", metrics=['accuracy'])
        model.fit(x=X, y= y, verbose=1, batch_size=20 , epochs=100, shuffle="true")
    
        x_span = np.linspace(min(X[:,0]) - 0.25, max(X[:,0]) + 0.25, 50)
        y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:,0]) + 0.25, 50)
        xx, yy = np.meshgrid(x_span, y_span)
        
        xx_, yy_ = xx.ravel(), yy.ravel()
        grid = np.c_[xx_, yy_]
        
        pred_func = model.predict(grid)
        z = pred_func.reshape(xx.shape)
        
        plt.contourf(xx, yy, z)
        plt.scatter(X[y == 0, 0 ] , X[y == 0 , 1])
        plt.scatter(X[y == 1, 0 ] , X[y == 1 , 1])
        
        x = 0.1 
        y = 0
        
        point = np.array([[x ,y]])
        
        prediction = model.predict(point)
        plt.plot([x], [y], marker='o', markersize=10, c='r')
        
        print(prediction)
       # plt.show()
        
    def show(self): 
        
        plt.scatter(self.X[self.y == 0, 0 ] , self.X[self.y == 0 , 1])
        plt.scatter(self.X[self.y == 1, 0 ] , self.X[self.y == 1 , 1])
        self.plot_boundary()
        plt.show()
                           
                           
d = dnn(500)        
    

       