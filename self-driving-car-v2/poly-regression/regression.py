# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:25:22 2020

@author: wyckliffe
"""


import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.layers import  Dense

class Poly(): 
    
    def __init__(self, n): 
        
        self.n = n 
        self.x = []
        self.y = []
        
        self.get_data()
        
        self.net()
        
        
    def get_data(self): 
        np.random.seed(0)
        
        self.x = np.linspace(-3, 3, self.n)
        self.y = np.sin(self.x) + np.random.uniform(-0.5, 0.5, self.n)
        
        
    def net(self): 
        model = tf.keras.Sequential()
        model.add(Dense(50, input_dim=1 , activation='sigmoid'))
        model.add(Dense(30 , activation='sigmoid'))
        model.add(Dense(1))
        adam = tf.keras.optimizers.Adam
        model.compile(optimizer = adam(lr=0.01), loss="mse", metrics=['accuracy'])
        model.fit(self.x, self.y, epochs=50)
        model.save('polymodel.h5')
        
        plt.scatter(self.x, self.y)
        
        predictions = model.predict(self.x)
        plt.plot(self.x, predictions,'ro')
        plt.show()
        
p = Poly(1000)