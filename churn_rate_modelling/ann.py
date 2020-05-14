# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:35:16 2020

@author: wyckliffe
"""


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras 
from keras.models import Sequential
from keras.layers import Dense


class ANN(): 
    
    def __init__(self): 
        
        self.data = self.get_data() 
        
    
    def get_data(self): 
        
        data = pd.read_csv('bank_data.csv')
        
        X = data.iloc[:, 3:13]
        y = data.iloc[:, 13]
        
        return X, y 
    
    def encoder(self): 
      
        X , y = self.get_data() 
        X = pd.get_dummies(X)
        
        return X.values , y.values
    
    def pre_process(self): 
        X, y = self.encoder()
        x_train , x_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        scaler = StandardScaler() 
        
        return (scaler.fit_transform(x_train) , 
                scaler.fit_transform(x_test), 
                y_train, y_test) 
    
    def ann(self): 
        model = Sequential()
        model.add(Dense(6, input_dim=10), activation='relu')
        model.add(Dense(1))
        
        
        
a = ANN()