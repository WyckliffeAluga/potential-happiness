# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:54:28 2020

@author: wyckliffe
"""


import numpy as np 
import matplotlib.pyplot as plt


def draw(x1, x2): 
    
    ln = plt.plot(x1,x2)
   
    
    
def sigmoid(z): 
   
    return 1/(1 + np.exp(-z))


def err(weights, points, y): 
    
    m = points.shape[0]
    p = sigmoid(points * weights)
    
    return - (1/m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))


def gradient_descent(weights, points, y, alpha): 
    m = points.shape[0]
    
    for i in range(2000): 
         p = sigmoid(points * weights)
         gradients = (points.T * (p - y)) * (alpha/m)
         weights = weights - gradients
         w1 = weights.item(0)
         w2 = weights.item(1)
         b  = weights.item(2)
         x1 = np.array([points[:,0].min(), points[:,0].max()])
         x2 = -b/w2 + (x1*(-w1/w2))
    draw(x1,x2) 

n = 100
np.random.seed(0)
bias= np.ones(n)

red = np.array([np.random.normal(10,2,n), np.random.normal(12,2,n), bias]).T
blue= np.array([np.random.normal(5,2, n), np.random.normal(6,2, n), bias]).T
all_points=np.vstack((red, blue))
 
line_parameters = np.matrix([np.zeros(3)]).T
y = np.array([np.zeros(n), np.ones(n)]).reshape(n*2, 1)
 
_, ax= plt.subplots(figsize=(4,4))
ax.scatter(red[:,0], red[:,1], color='r')
ax.scatter(blue[:,0], blue[:,1], color='b')

gradient_descent(line_parameters, all_points, y , 0.06)
print(err(line_parameters, all_points,y))
plt.show()
       
    
