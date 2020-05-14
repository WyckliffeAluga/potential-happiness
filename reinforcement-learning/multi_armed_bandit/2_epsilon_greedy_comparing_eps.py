# -*- coding: utf-8 -*-
"""
Created on Tue May 12 03:08:01 2020

@author: wyckl
"""


import numpy as np
import matplotlib.pyplot as plt


class Bandit: 
    
    def __init__(self, p) : 
        
        self.p = p 
        self.p_estimate = 0 
        self.N = 0 
        
    def pull(self): 
        
        return np.random.randn() + self.p 
    
    def update(self, x): 
        
        self.N += 1 
        self.p_estimate = (1 - 1.0 / self.N ) * self.p_estimate + 1.0 / self.N * x 
        

def experiment(p1, p2, p3, eps, N): 
    
    bandits = [Bandit(p1), Bandit(p2), Bandit(p3)]
    
    # count number of sub optimal choices 
    means = np.array([p1, p2, p3])
    true_best = np.argmax(means)
    count_suboptimal = 0 

    data = np.empty(N)

    for i in range(N): 
        # epsilon greedy 
        p = np.random.random() 
        if p < eps: 
            j = np.random.choice(len(bandits))
            
        else:
            j = np.argmax([b.p_estimate for b in bandits])
        
        x = bandits[j].pull()
        bandits[j].update(x)

        if j != true_best :
            count_suboptimal += 1 
            
        data[i] = x 
    
    average = np.cumsum(data) / (np.arange(N) + 1)
    
    # plot moving average 
    plt.plot(average)
    plt.plot(np.ones(N) * p1)
    plt.plot(np.ones(N) * p2)
    plt.plot(np.ones(N) * p3)
    plt.xscale('log')
    plt.show() 
    
    for b in bandits: 
        print(b.p_estimate)
        
    print("Percent sub optimal for epsilon = %s:", eps, float(count_suboptimal) / N)
    
    return average 

if __name__ == '__main__': 
    p1, p2, p3 = 1.5 , 2.5 , 3.5 
    c1 = experiment(p1, p2, p3, 0.1, 100000)
    c2 = experiment(p1, p2, p3, 0.05,100000)
    c3 = experiment(p1, p2, p3, 0.01,100000)
    
    # log scale plot 
    plt.plot(c1, label="eps =  0.1")
    plt.plot(c2, label='eps = 0.05')
    plt.plot(c3, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show() 
    
    # plot linear scale 
    plt.plot(c1, label="eps =  0.1")
    plt.plot(c2, label='eps = 0.05')
    plt.plot(c3, label='eps = 0.01')
    plt.legend()
    plt.show() 
                        
            
                
            