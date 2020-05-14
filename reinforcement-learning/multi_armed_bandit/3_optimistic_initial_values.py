# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:05:55 2020

@author: wyckliffe
"""


import numpy as np 
import matplotlib.pyplot as plt 


number_of_trials  = 10000 
eps = 0.1 
bandit_probabilities = [0.2, 0.5, 0.75]

class Bandit: 
    
    def __init__(self, p): 
        
        self.p = p 
        self.p_estimate = 5.
        self.N = 1. 
        
    
    def pull(self): 
        
        return np.random.random() < self.p 
    
    def update(self , x): 
        self.N += 1
        
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N 
        
        
def experiment(): 
    
    bandits = [Bandit(p) for p in bandit_probabilities]
    
    rewards = np.zeros(number_of_trials) 
    
    for i in range(number_of_trials) : 
        
        # use optimistic initial values with the next bandit 
        j = np.argmax([b.p_estimate for b in bandits])
        
        # pull the words 
        x = bandits[j].pull()
        
        # updates log 
        
        rewards[i] = x 
        
        # update reward 
        bandits[j].update(x)
        
    # print mean estimates for each bandit 
    
    for b in bandits: 
        print("Mean estimate :", b.p_estimate)
        
    # print total reward 
    print("Total reward earned:", rewards.sum())
    print("Overall win rate: ", rewards.sum() / number_of_trials)
    print("Number of times agent selected each bandit:", [b.N for b in bandits])
    
    # plot results 
    rewards = np.cumsum(rewards)
    win_rates = rewards / (np.arange(number_of_trials) + 1)
    plt.ylim([0,1])
    plt.plot(win_rates)
    plt.plot(np.ones(number_of_trials) * np.max(bandit_probabilities))
    plt.show()
    
    
if __name__ == "__main__": 
    experiment() 
    
    