# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:44:51 2020

@author: wyckliffe
"""


#upper confidence boundaries 

import numpy as np 
import matplotlib.pyplot as plt 

number_of_trials = 10000
eps = 0.1
bandit_probabilities = [0.2, 0.5, 0.75]

class Bandit: 
    
    def __init__(self, p): 
        
        self.p = p # win rate 
        self.p_estimate = 0.
        self.N = 0. # number of samples collected so far 
        
    def pull(self): 
        
        # draw a 1 with probability p 
        return np.random.random() < self.p 
    
    def update(self, x ) : 
        self.N += 1 
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N
        
        
def ucb(mean, n, nj ): 
    
    return mean + np.sqrt(2 * np.log(n) / nj)

def experiment(): 
    
    bandits = [Bandit(p) for p in bandit_probabilities]
    rewards = np.empty(number_of_trials)
    total_plays = 0 
    
    
    # initialization play each bandit once 
    for j in range(len(bandits)): 
        x = bandits[j].pull()
        total_plays += 1 
        bandits[j].update(x)
        
    for i in range(number_of_trials): 
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1 
        bandits[j].update(x)
        
        rewards[i] = x 
        # print mean estimates for each bandit 
    
    for b in bandits: 
        print("Mean estimate :", b.p_estimate)
        
    # print total reward 
    print("Total reward earned:", rewards.sum())
    print("Overall win rate: ", rewards.sum() / number_of_trials)
    print("Number of times agent selected each bandit:", [b.N for b in bandits])
    
    average = np.cumsum(rewards) / (np.arange(number_of_trials) + 1)
    
    plt.plot(average)
    plt.plot(np.ones(number_of_trials) * np.max(bandit_probabilities))
    plt.xscale('log')
    plt.show()
    
    
if __name__ == "__main__": 
    experiment()        
         