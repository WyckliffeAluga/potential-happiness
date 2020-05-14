# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:07:53 2020

@author: wyckliffe
"""


from __future__ import print_function, division 
from builtins import range 


import  matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import beta 

number_of_trials = 2000 
bandit_probabilities = [0.2, 0.5, 0.75]

class Bandit: 
    
    def __init__(self, p) : 
        
        self.p = p 
        self.a = 1
        self.b = 1
        self.N = 0 
        
    def pull(self): 
        
        return np.random.random() < self.p 
    
    def sample(self): 
        
        return np.random.beta(self.a , self.b)
    
    def update(self, x) : 
        self.a += x
        self.b += 1 - x
        self.N += 1 
        

def plot(bandits, trial): 
    
    x = np.linspace(0, 1, 200)
    for b in bandits:  
        y  = beta.pdf(x, b.a, b.b)   
        plt.plot(x, y, label=f"real p: {b.p:.4f}, win rate = {b.a -1 } / {b.N}")
    plt.title(f"Bandit distribution after {trial} trials")
    plt.legend() 
    plt.show() 
    
def experiment(): 
    
    bandits = [Bandit(p) for p in bandit_probabilities]
    
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(number_of_trials)
    
    for i in range(number_of_trials): 
        # thomspon sampling 
        j = np.argmax([b.sample() for b in bandits])
        # plot the posterios 
        for i in sample_points : 
            plot(bandits, i)
            
        # pull the arm 
        x = bandits[j].pull() 
        # update rewards 
        rewards[i] = x 
        # update the distribution 
        bandits[j].update(x) 
    
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
    
        
        
        
    