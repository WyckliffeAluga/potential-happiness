# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:18:38 2020

@author: wyckliffe
"""


from __future__ import print_function, division 
from builtins import range 


import  matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import norm 

np.random.seed(0)
number_of_trials = 2000 
bandit_means = [1, 2, 3]

class Bandit: 
    
    def __init__(self, true_mean) : 
        
        self.true_mean = true_mean
        self.predicted_mean = 0 
        self.lambda_ = 1 
        self.sum_x = 0 
        self.tau = 1 
        self.N = 0
        
        
    def pull(self): 
        
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean
    
    def sample(self): 
        
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean
    
    def update(self, x) : 
        self.lambda_ += self.tau 
        self.sum_x += x 
        self.predicted_mean = self.tau * self.sum_x / self.lambda_ 
        self.N += 1 
        

def plot(bandits, trial): 
    
    x = np.linspace(-3, 6, 200)
    for b in bandits:  
        y  = norm.pdf(x, b.predicted_mean,np.sqrt(1. / b.lambda_))   
        plt.plot(x, y, label=f"real p: {b.true_mean:.4f}, num plays = {b.N}")
    plt.title(f"Bandit distribution after {trial} trials")
    plt.legend() 
    plt.show() 
    
def experiment(): 
    
    bandits = [Bandit(m) for m in bandit_means]
    
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(number_of_trials)
    
    for i in range(number_of_trials): 
        # thomspon sampling 
        j = np.argmax([b.sample() for b in bandits])
        # plot the posterios 
      #  for i in sample_points : 
          #  plot(bandits, i)
            
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
    plt.plot(np.ones(number_of_trials) * np.max(bandit_means))
    plt.show()
    
    
if __name__ == "__main__": 
    experiment() 
    