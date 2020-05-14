# -*- coding: utf-8 -*-
"""
Created on Tue May 12 01:53:22 2020

@author: wyckliffe
"""
import numpy as np 
import matplotlib.pyplot as plt

number_trials = 10000
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
        
def experiment() : 
        
        bandits = [Bandit(p) for p in bandit_probabilities]
        
        rewards = np.zeros(number_trials)
        number_times_explored = 0 
        number_times_exploited= 0 
        number_optimal = 0 
        optimal_j = np.argmax([b.p for b in bandits])
        
        print("Optimal J:", optimal_j)
        
        for i in range(number_trials): 
            
            # select next bandit 
            if np.random.random( ) < eps: 
                number_times_explored += 1 
                j = np.random.randint(len(bandits)) 
                
            else:
                number_times_exploited += 1 
                j = np.argmax([b.p_estimate for b in bandits])
                
            if j == optimal_j : 
                number_optimal += 1 
                
            # pull arm with largest sample 
            x = bandits[j].pull()
            
            # update rewards 
            rewards[i] = x 
            
            # update bandit pulled 
            bandits[j].update(x)
            
        # print means 
        for b in bandits: 
            print("mean estimate: ", b.p_estimate)
        
        # print total reward 
        print("total reward: ", rewards.sum())
        print("overall win rate: ", rewards.sum() / number_trials)
        print("number of times explored: ", number_times_explored)
        print("number of times exploited: ", number_times_exploited)
        print("number timers selected optimal bandit: ", number_optimal)   

        # plot the results 
        results = np.cumsum(rewards)
        win_rates = results / (np.arange(number_trials) + 1)
        plt.plot(win_rates) 
        plt.plot(np.ones(number_trials) * np.max(bandit_probabilities))
        plt.show() 
    
if __name__ == "__main__": 
    experiment()        
        
        