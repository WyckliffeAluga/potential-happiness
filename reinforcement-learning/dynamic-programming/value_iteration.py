# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:33:25 2020

@author: wyckliffe
"""

import numpy as np 
from grid_world import standard_grid , negative_grid 
from iterative_policy_evaluation import print_values, print_policy 

e = 10e-4 # threshold 
gamma = 0.9 
all_actions = ('U', 'D' , 'L', 'R')

# this is determinitstic that is all p(s',r | s, a) = 1 or 0 

if __name__ == "__main__":
    
    grid = negative_grid() 
    
    # rewards 
    print("Rewards")
    print_values(grid.rewards, grid)
    
    # initialize policy 
    policy = {}
    for s in grid.actions.keys(): 
        policy[s] = np.random.choice(all_actions)
        
    print("Initial policy")
    print_policy(policy, grid)
    
    # initialize V(s)
    V = {}
    
    states = grid.all_states() 
    for s in states : 
        if s in grid.actions : 
            V[s] = np.random.random() 
        else: 
            V[s] = 0 
            
    
    # repeat until convergence 
    while True : 
        biggest_change = 0 
        for s in states: 
            old_v = V[s]
            
            if s in policy : 
                new_v = float("-inf")
                for a in all_actions : 
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + gamma * V[grid.current_state()]
                    
                    if v > new_v : 
                        new_v = v 
                V[s] = new_v
                biggest_change = max(biggest_change , np.abs(old_v - V[s]))
        
        if biggest_change < e : 
            break 
    
    # find policy that lead to optimal value function 
    for s in policy.keys(): 
        best_a = None 
        best_value = float('-inf')
        
        # loop through all the possible actions to find the best current action 
        
        for a in all_actions : 
            grid.set_state(s)
            r = grid.move(a)
            v = r + gamma * V[grid.current_state()]
            
            if v > best_value : 
                best_value = v 
                best_a = a 
        policy[s] = best_a 
    
    print("values")
    print_values(V, grid)
    print("Policy")
    print_policy(policy, grid)
        
        
    
