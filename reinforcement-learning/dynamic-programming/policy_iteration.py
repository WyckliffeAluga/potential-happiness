# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:50:59 2020

@author: wyckliffe
"""

import numpy as np 
from grid_world import standard_grid , negative_grid 
from iterative_policy_evaluation import print_values, print_policy 

e = 10e-4 # threshold 
gamma = 0.9 
all_actions = ('U', 'D' , 'L', 'R')

# this is determinitstic that is all p(s',r | s, a) = 1 or 0 

if __name__ == '__main__' : 
    
    # the grid gives a reward of -.1 for every non-terminal state 
    # This is should encourage finding a shorter parth to the goal 
    
    grid = negative_grid() 
    
    # print rewards a
    print("rewards" )
    print_values(grid.rewards , grid )
    
    # state ---> action 
    # randomly choose an action and update as we learn 
    policy = {} 
    
    for s in grid.actions.keys() : 
        policy[s]  = np.random.choice(all_actions) 
        
    
    # print initial policy 
    print("Initial policy") 
    print_policy(policy, grid) 
    
    # intialize V(s) 
    states = grid.all_states() 
    V = {} 
    for s in states: 
        
        if s in grid.actions: 
            V[s] = np.random.random()
        
        else: 
            # terminal state 
            V[s] = 0 
        
    # repeat until convergence 
    while True: 
        
        # policy evaluation step 
        while True : 
            biggest_change = 0 
            
            for s in states: 
                old_v = V[s]
                
                # V(s) only has value if it's not a terminal state 
                if s in policy: 
                    a = policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + gamma * V[grid.current_state()]
                    biggest_change = max(biggest_change , np.abs(old_v- V[s]))
                    
            if biggest_change < e : 
                break 
        
        # policy improvement step 
        is_policy_converged = True 
        
        for s in states : 
            if s in policy : 
                old_a = policy[s]
                new_a = None 
                best_value = float('-inf')
                
                # looping through all possible actions to find the best current action
                
                for a in all_actions : 
                    grid.set_state(s)  
                    r = grid.move(a)
                    v = r + gamma * V[grid.current_state()]
                    
                    if v > best_value : 
                        best_value = v 
                        new_a = a 
                
                policy[s] = new_a 
                if new_a != old_a : 
                    is_policy_converged = False 
        if is_policy_converged:
            break

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)            
        
        
    

