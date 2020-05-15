# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:01:46 2020

@author: wyckliffe
"""

import numpy as np 
from grid_world import standard_grid , negative_grid 
from iterative_policy_evaluation import print_values, print_policy 

e = 10e-4 # threshold 
gamma = 0.9 
all_actions = ('U', 'D' , 'L', 'R')

# the next state and reward will have some randomness 
# the probability of going in the desired location will be set to 0.5 
# the probability of going in a direaction a' which i s!= a will be set to 0.5/3


if __name__ == '__main__' : 
    
    # the grid will reward the agent -.1 for every none terminal state 
    
    grid = negative_grid(step_cost = -1.0)
    #grid = negative_grid(step_cost = -0.1)
    #grid = standard_grid()
    
    # print rewards a
    print("rewards" )
    print_values(grid.rewards , grid )
    
    # state ---> action 
    # randomly choose an action and update as we learn 
    policy = {} 
    
    for s in grid.actions.keys() : 
        policy[s]  = np.random.choice(all_actions) 
        
    # initial policy 
    print("Initial Policy")
    print_policy(policy, grid)

    # initialize V(s)
    V = {} 
    states = grid.all_states() 
    
    for s in states: 
        
        if s in grid.actions: 
            V[s] = np.random.random()
        
        else: 
            # terminal state 
            V[s] = 0 
            
    # repeat until convergence 
    while True : 
        
        # policy evaluation step 
        while True : 
            biggest_change = 0 
            
            for s in states: 
                old_v = V[s]
                
            # V(s) only has a value if it i snot a terminal state 
            new_v = 0 
            
            if s in policy : 
                for a in all_actions : 
                    if a == policy[s]:
                        p = 0.5 
                    else: 
                        p = 0.5/3 
                    
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p * (r + gamma * V[grid.current_state()])
                
                V[s] = new_v 
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
                
            if biggest_change < e : 
                break 
        # policy improvement step 
        is_policy_converged = True 
        
        for s in states: 
            if s in policy : 
                old_a = policy[s]
                new_a = None 
                best_value = float('-inf')
                
                # loop through all possible actions 
                
                for a in all_actions : 
                    v = 0
                    for a2 in all_actions : 
                        if a == a2: 
                            p = 0.5 
                        else: 
                            p = 0.5/3 
                        
                        grid.set_state(s)
                        r = grid.move(a2)
                        v += p * (r + gamma * V[grid.current_state()])
                    
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
            
                