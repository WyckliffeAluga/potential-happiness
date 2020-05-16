# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:42:07 2020

@author: wyckliffe
"""


import numpy as np 
from grid_world import standard_grid 
import matplotlib.pyplot as plt 

e = 10e-4 # threshold for convergence 

def print_values(V, g):
    
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
    
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")
        

if __name__ == "__main__" : 
    
    # iteractive policy evaluations 
    # given a policu, find v(s)
    # use both uniform random policy and fixed p olicy 
    # sources of randomness: 
        # p(a|s) --> deciding what action to take given the state 
        # p(s' , r | s, a) --> the next state and reward given action-state pair 
    
    # let us modle p(a|s) = uniform 
    
    grid = standard_grid() 
    
    # states will be positions (i,j)
    states = grid.all_states() 
    
    # uniformly random actions 
    # initialize V(s) = 0 
    
    V = {}
    
    for s in states : 
        V[s] = 0 
    gamma = 1.0 # discount gactor 
    
    # repeat until convergence 
    
    while True : 
        biggest_change = 0 
        
        for s in states : 
            old_v = V[s]
            
            # V(s) only has value if it is not a terminal state 
            if s in grid.actions: 
                
                new_v = 0 # we will accumulate the answer 
                p_a   = 1.0 / len(grid.actions[s]) # each action has equal probability 
                
                for a in grid.actions[s] : 
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                    
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        
        if biggest_change < e : 
            break 
    
    print("Values for uniformly rabdom actions")
    print_values(V, grid) 
    print("\n\n") 
    
    
    # fixed policy 
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
      }
    print_policy(policy, grid)
    
    # initialize V(s) = 0 
    V = {}
    for s in states : 
        V[s] = 0 
        
    
    # how does V(s) change as we get further away from the reward 
    gamma = 0.9  # discount factor 
    
    # repeat untill convergence 
    while True : 
        biggest_change = 0 
        
        for s in states: 
            old_v = V[s]
            
            # V(s) only has value if it i snot a terminal state 
            if s in policy : 
                a = policy[s]
                grid.set_state(s) 
                r = grid.move(a)
                V[s] = r + gamma * V[grid.current_state()]
                biggest_change = max(biggest_change , np.abs(old_v - V[s]))
                
        if biggest_change < e : 
            break 
    
    print("Values for fixed policy.")
    print_values(V, grid)