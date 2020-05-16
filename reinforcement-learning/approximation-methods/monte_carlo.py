# -*- coding: utf-8 -*-
"""
Created on Fri May 15 17:42:04 2020

@author: wyckliffe
"""

import numpy as np 
import matplotlib.pyplot as plt
from grid_world import standard_grid , negative_grid 
from iterative_policy_evaluation import print_values, print_policy 

e = 10e-4 # threshold 
gamma = 0.9 
all_actions = ('U', 'D' , 'L', 'R')

# this is policy evaluation not optimization 

def play_game(grid, policy) : 
    
    # returns a list of states and corresponding returns 
    # resets the game to start at a random position 
    
    # we need to reset the game because given our current deterministic policy 
    # we would never end up at certain statest, but we still want to measure them 
    
    starting_states = list(grid.actions.keys() )
    starting_idx    = np.random.choice(len(starting_states))
    grid.set_state(starting_states[starting_idx])

    
    s = grid.current_state() 
    states_rewards = [(s,0)] # a list of tuples of (state, reward)
    
    while not grid.game_over() : 
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state() 
        states_rewards.append((s,r))
        
    
    # calculate the returns by working backwards from the terminal state 
    G = 0 
    states_returns = [] 
    first = True 
    
    for s , r in reversed(states_rewards) : 
       
        # the value of the terminal state is 0 by definnation 
        # therefore we should be able to ignore the first state and ignore the last G 
        
        if first : 
            first = False 
        else : 
            states_returns.append((s,G))
        
        G = r + gamma * G 
        
    states_returns.reverse() # put it in order of states visited
    
    return states_returns



if __name__ == '__main__' : 
    
    # using standard grid with 0 for every step 
    grid = standard_grid() 
    
    # print rewards 
    print("rewards :")
    print_values(grid.rewards, grid)
    
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
    
    # initialize V(s) and returns 
    V = {} 
    returns = {} 
    states = grid.all_states()
    
    for s in states : 
        if s in grid.actions : 
            returns[s] = [] 
        else: 
            # terminal state or a state we can't get to 
            V[s] = 0 
            
    # repeat 
    for t in range(100): 
        
        # generate an episode 
        states_returns = play_game(grid, policy)
        seen_states = set() 
        
        for s , G in states_returns : 
            # check the state has already been seen 
            # first-visit monte carlo 
            if s not in seen_states : 
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)
    
    print("values :")
    print_values(V, grid)
    print("Policy :")
    print_policy(policy, grid)
    