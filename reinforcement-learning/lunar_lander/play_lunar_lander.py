# -*- coding: utf-8 -*-
"""
Created on Tue May 12 00:32:56 2020

@author: wyckliffe
"""


import gym 
import numpy as np 

env = gym.make("LunarLander-v2")
qtable = np.load("qtable.npy")

class Play(): 
    
    def __init__(self): 
        
        env = gym.make("LunarLander-v2")
        qtable = np.load("qtable.npy")
        
        for i in range(100): 
            
            s = env.reset()
            state = self.discretize_state(s)
            
            for step in range(10000): 
                
                env.render() 
                
                action = np.argmax(qtable[state])
                
                (new_state, reward , done, _) = env.step(action)
                new_state = self.discretize_state(new_state)
                
                if done or step == 9999 : 
                    break 
                
                state = new_state
        
    
    def discretize_state(self, state): 
        
        # landing pad is always at coordinates (0,0)
        # first two numbers in state vector in the state vector 
        # reward for moving from the top of the screen to landing pad and zero speed is ~100..140 points 
        # if the land moves away from landing pad it loses reward back 
        # episode finishes if the land crashes or comes to rest , receiving additional -100 or + 100 points. 
        # Each leg ground contact is +10 
        # firing main engine is -0.3 points each frame. 
        # solved is 200 points 
        
        d_state   = list(state[:5])
        d_state[0]= int(0.5 * (state[0] + 0.7) * 10/2.0 )  # position x 
        d_state[1]= int(0.5 * (state[1] + 0.5) * 10/2.0 )  # position y
        d_state[2]= int(0.5 * (state[2] + 1.5) * 10/3.0 )  # velocity x
        d_state[3]= int(0.5 * (state[3] + 2.0) * 10/3.0 )  # velocity y 
        d_state[4]= int(0.5 * (state[4] + 3.142) * 10/(2 * 3.14159) )  # angle
        
        if d_state[0] >= 5 : 
            d_state[0] = 4 
            
        if d_state[1] >= 5 : 
            d_state[1] = 4 
            
        if d_state[2] >= 5 : 
            d_state[2] = 4 
            
        if d_state[3] >= 5 : 
            d_state[3] = 4 
            
        if d_state[4] >= 5 : 
            d_state[4] = 4 
            
        if d_state[0] < 0 : 
            d_state[0] = 0

        if d_state[1] < 0 : 
            d_state[1] = 0

        if d_state[2] < 0 : 
            d_state[2] = 0

        if d_state[3] < 0 : 
            d_state[3] = 0      
            
        if d_state[4] < 0 : 
            d_state[4] = 0
        
        
        return tuple(d_state)
    
Play()