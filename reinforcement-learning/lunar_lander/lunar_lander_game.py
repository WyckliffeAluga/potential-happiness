# -*- coding: utf-8 -*-
"""
Created on Mon May 11 23:18:33 2020

@author: wyckl
"""


import gym 
import random
import numpy as np 

class Game(): 
    
    def __init__(self): 
        
        self.env = gym.make("LunarLander-v2")
        self.actions = self.env.action_space.n # no op, fire left engine, fire right engine main 
        
        # run 10, 000 episodes 
        
        (max_rewards, last_reward, qtable) = self.run(num_episodes=10000, 
                                                      alpha=0.1, gamma=0.95, explore_multiple=0.995)
        print(np.mean(last_reward) , np.max(max_rewards), np.mean(max_rewards))
        
        np.save('qtable.npy', qtable)
        
    def random_choices(self): 
        self.env.reset()
        
        for rep in range(10): 
            i = 0 
            
            while i < 100 : 
                self.env.step(random.choice(([0,1,2,3])))
                self.env.render() 
                i += 1 
            self.env.reset()
            
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
    
    def run(self, num_episodes, alpha, gamma, explore_multiple) : 
        
        max_rewards = [] 
        last_reward = [] 
        
        q_table = np.subtract(np.zeros((5, 5, 5, 5, 5 , self.actions)), 100) # start all rewards at -100
        explore_rate = 1.0 
        
        for episode in range(num_episodes): 
            s = self.env.reset() 
            state = self.discretize_state(s)
            
            for step in range(10000) : 
                
                # select an action (we have to do something)
                if random.random() < explore_rate :  # the explore rate will decrease over time 
                    action = random.choice(range(self.actions)) # we want the agent to randomly decide early on
                
                else: 
                    action = np.argmax(q_table[state]) # figure out what the best action is using argmax
                    
                (new_state , reward , done , _) = self.env.step(action)
                new_state = self.discretize_state(new_state)
                
                # update Q 
                best_future_q =  np.amax(q_table[new_state])  # return best possible reward from next state 
                prior_value   =  q_table[state + (action, )]
                q_table[state + (action, )] = (1.0 - alpha)*prior_value + \
                                             alpha*(reward + gamma * best_future_q)
                state = new_state
                
                if done or step == 9999 : 
                    last_reward.append(reward)
                    break 
                
                
            if explore_rate > 0.01 : 
                explore_rate *= explore_multiple
            max_rewards.append(np.amax(q_table))
            
        return (max_rewards, last_reward[-50:] , q_table) # return rewards from last episodes 
    
    def check_run(self): 
        
        num_episodes = 300 
        for alpha in [0.05, 0.10, 0.15] : 
            for gamma in [0.85, 0.90, 0.95] : 
                (max_rewards, last_reward, _) = self.run(num_episodes=num_episodes, 
                                                         alpha= alpha, 
                                                         gamma = gamma, 
                                                         explore_multiple = 0.995)
                
                print(alpha, gamma, np.mean(last_reward), np.mean(max_rewards))
            
g = Game()