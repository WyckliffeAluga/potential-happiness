# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:51:53 2020

@author: wyckliffe
"""


import numpy as np 



class Grid:  # create the environment 

    def __init__(self, width, height, start): 
        
        self.width = width 
        self.height = height 
        self.i = start[0]
        self.j = start[1]
        
    
    def set(self, rewards, actions) : 
        # rewards should be in a dict of (i, j) : r
        # actions should be in a dict of (i ,j) : A
        
        self.rewards = rewards
        self.actions = actions 
        
    def set_state(self, s): 
        self.i = s[0]
        self.j = s[1]
        
    def current_state(self): 
        
        return (self.i , self.j)
    
    def is_terminal(self, s): 
        
        return s not in self.actions 
    
    def move(self, action): 
        
        # check if legal move first 
        if action in self.actions[(self.i, self.j)]: 
            
            if action == "U": 
                self.i -= 1 
            
            elif action == "D" : 
                self.i =+ 1 
                
            elif action == "R": 
                self.j += 1 
            
            elif action == "L" : 
                self.j -= 1
        
        # return rewards if any 
        return self.rewards.get((self.i, self.j) , 0)
    
    def undo_move(self, action): 
        
        # this is the opposite of what move will do 
        
        if action == "U": 
            self.i += 1 
        
        elif action == "D": 
            self.i -= 1 
        
        elif action == "R" : 
            self.j -= 1 
    
        elif action == "L" : 
            self.j += 1
        
        # raise an exception if we arrive somewhere we shouldn't be 
        assert(self.current_state() in self.all_states())
        
    
    def game_over(self): 
        
        # returns true if fame is over, else falste 
        # true if we are in a state where no actions are possible 
        
        return (self.i, self.j) not in self.actions 
    
    def all_states(self): 
        
        # get all states either a position that is possible next actions or a position that will yield a reward 
        
        return set(self.actions.keys()) | set(self.rewards.keys())
        

def standard_grid(): 
    
    # define grit that described the reward fro arriving at each state 
    # possible actions at each state 
    # that is (x) means you cant go there 
    # (s) means start position 
    
      g = Grid(3, 4, (2, 0))
      rewards = {(0, 3): 1, (1, 3): -1}
      actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
      }
      
      g.set(rewards, actions)
      return g
  
def negative_grid(step_cost=-0.1):
    
  # try to minimize the number of moves
  # penalize every move
  g = standard_grid()
  g.rewards.update({
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (0, 3): 1 ,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (1, 3): -1,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
  })
  
  return g