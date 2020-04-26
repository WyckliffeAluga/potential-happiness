# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:13:54 2020

@author: wyckliffe
"""


# importing the libraries 
import numpy as np  # work with arrays 
import random 
import os 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# creating the architecture of the NN 

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        
        super(Network, self).__init__()
        self.input_size = input_size 
        self.nb_action  = nb_action 
        self.fc1 = nn.Linear(input_size, 30) # first layer
        self.fc2 = nn.Linear(30, nb_action)  # last layer

    def forward(self, state):
        """
        Performs forward propagation

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.

        Returns
        -------
        q_values : TYPE
            DESCRIPTION.

        """
        x = F.relu(self.fc1(state)) # activate the first layer
        q_values = self.fc2(x)      # get the q values 
        
        return q_values
    
# Implementing Experience Replay 

class ReplayMemory(object):
    
    def __init__(self, capacity):
        
        self.capacity = capacity   # max number of transitions to have in the memory of events
        self.memory   = []         # last transitions according to the capacity 
        
    def push(self, event):
        """
        Appends the last transitions into the memory
        Delete older transitions if > capacity

        Returns
        -------
        None.

        """
        self.memory.append(event)
        
        if len(self.memory) > self.capacity: 
            del self.memory[0]
        
    def sample(self, batch_size): 
        """
        Parameters
        ----------
        batch_size : TYPE

        Returns
        -------
        TYPE
            samples

        """
    
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x : Variable(torch.cat(x,0)), samples)
        
    
# Implement Deep Q Learning 

class Dqn(): 
    
    def __init__(self, input_size, nb_action, gamma):
        
        self.gamma = gamma
        self.reward_window = [] # 
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) # 100,000 transitions 
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        """
        Selects the best action

        Parameters
        ----------
        state : TYPE
        
        Returns
        -------
        TYPE
            Action.

        """
        
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 0) # T = 7
        action = torch.multinomial(probs, 1) # maybe edit 
        return action.data[0,0]
      
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """
        Implement learning and back propagation 

        Parameters
        ----------
        batch_state : TYPE
            DESCRIPTION.
        batch_next_state : TYPE
            DESCRIPTION.
        batch_reward : TYPE
            DESCRIPTION.
        batch_action : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()
        
    def update(self, reward, new_signal): 
        """
        

        Parameters
        ----------
        reward : TYPE
            DESCRIPTION.
        new_signal : TYPE
            DESCRIPTION.

        Returns
        -------
        action : TYPE
            DESCRIPTION.

        """
        
        new_state   = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        
        action = self.select_action(new_state)
        
        if len(self.memory.memory) > 100: 
             batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
             self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        
        if len(self.reward_window) > 1000: 
            del self.reward_window[0]
            
        return action
    
    
    def score(self): 
        """
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self): 
        """
        

        Returns
        -------
        None.

        """
        
        torch.save({"state_dict": self.model.state_dict() , 
                    "optimizer" : self.optimizer.state_dict()}
                   ,"last_brain.pth")
    
    def load(self): 
        """
        

        Returns
        -------
        None.

        """
        
        if os.path.isfile("last_brain.pth"): 
            print("--> loading checkpoint......")
            checkpoint = torch.load("last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["state_dict"])
            print("Done!")
            
        else: 
            print("No checkpoint found!!!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        