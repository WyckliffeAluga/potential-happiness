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
from torch.autograd import  Variable

# creating the architecture of the NN 

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        
        super(Network, self).__init__()
        self.input_size = input_size 
        self.nb_action  = nb_action 
        self.fc1 = nn.Linear(input_size, 30) # first layer
        self.fc2 = nn.Linear(30, nb_action)  # last layer

    def forward(self, state):
        
        x = F.relu(self.fc1(state)) # activate the first layer
        q_values = self.fc2(x)      # get the q values 
        
        return q_values
    
# Implementing Experience Replay 
        