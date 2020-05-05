# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:57:09 2020

@author: wyckliffe
"""

# import the libraries 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# import the packages for OpenAI and DOOM 
import gym 
from gym.wrappers import  SkipWrapper
from ppaquatte_gym_doom.wrappers.action_space import ToDiscrete

# import the internal files 
import experience_replay, image_preprocessing 
 

# Build the convolutional neural network and the artificial neural network (brain)

class CNN(nn.Module) : 
    
   def __init__(self, number_actions):
       
       super(CNN, self).__init__()
       self.convolutional1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) # B/W image input --> image with 32 new images
       self.convolutional2 = nn.Conv2d(in_channels = 32,out_channels = 32, kernel_size = 3) # input from conv 1 
       self.convolutional3 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = 2) # input from conv 2 
       
       self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40) # first artificial neural network
       self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
       
   def count_neurons(self, image_dim):
       
       x = Variable(torch.rand(1, *image_dim)) # * pass elements of a tuple as a list 
       x = F.relu(F.max_pool2d(self.convolutional1(x), 3, 2)) # step 1--> convolution layer 1, step 2 --> Max Pooling , step 3 --> rectifier activation function
       x = F.relu(F.max_pool2d(self.convolutional2(x), 3, 2)) # step 1--> convolution layer 2, step 2 --> Max Pooling , step 3 --> rectifier activation function
       x = F.relu(F.max_pool2d(self.convolutional3(x), 3, 2)) # step 1--> convolution layer 3, step 2 --> Max Pooling , step 3 --> rectifier activation function
       
       return x.data.view(1, -1).size(1)

   def forward(self, x): 
       
       # Convolution network
       x = F.relu(F.max_pool2d(self.convolutional1(x), 3, 2)) # step 1--> convolution layer 1, step 2 --> Max Pooling , step 3 --> rectifier activation function
       x = F.relu(F.max_pool2d(self.convolutional2(x), 3, 2)) # step 1--> convolution layer 2, step 2 --> Max Pooling , step 3 --> rectifier activation function
       x = F.relu(F.max_pool2d(self.convolutional3(x), 3, 2)) # step 1--> convolution layer 3, step 2 --> Max Pooling , step 3 --> rectifier activation function
       
       # flatten
       x = x.view(x.size(0) , -1)
       
       # Neural network 
       x = F.relu(self.fc1(x))  # step 1 --> propagate signals from the flattening step2--> break linearity with a rectifier 
       x = self.fc2(x)
       
       return x 
   
# Build how the ai will respond to CNN (body)
       
class SoftmaxBody(nn.Module): 
    
    def __init__(self, T): 
        
        super(SoftmaxBody, self).__init__()
        self.T = T 
        
    def forward(self, output):  
        
        # actions left, right, forward, shoot , run 
        
        probs = F.softmax(output * self.T)
        actions = probs.multinomial()
        
        return actions
        
 
# Build the AI 
class AI: 
    
    def __init__(self, brain, body): 
        
        self.brain = brain 
        self.body  = body 
        
     
    def __call__(self, inputs): 
        
        # format the images 
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32))) # step 1 --> convert image to numpy array. step 2 --> convert to torch tensor. step 3 --> convert to torch variable
        
        # insert the input into the "eyes" of the AI 
        output = self.brain(input) # output signal of the brain 
       
        # insert the signal into the body 
        actions = self.body(output)
        
        return actions.data.numpy() # return actions 
        
        
# Train the AI with Deep Convolutional Q-Learning 

# Getting the DOOM environment 
        
# Step 1: Import environment DoomCorridor-v0 
# Step 2: Preprocess the image and make sure it is 80 x 80 
# Step 3: Import the rest of the game and save the videos to videos folder
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)       

# create the number of actions 
number_of_actions = doom_env.action_space.n # make it able to wreck different DOOM environments 

# Build the AI 
cnn = CNN(number_of_actions) # brain 
softmax_body = SoftmaxBody(T = 1.0) # body 
ai = AI(cnn, softmax_body) # and now we have an AI ready to be trained



      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       



