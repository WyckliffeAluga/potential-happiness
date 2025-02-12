# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:11:01 2020

@author: wyckliffe
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import  Variable

# import data set
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# prepare training and testing sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
testing_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
testing_set = np.array(testing_set, dtype = 'int')

# get the total number of users and movies
nb_users = int(max(max(training_set[:,0]), max(testing_set[:,0])))
nb_movies =  int(max(max(training_set[:,1]), max(testing_set[:,1])))

# convert the data into an array where users are in rows and movies in columns and dump timestamps
def convert(data):
  new_data = []
  for id_users in range(1, nb_users+1):
    id_movies = data[:,1][data[:,0] == id_users]
    id_ratings = data[:,2][data[:,0] == id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies - 1] = id_ratings
    new_data.append(list(ratings))

  return new_data

training_set = convert(training_set)
testing_set = convert(testing_set)

# convert into torch tensors
training_set = torch.FloatTensor(training_set)
testing_set  = torch.FloatTensor(testing_set)

# convert the ratings into binary rating 1 (Liked) or 0 (not liked) and -1 (didn't rate)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

testing_set[testing_set == 0] = -1
testing_set[testing_set == 1] = 0
testing_set[testing_set == 2] = 0
testing_set[testing_set >= 3] = 1

class RBM():
    def __init__(self, nv, nh):

        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh) # bias for hidden nodes
        self.b = torch.randn(1, nv)  # bias for visible nodes

    def sample_h(self, x):

        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation) # probability that the hidden node is activated
        return p_h_given_v, torch.bernoulli(p_h_given_v)  # return the probabilities and the bernoulli distribution of the probabilities

    def sample_v(self, y):

        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation) # probability that the visible node is activated
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):

        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0])
nh = 1000
batch_size = 100
rbm = RBM(nv=nv, nh=nh)

# train the rbm
nb_epoch = 20
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = testing_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))

