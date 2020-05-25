# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:32:55 2020

@author: wyckliffe
"""


import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque

# select the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience replay

class ReplayBuffer:

  def __init__(self, max_size=1e6):

    self.memory = []
    self.max_size = max_size
    self.index = 0

  def add(self, transition):

    if len(self.memory) == self.max_size:
      self.memory[int(self.index)] = transition
      self.index = (self.index + 1) % self.max_size
    else:
      self.memory.append(transition)

  def sample(self, batch_size):

    indexes = np.random.randint(0, len(self.memory), size=batch_size)

    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

    for i in indexes:

      state, next_state, action, reward, done = self.memory[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))

    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


# create the actor class
class Actor(nn.Module):

  def __init__(self, state_dim, action_dim, max_action):

    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):

    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x))

    return x

# create the critic class
# this class has two neural networks that will be build simultenous
class Critic(nn.Module):

  def __init__(self, state_dim, action_dim):

    super(Critic, self).__init__()
    # first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)

    # second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):

    xu = torch.cat([x, u], 1)

    # Forward-Propagation on the first Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)

    # Forward-Propagation on the second Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):

    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1


class TD3:

  def __init__(self, state_dim, action_dim, max_action):

    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    self.max_action = max_action

  def select_action(self, state):

    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

    for it in range(iterations):

      # sample batches from replay buffer
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)

      # play an actor target from next states and for a'
      next_action = self.actor_target(next_state)

      # add gaussian noise on the actions
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

      # get targes q's
      tq1, tq2 = self.critic_target(next_state, next_action)

      # get min (q1, q2)
      q = torch.min(tq1, tq2)

      # compute Qt
      Qt = reward + ((1 - done) * discount * q).detach()

      # get q1 or q2
      q1, q2 = self.critic(state, action)

      # compute loss
      loss = F.mse_loss(q1, Qt) + F.mse_loss(q2, Qt)

      # back propagate
      self.critic_optimizer.zero_grad()
      loss.backward()
      self.critic_optimizer.step()

      # update optimizers with delay
      if it % policy_freq == 0:

        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update weights of the actor target
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # update weights of the critic target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  # save trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

  # load function to load saved model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


def evaluate_policy(policy, env, eval_episodes=10):
  avg_reward = 0.

  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes

  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

# set the parameters
env_name = "AntBulletEnv-v0"
seed = 0
start_timesteps = 1e4 # number of iterations before start learning (at this time do random actions)
eval_freq = 5e3 # how often to evaluate
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True
expl_noise = 0.1 # Exploration noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma
tau = 0.005 # Target network update rate
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2


# create a file name for the two saved models: the Actor and Critic models
file_name = "%s_%s_%s" % ("Solution", env_name, str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

# create a folder inside which will be saved the trained models
if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

# instantiate the environment
env = gym.make(env_name)

# set the seeds
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# get the state dimensions, action dimensions and max actions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# create the policy network (the Actor model)
policy = TD3(state_dim, action_dim, max_action)

# create the Experience Replay memory
replay_buffer = ReplayBuffer()

# define a list where all the evaluation results over 10 episodes are stored
evaluations = [evaluate_policy(policy, env)]