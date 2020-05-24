# -*- coding: utf-8 -*-
"""
Created on Sat May 23 01:14:41 2020

@author: wyckliffe
"""
import os
import numpy as np
import random as rn
import env
import network
import q_learning

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# set the parameters
epsilon = .3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5


# build the environment
env = env.Env(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# build the network
brain = network.Network(learning_rate = 0.00001, number_actions = number_actions)

# build the deep q learning
dqn = q_learning.DQN(max_memory = max_memory, discount = 0.9)

# chose the mode
train = True

# train the AI
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

if (env.train):
    # loop over all epochs (1 epoch = 5 months)
    for epoch in range(1, number_epochs):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        # loop over all time steps( 1 timestep = 1 min)
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
            # explore
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            # exploit
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step

            # upda the environment
            next_state, reward, game_over = env.update(direction, energy_ai, int(timestep / (30 * 24 * 60)))
            total_reward += reward

            # store the transition into memory
            dqn.remember([current_state, action, reward, next_state], game_over)

            # Get the batches
            inputs, targets = dqn.get_batch(model, batch_size = batch_size)

            # calculate loss
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state

        # verbose
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))

        # if early stoping
        if (early_stopping):
            if (total_reward <= best_total_reward):
                patience_count += 1
            elif (total_reward > best_total_reward):
                best_total_reward = total_reward
                patience_count = 0
            if (patience_count >= patience):
                print("Early Stopping")
                break
        # save the model
        model.save("model.h5")
