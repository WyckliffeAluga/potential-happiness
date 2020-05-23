# -*- coding: utf-8 -*-
"""
Created on Sat May 23 01:14:41 2020

@author: wyckl
"""


import os
import numpy as np
import random as rn
import env
import network
import q_learning


# set seeds
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(1234)

epsilon = 0.3
number_actions = 5
direction_boundary = (number_actions -1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# Build the enviroment
env = env.Env(temp_range=(10.0, 24.0), initial_month=0, initial_number_of_users=20, initial_rate_data=30)

# launch network
brain = network.Network(learning_rate=0.0001, number_actions=number_actions)

# launch q_learning
dqn = q_learning.DQN(max_memory=max_memory, discount=0.9)

# mode
train = True


# train the AI
env.train = train

model = brain.model

# 1 epoch = 5 months
if (env.train) :
    for epoch in range(1, number_actions) :
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month=new_month)
        game_over = False
        current_state, _, _, = env.observe()

        timestep = 0
        # one time step is 1 minute
        while ((not game_over) and timestep < 5 * 30 * 24 * 60) :
            # play next action by exploration
            if (np.random.rand() < epsilon) :
                action = np.random.randint(0,number_actions)
                if (action - direction_boundary < 0) : # cooling down the server
                    direction = -1
                else:
                    direction = 1
                ai_energy = abs(action - direction_boundary) * temperature_step
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0) :
                    direction = -1
                else:
                    direction = 1
                ai_energy = abs(action - direction_boundary) * temperature_step

            # update the environment
            month = int(timestep/ (30*24*60))
            next_state, reward, game_over = env.update(direction, ai_energy, month)
            total_reward += reward

            # store transitions
            transition = [current_state, action, reward, next_state]
            dqn.remember(transition, game_over)

            # gather inputs and targets
            inputs, targets = dqn.get_batch(model=model, batch_size=batch_size)

            # computing the loss
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print('Total Energy spent with an AI: {:.0f}'.format(env.ai_total_energy))
        print("Total Energy spent without AI: {:.0f}".format(env.no_ai_total_energy))

        # save model
        model.save("model.h5")

