# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:43:44 2020
@author: wyckliffe
"""

import os
import numpy as np
import random as rn
from keras.models import load_model
import env

# Setting seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETERS
number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5

env = env.Env(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)
model = load_model("model.h5")
train = False

env.train = train
current_state, _, _ = env.observe()
for timestep in range(0, 12 * 30 * 24 * 60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    
    if (action - direction_boundary < 0):
        direction = -1
    else:
        direction = 1

    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update(direction, energy_ai, int(timestep / (30 * 24 * 60)))
    current_state = next_state

    print("Simulation timestep: {:03d}/{:03d}".format(timestep , 12*30*24*60))

print("\n")
print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))
