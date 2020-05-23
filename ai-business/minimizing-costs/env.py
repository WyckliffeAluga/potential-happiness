# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:32:28 2020

@author: wyckliffe
"""

import numpy as  np


class Env:

    def __init__(self, temp_range=(10.0, 24.0), initial_month=0, initial_number_of_users=10, initial_rate_data=60) :

        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0,23.0, 20.0, 24.0, 22.0, 10.0, 5.0 , 1.0]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[self.initial_month]
        self.optimal_temperature = temp_range
        self.min_temperature = -20
        self.max_temperatue = 90
        self.min_number_users = 10
        self.max_number_users = 100
        self.max_update_users = 5
        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10
        self.initial_number_of_users = initial_number_of_users
        self.current_number_users = initial_number_of_users
        self.initial_rate_data = initial_rate_data
        self.current_rate_data = initial_rate_data

        # linear model approximation (remember assumption 1)
        self.server_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.ai_temperature = self.server_temperature
        self.no_ai_temperature = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0

        # energy initialization
        self.ai_total_energy = 0.0
        self.no_ai_total_energy = 0.0

        # other ai housekeeping stuff
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    def update(self, direction, ai_energy, month) :

        # compute the energy spent by server's cooling system
        no_ai_energy = 0
        if (self.no_ai_temperature < self.optimal_temperature[0]) : # if temp is low
            no_ai_energy = self.optimal_temperature[0] - self.no_ai_temperature # has to bring it up to min at least
            self.no_ai_temperature = self.optimal_temperature[0]

        elif (self.no_ai_temperature > self.optimal_temperature[1]) : # if temp is high
            no_ai_energy = self.no_ai_temperature - self.optimal_temperature[1] # has to bring it down to max at least
            self.no_ai_temperature = self.optimal_temperature[1]

        self.reward = no_ai_energy - ai_energy

        # scale the reward to stabilize DQN computation
        self.reward = self.reward * 1e-3

        # get the next state
        # update atmospheric temperature
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]

        # update the number of users
        self.current_number_users += np.random.randint(-self.max_update_users, self.max_update_users)

        if (self.current_number_users > self.max_number_users) :
            self.current_number_users = self.max_number_users

        elif (self.current_number_users < self.min_number_users):
            self.current_number_users = self.min_number_users

        # update the rate of data transfer
        self.current_rate_data += np.random.randint(-self.max_update_data, self.max_update_data )

        if (self.current_rate_data > self.max_rate_data) :
            self.current_rate_data = self.max_rate_data

        elif (self.current_rate_data < self.min_rate_data) :
            self.current_rate_data = self.min_rate_data

        # update intrinsic temperature and compute delta server temperature
        previous_server_temperature = self.server_temperature
        self.server_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        delta_server_temperature = self.server_temperature - previous_server_temperature

        # compute the delta of temperature caused by ai

        if (direction == -1) :
            delta_ai_temperature = - ai_energy
        elif (direction == 1) :
            delta_ai_temperature = ai_energy

        # update the server temp caused by ai
        self.ai_temperature += delta_server_temperature + delta_ai_temperature

        # update the server temp when no ai
        self.no_ai_temperature += delta_server_temperature

        # get game over !!!!!!!!!
        if (self.ai_temperature < self.min_temperature):
            if (self.train == 1) :
                self.game_over = 1
            else:
                self.ai_total_energy += self.optimal_temperature[0] - self.ai_temperature # check this to confirm later (debug point)
                self.ai_temperature = self.optimal_temperature[0]
        elif(self.ai_temperature > self.max_temperatue) :
            if (self.train ==1) :
                self.game_over = 1
            else:
                self.ai_total_energy += self.optimal_temperature[1] - self.ai_temperature
                self.ai_temperature = self.optimal_temperature[1]


        # update scores

        # update total energy spent by the ai
        self.ai_total_energy += ai_energy
        # update total energy spent by server cooling systtem
        self.no_ai_total_energy += no_ai_energy

        # scale the next states for the neural network
        scaled_ai_temperature = (self.ai_temperature - self.min_temperature) / (self.max_temperatue - self.min_temperature)
        scaled_number_users   = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data  = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)

        next_state = np.matrix([scaled_ai_temperature, scaled_number_users, scaled_rate_data])

        return next_state, self.reward, self.game_over

    def reset(self, new_month) :

        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.initial_month = new_month
        self.current_number_users =  self.initial_number_of_users
        self.current_rate_data = self.initial_rate_data
        self.server_temperature = self.atmospheric_temperature + 1.25 * self.current_number_users + 1.25 * self.current_rate_data
        self.ai_temperature = self.server_temperature
        self.no_ai_temperature = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.ai_total_energy = 0.0
        self.no_ai_total_energy = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    def observe(self) :

        scaled_ai_temperature = (self.ai_temperature - self.min_temperature) / (self.max_temperatue - self.min_temperature)
        scaled_number_users   = (self.current_number_users - self.min_number_users) / (self.max_number_users - self.min_number_users)
        scaled_rate_data  = (self.current_rate_data - self.min_rate_data) / (self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_ai_temperature, scaled_number_users, scaled_rate_data])

        return current_state, self.reward, self.game_over
