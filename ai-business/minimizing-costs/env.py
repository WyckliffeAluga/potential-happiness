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





if __name__ == '__main__' :
    e = Env()
    print((e.monthly_atmospheric_temperatures))