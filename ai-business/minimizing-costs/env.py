# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:32:28 2020

@author: wyckliffe
"""

import numpy as  np


class Env:

    def __init__(self, temp_range=(10.0, 24.0), initial_month=0, initial_number_of_users=10, initial_rate_data=60) :

        self.monthly_atmospheric_temperatures = [1.0, 5.0, 7.0, 10.0, 11.0,23.0, 20.0, 24.0, 22.0, 10.0, 5.0 , 1.0]



if __name__ == '__main__' :
    e = Env()
    print((e.monthly_atmospheric_temperatures))