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


