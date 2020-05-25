# -*- coding: utf-8 -*-
"""
Created on Sun May 24 00:42:03 2020

@author: wyckliffe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser

# load dataset
data = pd.read_csv('appdata10.csv')

# clean the hour column
data['hour'] = data.hour.str.slice(1,3).astype(int)

#