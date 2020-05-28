# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:20:16 2020

@author: wyckliffe
"""


import numpy as np
import matplotlib.pyplot as plt

grid = np.load('new_york.npy')
plt.imshow(grid)
plt.tight_layout()
plt.show()

