# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:25:23 2020

@author: wyckliffe
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

plt.rcParams["figure.figsize"] = (7,7)

# Define a 4x4 edge filter kernel

conv_kernel = np.ones((4,4))
conv_kernel[2:,:] = -1
print(conv_kernel)


conv2d = nn.Conv2d(
    1, # Input size
    1, # Output size
    kernel_size = (4, 4), # size   filter kernel
    bias = False)
conv2d

# change into a tensor
params = torch.from_numpy(conv_kernel).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)

# the unsqeeze operation adds empty dimensions to the vector bringing it to 4 dimensions
# Torch expects parameter vector of size (output_channels, input_channels, kernel x dimension, kernel y dimension)

conv2d.weight = torch.nn.Parameter(params)

# load the walrus
walrus = Image.open('data/walrus.jpg')
plt.figure()
plt.imshow(walrus)

# convert into grayscale
walrus = walrus.convert("L")
walrus = np.array(walrus)
walrus = walrus.astype(np.single)/0xff
plt.figure()
plt.imshow(walrus, cmap="gray")


# convert into a tensor

walrus_tensor = torch.from_numpy(walrus).unsqueeze(0).unsqueeze(1)


convolved = conv2d(walrus_tensor)
relu = F.relu(convolved)
plt.figure()
plt.imshow(np.squeeze(convolved.detach().numpy()), cmap="gray")

plt.figure()
plt.imshow(np.squeeze(relu.detach().numpy()), cmap="gray")