# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:16:40 2020

@author: wyckliffe
"""


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# load the image
img = nib.load('datasets/volume.nii')

# show image
image = img.get_fdata()
plt.imshow(image[100,:,:], cmap='gray')

# orthographic projectsion
vr = np.zeros((image.shape[1], image.shape[2]))

for z in range(image.shape[0]):
    vr += image[z,:,:]
plt.imshow(nd.rotate(vr, 90), cmap="gray")

# max intensity projection
mip = np.zeros((image.shape[0], image.shape[2]))
for z in range(image.shape[1]):
    mip = np.maximum(mip, image[:,z,:])
plt.imshow(nd.rotate(mip, 90), cmap='gray')