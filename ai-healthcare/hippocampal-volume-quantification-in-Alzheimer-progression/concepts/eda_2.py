# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 20:05:02 2020

@author: wyckliffe
"""


import pydicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import numpy as np
import os

plt.rcParams['figure.figsize'] = (10, 10)
path = f"samples\DICOM CT study\JohnDoe_CT_series"
slices = [pydicom.dcmread(os.path.join(path, f)) for f in sorted(os.listdir(path))]

print(len(slices))

print(f"Pixel Spacing: {slices[0].PixelSpacing}")
print(f"Slice Thickness: {slices[0].SliceThickness}mm")
print(f"Pixel Spacing: {slices[0][0x0028,0x0030].value}")

# extra the pixel data
image_data = np.stack([s.pixel_array for s in slices])

print(image_data.shape)
print(image_data.dtype)

# visualize a slice

image = image_data[115, :,:]
plt.imshow(image, cmap='gray')

# visualize a bunch of slices
fig, ax = plt.subplots(5, 5, figsize=[10,10])

for i in range(25) :
    ix = i * int(len(slices)/25)
    ax[int(i/5) , int(i%5)].set_title(f"slice {ix}")
    ax[int(i/5) , int(i%5)].imshow(image_data[ix, :, :], cmap='gray')
    ax[int(i/5) , int(i%5)].axis('off')

plt.show()

# coronal slice
img_coronal = image_data[:, 250, :]
plt.imshow(img_coronal, cmap='gray')

# scale the images
aspect_ratio = slices[0].SliceThickness / slices[0].PixelSpacing[0]

plt.imshow(img_coronal, cmap='gray', aspect=aspect_ratio)

# crop the image to visualize
img_crop = image[110:400, :]
plt.imshow(img_crop, cmap=cm.Greys_r)

# visualize the histogram
p = img_crop.flatten()
vals, bins , ignored = plt.hist(p, bins = 200)
plt.show()

print(img_crop.max())
print(img_crop.min())

# bottom ends
np.sort(np.unique(img_crop))

# apply the window
min_ = 2000
max_ = 4000

window_img = np.copy(img_crop)
window_img[np.where(window_img < min_)]  = min_
window_img[np.where(window_img > max_)]  = max_

plt.imshow(window_img, cmap='gray')

# zoom in
plt.imshow(img_crop[15:60, 380:430], cmap="gray")

# use numpy mask
masked = ma.masked_outside(img_crop, min_, max_)
np.sort(np.unique(img_crop[~masked.mask]))

_ = plt.hist(masked.flatten(), bins=15)
plt.show()

print(f"range: {img_crop.max()-img_crop.min()}")
