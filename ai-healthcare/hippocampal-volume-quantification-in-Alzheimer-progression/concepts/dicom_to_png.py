# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:01:50 2020

@author: wyckliffe
"""


import pydicom
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


dcm = pydicom.dcmread('datasets/instance.dcm')
print(f"Modality : {dcm.Modality}")

pixels = np.copy(dcm.pixel_array)
print(f"Min: {np.min(pixels)}, Max: {np.max(pixels)}")

wc = 2472
ww = 4144

min_ = wc - ww/2
max_ = wc + ww/2

pixels[np.where(pixels < min_)] = min_
pixels[np.where(pixels > max_)] = max_
pixels = (pixels - min_) / (max_ - min_)

plt.imshow(pixels, cmap='gray')

png = (pixels*0xff).astype(np.uint8)
img = Image.fromarray(png, mode='L')
img.save('dicom.png')
