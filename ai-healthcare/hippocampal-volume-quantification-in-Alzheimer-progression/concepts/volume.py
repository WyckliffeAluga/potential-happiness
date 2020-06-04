# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:02:12 2020

@author: wyckliffe
"""
import pydicom
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# load the volume dataset
path = f'volume'
slices = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]
slices = sorted(slices, key = lambda dcm: dcm.ImagePositionPatient[0])

# check the dimensions
print(f"{len(slices)} of size {slices[0].Rows}x{slices[0].Columns}")

# check modularity
print(f"Modality: {slices[0].Modality}")

# pixel spacing
print(f"Pixel Spacing: {slices[0].PixelSpacing}, slice thickness: {slices[0].SliceThickness}")

#load into an array
img = np.stack([s.pixel_array for s in slices])
print(img.shape)

print("Saving axial: ")
axial = img[img.shape[0]//2]
plt.figure()
plt.imshow(axial, cmap="gray")
im = Image.fromarray((axial/np.max(axial)*0xff).astype(np.uint8), mode="L")
im.save("axial.png")


print("saving sagittal.............")
sagittal = img[:,:,img.shape[2]//2]
aspect = slices[0].SliceThickness / slices[0].PixelSpacing[0]
print(sagittal.shape)
plt.figure()
plt.imshow(sagittal, cmap='gray', aspect=aspect)
im = Image.fromarray((sagittal/np.max(sagittal)*0xff).astype(np.uint8), mode='L')
im = im.resize((sagittal.shape[1], int(sagittal.shape[0] * aspect)))
plt.imshow(im, cmap='gray')
im.save("Sagittal.png")

print("Saving coronal: ")
coronal = img[:, img.shape[1]//2, :]
aspect = slices[0].SliceThickness / slices[0].PixelSpacing[0]
print(coronal.shape)
plt.figure()
plt.imshow(coronal, cmap="gray", aspect = aspect)
im = Image.fromarray((coronal/np.max(coronal)*0xff).astype(np.uint8), mode="L")
im = im.resize((coronal.shape[1], int(coronal.shape[0] * aspect)))
im.save("coronal.png")
