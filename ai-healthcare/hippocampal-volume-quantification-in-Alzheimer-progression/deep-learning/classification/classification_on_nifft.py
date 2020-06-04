# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:31:53 2020

@author: wyckliffe
"""
import pydicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from PIL import Image
import math



# load the image
nii_img = nib.load("data/spleen.nii.gz")
img = nii_img.get_fdata()

# visualize a slice:
plt.imshow(img[:,:,0], cmap="gray")
nii_img.header["pixdim"]

plt.imshow(np.rot90(img[250,:,:]), cmap = "gray")

nii_img.header["pixdim"]

img2d = np.rot90(img[250,:,:])
plt.imshow(img2d, cmap = "gray", aspect=nii_img.header["pixdim"][3]/nii_img.header["pixdim"][1])


def display_volume_slices(img, w, h):
    plot_w = w
    plot_h = h

    # You can play with figsize parameter to adjust how large the images are
    fig, ax = plt.subplots(plot_h, plot_w, figsize=[35,35])

    for i in range(plot_w*plot_h):
        plt_x = i % plot_w
        plt_y = i // plot_w
        if (i < len(img)):
            ax[plt_y, plt_x].set_title(f"slice {i}")
            ax[plt_y, plt_x].imshow(img[i], cmap='gray')
        ax[plt_y, plt_x].axis("off")

    plt.show()

# visualize all slices
display_volume_slices(np.transpose(img, (2, 0, 1)), 7, 7)

plt.rcParams["figure.figsize"] = (7,7)

# Define a 4x4 edge filter kernel

conv_kernel = np.ones((4,4))
conv_kernel[2:,:] = -1
print(conv_kernel)

# 2D convolutions
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

slices = []

for ix in range(0, img.shape[2]):
    tensor = torch.from_numpy((img[:,:,ix].astype(np.single)/0xff)).unsqueeze(0).unsqueeze(1)
    convolved = conv2d(tensor)
    slices.append(np.squeeze(convolved.detach().numpy()))

plt.imshow(slices[15], cmap="gray")

display_volume_slices(slices, 7, 7)

print(f"Number of trainable parameters: {np.prod(conv2d.weight.shape)}")


# 3D convolutions
conv3d = nn.Conv3d(
    1,
    1,
    kernel_size = (4, 4, 4),
    bias = False)

conv_kernel3d = np.array([conv_kernel, conv_kernel, -conv_kernel, -conv_kernel])

params3d = torch.from_numpy(conv_kernel3d).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
conv3d.weight = torch.nn.Parameter(params3d)

tensor = torch.from_numpy((img.astype(np.single)/0xff)).unsqueeze(0).unsqueeze(1)

# Run convolution
convolved = conv3d(tensor)
convolved_np = np.transpose(np.squeeze(convolved.detach().numpy()), (2, 0, 1))

display_volume_slices(convolved_np, 7, 7)

print(f"Number of trainable parameters: {np.prod(conv3d.weight.shape)}")

patch_size = 16

def extract_patches(img, patch_size, z_level):

    # create holder array
    num_patches = (img.shape[0]//patch_size) * (img.shape[1]//patch_size)
    out = np.zeros((num_patches, 3, patch_size, patch_size))

    for p_x in range(0,img.shape[0]//patch_size):
        for p_y in range(0,img.shape[1]//patch_size):

            # Figure out where the patch should start in our main plane
            patch_start_x = p_x*patch_size
            patch_end_x = patch_start_x + patch_size
            x_center = patch_start_x + patch_size//2

            patch_start_y = p_y*patch_size
            patch_end_y = patch_start_y + patch_size
            y_center = patch_start_y + patch_size//2

            # Figure out where patch starts in ortho planes.
            # Note that we extract patches in orthogonal direction, therefore indices might go over
            # or go negative
            patch_start_z = max(0, z_level-patch_size//2)
            patch_end_z = patch_start_z + patch_size

            if (patch_end_z >= img.shape[2]):
                patch_end_z -= patch_end_z - img.shape[2]
                patch_start_z = patch_end_z - patch_size

            # Get axial, sagittal and coronal slices, assuming particular arrangement of respective planes in the
            # input image
            patch_a = img[patch_start_x:patch_end_x, patch_start_y:patch_end_y, z_level]
            patch_s = img[x_center, patch_start_y:patch_end_y, patch_start_z:patch_end_z]
            patch_c = img[patch_start_x:patch_end_x, y_center, patch_start_z:patch_end_z]

            patch_id = p_x*img.shape[0]//patch_size + p_y
            out[patch_id] = np.array([patch_a, patch_s, patch_c])

    return out


def build_slice_from_patches(patches):
    img_size = int(math.sqrt(patches.shape[0]))*patches.shape[1]

    out = np.zeros((img_size, img_size))

    for i in range(patches.shape[0]):
        x = i // (out.shape[0] // patches.shape[2])
        y = i % (out.shape[0] // patches.shape[2])

        x_start = x*patches.shape[2]
        x_end = x_start + patches.shape[2]

        y_start = y*patches.shape[2]
        y_end = y_start + patches.shape[2]

        out[x_start:x_end, y_start:y_end] = patches[i]

    return out

patches = extract_patches(img, patch_size, 40)
print(patches.shape)

# Rebuild and visualize axials
plt.imshow(build_slice_from_patches(np.squeeze(patches[:,0])))

# other planes
plt.imshow(build_slice_from_patches(np.squeeze(patches[:,1])))


conv25d = nn.Conv2d(
    3,
    1,
    kernel_size = (4, 4),
    bias = False)

print(conv25d.weight.shape)

conv_kernel25d = np.array([conv_kernel, conv_kernel, conv_kernel])

params = torch.from_numpy(conv_kernel25d).type(torch.FloatTensor).unsqueeze(0)
conv25d.weight = torch.nn.Parameter(params)


slices = []
for z in range(img.shape[2]):
    tensor = torch.from_numpy((extract_patches(img, patch_size, z).astype(np.single)/0xff))
    convolved = conv25d(tensor)
    slices.append(np.squeeze(convolved.detach().numpy()))


# Visualize a slice
plt.imshow(build_slice_from_patches(slices[20]), cmap="gray")

display_volume_slices([build_slice_from_patches(p) for p in slices], 7, 7)