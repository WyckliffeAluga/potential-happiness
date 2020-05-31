# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:40:53 2020

@author: wyckliffe
"""
from PIL import Image
import matplotlib.pyplot as plt
import  numpy as np
import scipy.ndimage as nd

# work on the phantom file
im = Image.open('datasets/phantom.bmp').convert('L')

# compute a projection
projection = []
m = np.array(im)
for x in range(0, im.width):
    projection.append(np.sum(m[x]))


# rotate the image
im_r = im.rotate(60)
plt.imshow(im_r, cmap='gray')

# projection of the rotation
projection = []
m = np.array(im_r)
for x in range(0, im_r.width):
    projection.append(np.sum(m[x]))
plt.plot(projection)

# compute the sinogram

projections = []
for angle in range(0, 180, 3):
    p = []
    im_r = im.rotate(angle)
    for x in range(0, im.width):
        p.append(np.sum(np.array(im_r)[x]))
    projections.append((angle, p))

for p in projections:
    if (p[0] % 15) == 0:
        plt.figure()
        plt.plot(p[1])


sinogram = np.stack([p[1] for p in projections])
plt.imshow(sinogram, cmap="gray", aspect = 2)


bp = np.zeros((im.width, im.height))

for y in range(sinogram.shape[1]):
    bp[:, y] = projections[0][1]

plt.imshow(bp, cmap="gray")

# backprojection
bp = np.zeros((im.width, im.height))

for p in projections:

    # Smear the projection
    img = np.zeros(bp.shape)
    for y in range(img.shape[1]):
        img[:, y] = p[1]

    # Rotate the projection back
    img = nd.rotate(img, -p[0], reshape = False)

    bp += img

plt.imshow(bp, cmap ="gray")


# work on the brain image
# Load and display the image
b_im = Image.open("datasets/sample_brain.png").convert("L")
plt.imshow(b_im, cmap="gray")


# Obtain the sinogram

b_projections = []
for angle in range(0, 180, 1):
    p = []
    im_r = b_im.rotate(angle)
    for x in range(0, b_im.width):
        p.append(np.sum(np.array(im_r)[x]))
    b_projections.append((angle, p))

plt.imshow(np.stack([p[1] for p in b_projections]), cmap="gray", aspect = 2)


# Reconstruct through backprojection

b_bp = np.zeros((b_im.width, b_im.height))

for p in b_projections:

    # Smear the projection
    img = np.zeros(b_bp.shape)
    for y in range(b_bp.shape[1]):
        img[:, y] = p[1]

    # Rotate the projection back
    img = nd.rotate(img, -p[0], reshape = False)

    b_bp += img

# Display result

plt.imshow(b_bp, cmap ="gray")