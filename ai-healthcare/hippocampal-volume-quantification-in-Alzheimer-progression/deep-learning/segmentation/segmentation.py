# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:37:33 2020

@author: wyckliffe
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from collections import OrderedDict
import torch.optim as optim

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        # This parameter controls how far the UNet blocks grow as you go down
        # the contracting path
        features = init_features

        # set up the layers
        self.encoder1 = self.unet_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.unet_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.unet_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.unet_block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.unet_block(features * 8, features * 16, name="bottleneck")

        # Note the transposed convolutions here. These are the operations that perform
        # the upsampling.
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self.unet_block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self.unet_block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self.unet_block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self.unet_block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.softmax = nn.Softmax(dim = 1)



    def forward(self, x):
        # Contracting/downsampling path. Each encoder here is a set of 2x convolutional layers
        # with batch normalization, followed by activation function and max pooling
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # This is the bottom-most 1-1 layer.
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Expanding path. Note how output of each layer is concatenated with the downsampling block
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out_conv = self.conv(dec1)

        return self.softmax(out_conv)

    # This method executes the "U-net block"
    def unet_block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

training_volume = nib.load("data/spleen1_img.nii.gz").get_fdata()
training_label = nib.load("data/spleen1_label.nii.gz").get_fdata()

plt.imshow(training_volume[:,:,5] + training_label[:,:,5]*500, cmap="gray")

print(np.unique(training_label))

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)


#input channel (one image at a time) and two output channels (background and label)
unet = UNet(1, 2)

# Move trainable parameters to the device
unet.to(device)

loss = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(unet.parameters(), lr=0.001)
optimizer.zero_grad()

sum(p.numel() for p in unet.parameters() if p.requires_grad)

# Set up the model for training
unet.train()

for epoch in range(0,10):
    for slice_ix in range(0,15):
        # extract the slice from the volume and convert it to tensor that the model
        #  normalize the volume to 0..1 range
        slc = training_volume[:,:,slice_ix].astype(np.single)/np.max(training_volume[:,:,slice_ix])

        # So create the missing dimensions.
        slc_tensor = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).to(device)

        # extract the slice from label volume into tensor .

        lbl = training_label[:,:,slice_ix]
        lbl_tensor = torch.from_numpy(lbl).unsqueeze(0).long().to(device)

        # Zero-out gradients from the previous pass
        optimizer.zero_grad()

        # Do the forward pass
        pred = unet(slc_tensor)

        # compute our loss function and do the backpropagation pass
        l = loss(pred, lbl_tensor)
        l.backward()
        optimizer.step()

    print(f"Epoch: {epoch}, training loss: {l}")


plt.imshow(pred.cpu().detach()[0,1])


unet.eval()

def inference(img):
    tsr_test = torch.from_numpy(img.astype(np.single)/np.max(img)).unsqueeze(0).unsqueeze(0)
    pred = unet(tsr_test.to(device))
    return np.squeeze(pred.cpu().detach())

level = 11

img_test = training_volume[:,:,level]
pred = inference(img_test)
plt.figure()
plt.imshow(pred[1])

#  convert this into binary mask using PyTorch's argmax function:
mask = torch.argmax(pred, dim=0)
plt.figure()
plt.imshow(mask)


mask3d = np.zeros(training_volume.shape)

for slc_ix in range(training_volume.shape[2]):
    pred = inference(training_volume[:,:,slc_ix])
    mask3d[:,:,slc_ix] = torch.argmax(pred, dim=0)


org_volume = nib.load("data/spleen1_img.nii.gz")
org_volume.affine

#save image
img_out = nib.Nifti1Image(mask3d, org_volume.affine)
nib.save(img_out, "data/out.nii.gz")