# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:57:10 2020

@author: wyckliffe
"""


import numpy as np
import pandas as pd
import pydicom

import matplotlib.pyplot as plt
import keras
from keras.models import model_from_json
from skimage.transform import resize

# This function reads in a .dcm file, checks the important fields for our device, and returns a numpy array
# of just the imaging data
def check_dicom(filename):
    # todo

    print('Load file {} ...'.format(filename))
    ds = pydicom.dcmread(filename)
    img = ds.pixel_array

    return img


# This function takes the numpy array output by check_dicom and
# runs the appropriate pre-processing needed for our model input
def preprocess_image(img, img_size, img_size, img_size):

    img = (img - img_mean)/img_std
    img = resize(img, img_size)
  
    return proc_img

# This function loads in our trained model w/ weights and compiles it
def load_model(model_path, weight_path):

    json_file = open(model_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_path)

    return model

# This function uses our device's threshold parameters to predict whether or not
# the image shows the presence of pneumonia using our trained model
def predict_image(model, img, thresh):

    prediction = (model.predict_proba(img) >= thresh).astype(int)

    return prediction

test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']

model_path ='my_model.json'
weight_path = 'xray_classmodel.best.hdf5'

IMG_SIZE=(1,224,224,3) # This might be different if you did not use vgg16
#img_mean = # loads the mean image value they used during training preprocessing
#img_std = # loads the std dev image value they used during training preprocessing

my_model = load_model(model_path, weight_path)
thresh = 0.70 #loads the threshold they chose for model classification

# use the .dcm files to test your prediction
for i in test_dicoms:

    img = np.array([])
    img = check_dicom(i)
    img_mean = np.mean(img)
    img_std  = np.std(img)

    if img is None:
        continue

    img_proc = preprocess_image(img,IMG_SIZE)
    pred = predict_image(my_model,img_proc,thresh)
    print(pred)



