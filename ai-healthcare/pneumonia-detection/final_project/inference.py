# -*- coding: utf-8 -*-

import numpy as np
import pydicom
import keras
from keras.models import model_from_json
from skimage.transform import resize

# This function reads in a .dcm file, checks the important fields for our device, and returns a numpy array
# of just the imaging data
def check_dicom(filename):

    print('Load file {} ...'.format(filename))
    ds = pydicom.dcmread(filename)
    if (ds.BodyPartExamined !='CHEST'):
        print("The body part is invalid")
        img = None
    elif (ds.Modality != 'DX'):
        print("The Modality is invalid")
        img = None
    elif (ds.PatientPosition not in ['AP', 'PA']):
        print('The Patient Position is invalid')
        img = None
    else:
        img = ds.pixel_array

    return img

# This function takes the numpy array output by check_dicom and
# runs the appropriate pre-processing needed for our model input
def preprocess_image(img, img_size):

    img_mean = np.mean(img)
    img_std  = np.std(img)
    img = (img - img_mean)/img_std
    img = resize(img, img_size)
    return img

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
    if prediction[0] == 1:
        prediction = 'Presence of Pneumonia highly probable'
    else:
        prediction = 'Presence of Pneumona less probable'

    return prediction

test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']

model_path ='my_model.json'
weight_path = 'xray_classmodel.best.hdf5'

IMG_SIZE=(1,224,224,3)

my_model = load_model(model_path, weight_path)
thresh = 0.60

# use the .dcm files to test the prediction
for i in test_dicoms:

    img = np.array([])
    img = check_dicom(i)

    if img is None:
        continue

    img_proc = preprocess_image(img, IMG_SIZE)
    pred = predict_image(my_model,img_proc,thresh)
    print(pred)

