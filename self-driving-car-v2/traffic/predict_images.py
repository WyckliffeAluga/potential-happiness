# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:43:56 2020

@author: wyckliffe
"""
import requests
from PIL import Image 
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import pandas as pd 

    
def testImage(url): 
        
    res = requests.get(url, stream='true')
    assert(res.status_code == 200), "Get request failed"
        
    img = Image.open(res.raw)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    
    img = np.asarray(img)
    img = cv2.resize(img, (32,32))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    assert(img.shape == (32,32)), "Test image size not 32 x 32"
    
        
    
    img = cv2.equalizeHist(img)
        
    img = img / 255 # normalize 
    img = img.reshape(1,32,32,1)
    df = pd.read_csv('german-traffic-signs/signnames.csv')
        
    model = tf.keras.models.load_model('traffic_model.h5')      
    prediction = model.predict_classes(img)
    
    prediction = df.iloc[prediction, 1]
         
    print("Predicted sign:", prediction)
    
    