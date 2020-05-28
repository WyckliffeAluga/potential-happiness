# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:25:03 2020

@author: wyckliffe
"""

from keras.models import load_model
from keras.preprocessing import image

import numpy as np


model = load_model('model.h5')
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
#training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

