# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 12:24:11 2020

@author: wyckliffe
"""


import numpy as np
from PIL import Image



def save(image, filename) :
    image = np.clip(image, 0.0 , 255.0)
    image = image.astype(np.uint8)
    with open(filename, 'wb') as file :
        Image.fromarray(image).save(file, 'jpeg')


def normalize(image) :
    min_ = image.min()
    max_ = image.max()
    return (image - min_) / (max_ - min_)

def resize(image, size=None, factor=None) :
    if factor is not None :
        size = np.array(image.shape[0:2]) * factor
        size = size.astype(int)

    size = tuple(reversed(size))   # compensate for the np/PIL diff
    image = np.clip(image, 0.0, 255.0)
    image = Image.fromarray(image.astype(np.uint8))
    return  np.float32(image.resize(size, Image.LANCZOS))

