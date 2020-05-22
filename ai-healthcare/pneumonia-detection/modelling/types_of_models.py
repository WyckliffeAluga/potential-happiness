# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:19:57 2020

@author: wyckliffe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

import skimage
from skimage import io
import glob

import sklearn
from scipy.ndimage import gaussian_filter


class Model :

    def __init__(self) :

        # load two mamamo images
        self.dense = io.imread('dense/mdb003.pgm')
        self.fatty = io.imread('fatty/mdb005.pgm')

        #plt.imshow(dense)
        #plt.imshow(fatty)

        self.fatty_imgs = glob.glob("fatty/*")
        self.dense_imgs = glob.glob("dense/*")

    def intensity(self, threshold=50) :

        plt.hist(self.dense.ravel(), bins=256)
        plt.title('Dense')
        plt.show()

        plt.hist(self.fatty.ravel(), bins=256)
        plt.title('Fatty')

        plt.show()

        dense_bin = (self.dense > threshold) * 255
        fatty_bin = (self.fatty > threshold) * 255

        plt.imshow(dense_bin)

        plt.imshow(fatty_bin)

    def classify(self , threshold=50) :

        fatty_intensities = []

        for i in self.fatty_imgs :
            img = plt.imread(i)
            img_mask = (img > threshold)
            fatty_intensities.extend(img[img_mask].tolist())

        plt.hist(fatty_intensities, bins=256)


        dense_intensities = []

        for i in self.dense_imgs :

            img = plt.imread(i)
            img_mask = (img > threshold)
            dense_intensities.extend(img[img_mask].tolist())

        plt.hist(dense_intensities, bins=256)

        print(scipy.stats.mode(fatty_intensities)[0][0])
        print(scipy.stats.mode(dense_intensities)[0][0])


    def predict(self, threshold=50) :

        fatty_intensities = []

        for i in self.fatty_imgs :
            img = plt.imread(i)
            img_mask = (img > threshold)
            fatty_intensities.extend(img[img_mask].tolist())

        dense_intensities = []

        for i in self.dense_imgs :

            img = plt.imread(i)
            img_mask = (img > threshold)
            dense_intensities.extend(img[img_mask].tolist())

        for i in self.fatty_imgs:

            img = plt.imread(i)
            img_mask = (img > threshold)

            fatty_delta = scipy.stats.mode(img[img_mask])[0][0] - scipy.stats.mode(fatty_intensities)[0][0]
            dense_delta = scipy.stats.mode(img[img_mask])[0][0] - scipy.stats.mode(dense_intensities)[0][0]

            if (np.abs(fatty_delta) < np.abs(dense_delta)) :
                print("Fatty")
            else:
                print('Dense')

        for i in self.dense_imgs :

            img = plt.imread(i)
            img_mask  = (img > threshold)

            fatty_delta = scipy.stats.mode(img[img_mask])[0][0] - scipy.stats.mode(fatty_intensities)[0][0]
            dense_delta = scipy.stats.mode(img[img_mask])[0][0] - scipy.stats.mode(dense_intensities)[0][0]

            if (np.abs(fatty_delta) < np.abs(dense_delta)):
                print("Fatty")
            else:
                print('Dense')



if __name__ == '__main__' :
    m = Model()
    m.predict()

