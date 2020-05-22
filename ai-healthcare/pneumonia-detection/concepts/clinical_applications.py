# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:55:19 2020

@author: wyckliffe
"""


import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import numpy as np
import pandas as pd

class Image:

    def __init__(self) :

        self.chest_xray_1 = imread('img/chest_xray_1.png')
        self.chest_xray_2 = imread('img/chest_xray_2.png')
        self.chest_xray_3 = imread('img/chest_xray_3.png')

        self.mammo_left = imread('img/mammo_left.pgm')
        self.mammo_right= imread('img/mammo_right.pgm')

    def show(self, img) :

        imshow(img)

    def divide_chest_xray(self) :

        #  roughly selecting coordinated for lung, heart and bone
        lung_x = [690, 760]
        lung_y = [500, 560]

        heart_x = [550, 615]
        heart_y = [750, 800]

        bone_x = [430, 480]
        bone_y = [770, 810]

        plt.figure(figsize=(5, 5))
        plt.hist(self.chest_xray_3[bone_y[0]:bone_y[1],bone_x[0]:bone_x[1]].ravel(), bins = 256,color='green')
        plt.hist(self.chest_xray_3[lung_y[0]:lung_y[1],lung_x[0]:lung_x[1]].ravel(),bins =256,color='blue')
        plt.hist(self.chest_xray_3[heart_y[0]:heart_y[1],heart_x[0]:heart_x[1]].ravel(),bins=256,color='red')
        plt.legend(['Bone','Lungs','Heart'])
        plt.show()

    def divide_mammogram(self) :


        tumor_x = [480,580]
        tumor_y = [310,380]

        normal_x = [440,500]
        normal_y = [300,360]

        plt.figure(figsize=(5,5))
        plt.hist(self.mammo_left[tumor_y[0]:tumor_y[1],tumor_x[0]:tumor_x[1]].ravel(), bins = 256,color='red')
        plt.hist(self.mammo_right[normal_y[0]:normal_y[1],normal_x[0]:normal_x[1]].ravel(),bins =256,color='blue')
        plt.legend(['Tumor','Normal Tissue'])
        plt.xlabel('intensity')
        plt.ylabel('number of pixels')
        plt.show()


if __name__ == '__main__' :

    i = Image()
    i.divide_chest_xray()
    i.divide_mammogram()