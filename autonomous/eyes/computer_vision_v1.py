# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:09:23 2020

@author: wyckliffe
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class Annette:

    def __init__(self):
        self.img = mpimg.imread('image.jpg')

    def show(self, img):

        plt.imshow(img , cmap=plt.get_cmap('gray'))

    def gray(self, img) :

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



def main():

    a = Annette()
    a.show(a.gray(a.img))

if __name__ == '__main__' :
    main()

