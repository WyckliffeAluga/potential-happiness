# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:06:03 2020

@author: wyckliffe
"""


"""
Extract imaging data from each DICOM, visualize it, normalize it, and then visualize only the section of the image
that contains the suspicious mass.
"""

import pandas as pd
import numpy as np
import pydicom
import skimage
import matplotlib.pyplot as plt
import glob



class Dicom:

    def __init__(self) :

        self.bbox = pd.read_csv('dicom/bounding_boxes.csv')
        self.dcm = pydicom.dcmread('dicom/dicom_00029579_005.dcm')
        self.img = []
        self.dicoms  = glob.glob('dicom/*.dcm')

    def show(self, img):

        plt.imshow(img.pixel_array, cmap='gray')

    def hist(self):

        plt.figure(figsize=(5,5))
        plt.hist(self.dcm.pixel_array.ravel(), bins=256)

    def standardize(self) :
        mean_intensity = np.mean(self.dcm.pixel_array)
        std_intensity  = np.std(self.dcm.pixel_array)

        self.img = self.dcm.pixel_array.copy()
        self.img = (self.img - mean_intensity) / std_intensity

        plt.figure(figsize=(5,5))
        plt.hist(self.img.ravel(), bins=256)

        # plot the pixels of the mass
        plt.figure(figsize=(5,5))
        plt.hist(self.img[535:(535+66),240:(240+73)].ravel(), bins = 256,color='red')
        plt.show()



    def mass(self) :

        plt.imshow(self.dcm.pixel_array[535:(535+66),240:(240+73)],cmap='gray')

    def create_attributes(self) :

        data = []

        for i in self.dicoms :
            dcm = pydicom.dcmread(i)
            fields = [dcm.PatientID, int(dcm.PatientAge), dcm.PatientSex, dcm.Modality, dcm.StudyDescription,
            dcm.Rows, dcm.Columns]

            data.append(fields)

        return data


if __name__ == '__main__' :

    d = Dicom()
    data = pd.DataFrame(d.create_attributes(),
                        columns = ['PatientID','PatientAge','PatientSex','Modality','Findings','Rows','Columns'])


