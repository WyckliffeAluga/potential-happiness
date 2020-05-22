# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:10:29 2020

@author: wyckliffe
"""


import pandas as pd
from sklearn.metrics import confusion_matrix

# read the labels
labels = pd.read_csv('labels.csv')

# assess the accuracy of the radiolofist
radiologist_accuracy = sum(labels.perfect_labeler == labels.radiologist) / len(labels)

(confusion_matrix(labels.perfect_labeler.values, labels.radiologist.values, labels=['cancer', 'benign']))

# change the dataframe to 0's and 1's

labels =  labels.replace('cancer', 1).replace('benign', 0)

threshold = (labels.algorithm > 0.6)

print(confusion_matrix(labels.perfect_labeler.values, threshold, labels=[1, 0]))
