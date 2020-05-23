# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:41:55 2020

@author: wyckliffe
"""




import pandas as pd
import numpy as np
import sklearn.metrics

data = pd.read_csv('labels_and_performance.csv')

tn, fp, fn, tp = sklearn.metrics.confusion_matrix(data.Pneumonia.values,
                                                  data.algorithm_output.values,labels=[1,0]).ravel()
sens = tp/(tp+fn)

spec = tn/(tn+fp)

# look at the algorithms performance in presence of other diseases
for i in ['Atelectasis','Effusion','Pneumothorax','Infiltration','Cardiomegaly','Mass','Nodule']:

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(data[data[i]==1].Pneumonia.values,
                                                  data[data[i]==1].algorithm_output.values,labels=[1,0]).ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)

    print(i)
    print('Sensitivity: '+ str(sens))
    print('Specificity: ' +str(spec))
    print()
