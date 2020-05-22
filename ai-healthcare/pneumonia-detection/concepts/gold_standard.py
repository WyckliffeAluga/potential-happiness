# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:28:22 2020

@author: wyckliffe
"""


import pandas as pd

labels = pd.read_csv('population/labels.csv')

labels2 = labels.replace('benign',1).replace('cancer',0)

gt1 = labels2['biopsy']

gt2 = labels2[['rad1','rad2','rad3']].sum(axis=1)
gt2 = (gt2 > 1).replace(True,1).replace(False,0)

weighted_labels = labels2.copy()
weighted_labels['rad2'] = weighted_labels['rad2'] * 0.67
weighted_labels['rad1'] = weighted_labels['rad1'] * 0.33

gt3 = weighted_labels[['rad1','rad2','rad3']].sum(axis=1)
gt3 = (gt3 > 1).replace(True,1).replace(False,0)

biopsy_to_votes = gt1 == gt2
biopsy_to_votes[biopsy_to_votes==False]

biopsy_to_weights = gt1 == gt3
biopsy_to_weights[biopsy_to_weights==False]
