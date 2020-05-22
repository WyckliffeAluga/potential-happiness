# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:02:05 2020

@author: wyckliffe
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
import scipy
from sklearn.model_selection import train_test_split

from random import sample
from itertools import chain



data = pd.read_csv('split/findings_data_5000.csv')

labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if len(x)>0]

ax = data[labels].sum().plot(kind='bar')
ax.set(ylabel = 'Number of Images with Label')
plt.show()


plt.figure(figsize=(16,6))
data[data.Pneumothorax==1]['Finding Labels'].value_counts()[0:30].plot(kind='bar')
plt.title('Pneumothorax (finding labels)')
plt.show()


##Since there are many combinations of potential findings, I'm going to look at the 30 most common co-occurrences:
plt.figure(figsize=(6,6))
data[data.Pneumothorax ==1]['Patient Gender'].value_counts().plot(kind='bar')
plt.title('Pneumothoraz (Patient Gender)')
plt.show()

plt.figure(figsize=(10,6))
plt.hist(data[data.Pneumothorax==1]['Patient Age'])
plt.title('Pneumothoraz (Patient Age)')
plt.show()


train_df , valid_df = train_test_split(data,
                                           test_size=0.2,
                                           stratify=data['Pneumothorax'])

print(train_df['Pneumothorax'].sum() / len(train_df))
print(valid_df['Pneumothorax'].sum() / len(valid_df))

p_inds = train_df[train_df.Pneumothorax==1].index.tolist()
np_inds = train_df[train_df.Pneumothorax==0].index.tolist()

np_sample = sample(np_inds,len(p_inds))
train_df = train_df.loc[p_inds + np_sample]

print(train_df['Pneumothorax'].sum()/len(train_df))

p_inds = valid_df[valid_df.Pneumothorax==1].index.tolist()
np_inds = valid_df[valid_df.Pneumothorax==0].index.tolist()

# The following code pulls a random sample of non-pneumonia data that's 4 times as big as the pneumonia sample.
np_sample = sample(np_inds,4*len(p_inds))
valid_df = valid_df.loc[p_inds + np_sample]

print(valid_df['Pneumothorax'].sum()/len(valid_df))