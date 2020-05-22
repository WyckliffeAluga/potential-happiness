# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:25:05 2020

@author: wyckliffe
"""


# exploring population metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample

from itertools import chain
import scipy

class Pop:

    def __init__(self) :

        self.d = pd.read_csv('population/findings_data.csv')
        self.data = self.d.copy()

        self.labels = np.unique(list(chain(*self.data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        self.labels = [x for x in self.labels if len(x)>0]
        #print('All Labels ({}): {}'.format(len(self.labels), self.labels))

        for c_label in self.labels:
            if len(c_label)>1: # leave out empty labels
                self.data[c_label] = self.data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
        self.data.sample(3)

    def show(self) :
        ax = self.data[self.labels].sum().plot(kind='bar')
        ax.set(ylabel='Number of images with label')

    def plot_common(self, n) :

        plt.figure(figsize=(16,6))
        self.data[self.data.Infiltration==1]['Finding Labels'].value_counts()[0:n].plot(kind='bar')

        plt.figure(figsize=(16,6))
        self.data[self.data.Effusion==1]['Finding Labels'].value_counts()[0:n].plot(kind='bar')

    def gender_distribution(self):

        plt.figure(figsize=(6,6))
        self.data['Patient Gender'].value_counts().plot(kind='bar')
        plt.title('All patients')

        plt.figure(figsize=(6,6))
        self.data[self.data.Infiltration ==1]['Patient Gender'].value_counts().plot(kind='bar')
        plt.title("Infiltration patients")

        plt.figure(figsize=(6,6))
        self.data[self.data.Effusion ==1]['Patient Gender'].value_counts().plot(kind='bar')
        plt.title("Effusion Patients")

    def mass_size_distribution(self) :

        plt.scatter(self.data['Patient Age'],self.data['Mass_Size'])
        plt.ylabel("Mass")
        plt.xlabel("Age")
        plt.show()

        mass_sizes = self.data['Mass_Size'].values
        mass_inds = np.where(~np.isnan(mass_sizes))
        ages = self.data.iloc[mass_inds]['Patient Age']
        mass_sizes=mass_sizes[mass_inds]
        print(scipy.stats.pearsonr(mass_sizes,ages))

        print(scipy.stats.ttest_ind(self.data[self.data['Patient Gender']== 'F']['Mass_Size'],
                                    self.data[self.data['Patient Gender']== 'M']['Mass_Size'],nan_policy='omit'))

if __name__ == '__main__' :
    p  = Pop()
    p.mass_size_distribution()