# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:10:46 2020

@author: wyckliffe
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


class Cancer :

    def __init__(self) :
        # load data
        cancer_data = load_breast_cancer()
        self.cancer_df = pd.DataFrame(np.c_[cancer_data['data'], cancer_data['target']],
                                 columns=np.append(cancer_data['feature_names'], ['target']))

    def visualize_1(self) :
        sns.pairplot(self.cancer_df, hue = 'target', vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

    def visualize_2(self) :
        sns.countplot(self.cancer_df['target'])

    def visualize_3(self) :
        sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue='target', data = self.cancer_df)

    def visualize_4(self) :
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.cancer_df.corr() , annot = True)

    def model(self) :

        X = self.cancer_df.drop(['target'], axis=1)
        y = self.cancer_df['target']

        x_train , x_valid , y_train, self.y_valid = train_test_split(X, y,
                                                                 test_size = 0.3 )
        min_x_train = x_train.min()
        range_x_train = (x_train - min_x_train).max()

        x_train = (x_train - min_x_train) / range_x_train

        min_x_valid = x_valid.min()
        range_x_valid = (x_valid - min_x_valid).max()

        self.x_valid = (x_valid - min_x_valid) / range_x_valid

        param_grid = {'C' : [0.1, 1, 10, 100],
                  'gamma': [1 , 0.1, 0.01, 0.001],
                  'kernel': ['rbf']}

        grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 0)

        grid.fit(x_train, y_train)

        return grid


if __name__ == '__main__' :
    c = Cancer()

    model = c.model()
    prediction = model.predict(c.x_valid)
    cm = confusion_matrix(c.y_valid, prediction)
    sns.heatmap(cm, annot=True)

    print(classification_report(c.y_valid, prediction))










