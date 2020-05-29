# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:35:16 2020

@author: wyckliffe
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential , Dropout
from keras.layers import Dense

data = pd.read_csv('bank_data.csv')

X = data.iloc[:, 3:13]
y = data.iloc[:, 13]

X = pd.get_dummies(X)

scaler = StandardScaler()

x_train , x_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=123)

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = Sequential()

model.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=13))
model.add(Dense(output_dim = 6, init='uniform', activation='relu'))
model.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, shuffle=1, batch_size=10)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

new_customer = np.array([[600, 0., 0, 0, 1, 40, 3, 60000, 2, 1, 0, 0, 50000]])
new_customer = scaler.fit_transform(new_customer)
new_prediction = model.predict(new_customer)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def model():
    model = Sequential()

    model.add(Dense(output_dim = 6, init='uniform', activation='relu', input_dim=13))
    model.add(Dense(output_dim = 6, init='uniform', activation='relu'))
    model.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

classifier = KerasClassifier(build_fn = model, batch_size=10, epochs=50)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=-1)

mean = accuracies.mean()
variance = accuracies.std()

