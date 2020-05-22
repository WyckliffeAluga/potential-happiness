# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:03:15 2020

@author: wyckliffe
"""


import numpy as np
import pandas as pd
import os

from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

class Tune:

    def __init__(self) :

        train_df = pd.read_csv('train.csv')
        valid_df = pd.read_csv('test.csv')

        image_size = (224, 224)

        train_idg = ImageDataGenerator(rescale=1. / 255.0,
                              horizontal_flip = True,
                              vertical_flip = False,
                              height_shift_range= 0.1,
                              width_shift_range=0.1,
                              rotation_range=20,
                              shear_range = 0.1,
                              zoom_range=0.1)

        self.train_gen = train_idg.flow_from_dataframe(dataframe=train_df,
                                         directory=None,
                                         x_col = 'img_path',
                                         y_col = 'class',
                                         class_mode = 'binary',
                                         target_size = image_size,
                                         batch_size = 9
                                         )

        val_idg = ImageDataGenerator(rescale=1. / 255.0
                                 )

        val_gen = val_idg.flow_from_dataframe(dataframe=valid_df,
                                         directory=None,
                                         x_col = 'img_path',
                                         y_col = 'class',
                                         class_mode = 'binary',
                                         target_size = image_size,
                                         batch_size = 6)
        self.testX, self.testY = val_gen.next()


    def model(self):
        model = VGG16(include_top=True, weights='imagenet')
        print(model.summary())

        transfer_layer = model.get_layer('block5_pool')
        vgg_model = Model(inputs=model.input, output=transfer_layer.output)

        # choose layer to fine-tune
        # freeze all but last layer

        for layer in vgg_model.layers[0:17] :
            layer.trainable = False

        for layer in vgg_model.layers :
            print(layer.name, layer.trainable)


        new_model = Sequential()

        # add layer part of the VGG16 model
        new_model.add(vgg_model)

        # flatten the outputs
        new_model.add(Flatten())

        # add a dropout layer
        new_model.add(Dropout(0.5))

        # add a dense layer
        new_model.add(Dense(1024, activation='relu'))

        # add a dropout layer
        new_model.add(Dropout(0.5))

        # add another layer
        new_model.add(Dense(512, activation='relu'))

        # add a droupout layer
        new_model.add(Dropout(0.5))

        new_model.add(Dense(256, activation='relu'))

        # add output layer
        new_model.add(Dense(1, activation='sigmoid'))

        # set optimizer , loss function and learning rate
        optimizer = Adam(lr=10e-4)
        loss = 'binary_crossentropy'
        metrics = ['binary_accuracy']

        new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # run a single epoch
        hist = new_model.fit(self.train_gen, validation_data=(self.testX, self.testY) , epochs=10)

        return hist

def plot_history(history):
    N = len(history.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["binary_accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_binary_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")


if __name__ == '__main__' :
    t = Tune()
    plot_history(t.model())