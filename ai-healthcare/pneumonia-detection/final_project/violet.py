# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:53:42 2020

@author: wyckliffe
"""

# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
from itertools import chain
from random import sample
import scipy
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from skimage import io
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, MaxPooling2D, Reshape
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, plot_precision_recall_curve, f1_score, confusion_matrix, accuracy_score



class Violet :

    def __init__(self) :

        ## Load the NIH data to df
        df = pd.read_csv('/data/Data_Entry_2017.csv')
        image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join('/data','images*', '*', '*.png'))}
        print('Scans found:', len(image_paths), ', Total Headers', df.shape[0])
        df['path'] = df['Image Index'].map(image_paths.get)

        self.df = df

    def labelling(self) :

        labels = np.unique(list(chain(*self.df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        print(labels)
        for label in labels:
            if len(label)>1: # leave out empty labels
                self.df[label] = self.df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0)
        print(self.df.sample(5))

        self.df['pneumonia_class'] = self.df['pneumonia']

    def create_splits(self) :

        train_df , valid_df = train_test_split(self.df,
                                               test_size= 0.2,
                                               stratify=self.df['pneumonia_class'])

        # make equal proportions of pneumonial in both sets

        # training data
        positive_index = train_df[train_df['pneumonia_class'] == 1].index.tolist()
        non_positive_index = train_df[train_df['pneumonia_class'] == 0].index.tolist()

        np_sample = sample(non_positive_index, len(positive_index))
        train_df = train_df.loc[positive_index + np_sample]

        # validation data
        positive_index = valid_df[valid_df['pneumonia_class'] == 1].index.tolist()
        non_positive_index= valid_df[valid_df['pneumonia_class'] == 0].index.tolist()

        np_sample = sample(non_positive_index, 4*len(positive_index))
        valid_df = valid_df.loc[positive_index + np_sample]

        return train_df, valid_df

    def image_augmentation(self):

        augment = ImageDataGenerator (rescale= 1./255.0,
                                      horizontal_flip = True,
                                      vertical_flip = False,
                                      height_shift_range=0.1,
                                      width_shift_range=0.1,
                                      rotation_range=20,
                                      shear_range=0.1,
                                      zoom_range=0.1)

        return augment

    def train_generator(self, train_data) :

        augment = self.image_augmentation()
        train_gen = augment.flow_from_dataframe(dataframe=train_data,
                                                directory=None,
                                                x_col = 'path',
                                                y_col = 'pneumonia_class',
                                                class_mode='raw',
                                                target_size=(224, 224),
                                                batch_size=64)

        return train_gen

    def valid_generator(self, valid_data) :

        augment = ImageDataGenerator(rescale = 1. / 255.)

        val_gen = augment.flow_from_dataframe(dataframe = valid_data,
                                              directory=None,
                                              x_col= 'path',
                                              y_col= 'pneumonia_class',
                                              target_size = (224, 224),
                                              batch_size=64)

        return val_gen

    def check_examples(self, train_gen):

        x, y = next(train_gen)
        fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
            for (c_x, c_y, c_ax) in zip(x, y, m_axs.flatten()):
                c_ax.imshow(c_x[:,:,0], cmap = 'bone')
                if c_y == 1:
                    c_ax.set_title('Pneumonia')
                else:
                    c_ax.set_title('No Pneumonia')
        c_ax.axis('off')

    def load_model(self) :

        model = VGG16(include_top=True, weights='imagenet')
        transfer_layer = model.get_layer('block5_pool')
        vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)

        for layer in vgg_model.layers[0:17]:
            layer.trainable = False

       return vgg_model


    def build_model(self) :

        model = Sequential()
        vgg_model = load_pretrained_model()
        model.add(vgg_model)
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

    return model


   def add_checkpoints(self) :

       weight_path="{}model.best.hdf5".format('xray_class')

       checkpoint = ModelCheckpoint(weight_path,
                             monitor= 'val_binary_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode= "auto",
                             save_weights_only = True)

        early = EarlyStopping(monitor= "val_binary_accuracy",
                      mode= "auto",
                      patience=5)

        def scheduler(epoch, lr):
            if epoch < 1:
                return lr
            else:
                return lr * np.exp(-0.1)

        lr_scheduler = LearningRateScheduler(scheduler)
        callbacks_list = [checkpoint, early, lr_scheduler]

        return callbacks_list

def train(model, train, x_val, y_val, callbacks) :

    optimizer = RMSprop(learning_rate=1e-4)
    loss = 'binary_crossentropy'
    metrics = ['binary_accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit_generator(train_gen,
                          validation_data = (x_val, y_val),
                           epochs = 10,
                           callbacks = callbacks_list)

    return history


def plot_auc(t_y, p_y):

    ## Hint: can use scikit-learn's built in functions here like roc_curve

    fpr, tpr, threshold = roc_curve(valY, pred_Y)
    roc_auc = auc(fpr, tpr)

    plt.title('Plot AUC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return

    return

def plot_prec_rec(val_Y, pred_Y):
    prec, rec, threshold = precision_recall_curve(val_Y, pred_Y)
    plt.title('Plot Precision Recall')
    plt.plot(prec, rec, 'b', label = 'score = %0.2f' % average_precision_score(val_Y,pred_Y))
    plt.legend(loc = 'upper right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

def plot_history(history):

    # Todo
    n = len(history.history["loss"])
    plt.figure()
    plt.plot(np.arange(n), history.history["loss"], label="train_loss")
    plt.plot(np.arange(n), history.history["val_loss"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    return

def optimize_accuracy(t_y, p_y):
    best_threshold = None
    best_accuracy = 0.
    for t in np.arange(0.5,1,0.1):
        pred = (p_y.reshape(-1)>t)*1.
        accuracy = np.mean(pred==t_y)
        if accuracy > best_accuracy:
            best_threshold = t
            best_accuracy = accuracy
    return best_threshold, best_accuracy


def main():

    # initialize the class
    v = Violet()

    # label the data so that we have binary indicators of certain diseases
    v.labelling()

    # split the data
    train_data, valid_data = v.create_splits()

    # perform image augmentation
    train_gen = v.train_generator(train_data)
    val_gen = v.valid_generator(valid_data)

    # pull as single large batch of random validation data for testing
    valX, valY = val_gen.next()

    # look at some examples and see how aumentation has worked on the training data
    v.check_examples(train_gen)

    # build the model and read the summary

    model = v.build_model()
    print(model.summary)

    # add checkpoints to save the bst version and also initialize early stop
    callbacks_list = v.add_checkpoints()

    # train the model and save
    history = train(model, train_gen, valX, valY, callbacks_list)

    weight_path = 'xray_classmodel.best.hdf5'
    model.load_weights(weight_path)
    pred_Y = model.predict(valX, batch_size = 32, verbose = True)


    plot_auc(valY, pred_Y)
    plot_prec_rec(valY, pred_Y)
    plot_history(history)

    best_threshold, best_accuracy = optimize_accuracy(valY, pred_Y)
    print("Threshold of %.2f gives best accuracy at %.4f"%(best_threshold, best_accuracy))
    pred_Y_class = pred_Y > best_threshold
    f1_score(valY, pred_Y_class)


    fig, m_axs = plt.subplots(8, 8, figsize = (16, 16))
    i = 0
    for (c_x, c_y, c_ax) in zip(valX[0:64], valY[0:64], m_axs.flatten()):
        c_ax.imshow(c_x[:,:,0], cmap = 'bone')
        if c_y == 1:
            if pred_Y[i] > best_threshold:
                c_ax.set_title('1, 1')
            else:
                c_ax.set_title('1, 0')
        else:
            if pred_Y[i] > best_threshold:
                c_ax.set_title('0, 1')
            else:
                c_ax.set_title('0, 0')
        c_ax.axis('off')
        i=i+1


    model_json = model.to_json()
    with open("my_model.json", "w") as json_file:
        json_file.write(model_json)

if __name__ = '__main__' :
    main()