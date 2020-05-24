# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:54:20 2020

@author: wyckliffe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential , load_model
from keras.optimizers import  Adam

class Fashion :

    def __init__(self) :

        train = pd.read_csv('fashion-mnist_train.csv', sep=',')
        valid = pd.read_csv('fashion-mnist_test.csv' , sep=',')

        assert(train.shape[1] == valid.shape[1])

        self.train = np.array(train, dtype='float32')
        self.valid  = np.array(valid, dtype='float32')
        self.x_test = self.valid[:, 1:] / 255
        self.y_test = self.valid[:, 0]
        self.x_test = self.x_test.reshape(self.x_test.shape[0], *(28, 28 ,1))


    def show(self) :

        row = np.random.randint(0, len(self.train) + 1)
        plt.imshow(self.train[row, 1:].reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.title(self.label_encoder(self.train[row, 0]))

    def label_encoder(self, label) :
        switcher = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot"

        }
        return switcher.get(label, " Invalid label")
    def show_grid(self):

        fig, ax = plt.subplots(5, 5, figsize=(10,10))
        ax = ax.ravel()

        for i in np.arange(0, 25) :

            row = np.random.randint(0, len(self.train)+1)
            ax[i].imshow(self.train[row, 1:].reshape(28,28), cmap=plt.get_cmap('gray'))
            ax[i].set_title(self.label_encoder(self.train[row, 0]))
            ax[i].axis('off')

        plt.subplots_adjust(hspace=0.4)

    def model(self) :
        x_train = self.train[:, 1:] / 255
        y_train = self.train[:, 0]

        self.x_test = self.valid[:, 1:] / 255
        self.y_test = self.valid[:, 0]

        x_train , x_validate, y_train , y_validate = train_test_split(x_train, y_train,
                                                                      test_size=0.2, random_state=42)

        x_train = x_train.reshape(x_train.shape[0], *(28, 28, 1))
        self.x_test = self.x_test.reshape(self.x_test.shape[0], *(28, 28 ,1))
        x_validate = x_validate.reshape(x_validate.shape[0], *(28, 28, 1))

        #model = Sequential()
       # model.add(Conv2D(32, 3, 3, input_shape=(28, 28, 1), activation='relu'))
       # model.add(MaxPooling2D(pool_size=(2,2)))
       # model.add(Flatten())

       # model.add(Dense(output_dim=32, activation='relu'))
       # model.add(Dense(output_dim=10, activation='sigmoid'))

       # model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

       # model.fit(x_train, y_train,
          #       validation_data=(x_validate, y_validate),
          #      verbose=1, epochs=50, batch_size=512)

      #  model.save('model.h5')
    def random_test(self) :

        model = load_model('model.h5')
        predictions = model.predict_classes(f.x_test)


        image = np.random.randint(0, len(f.x_test) + 1)

        plt.imshow(self.x_test[image].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.title("Prediction: {} \n True Class: {}".format(self.label_encoder(predictions[image]), self.label_encoder(self.y_test[image])))




if __name__ == '__main__' :

    f = Fashion()

    model = load_model('model.h5')
   # evaluation = model.evaluate(f.x_test, f.y_test)
   # print('Test accuracy : {:.3f}'.format(evaluation[1]))



