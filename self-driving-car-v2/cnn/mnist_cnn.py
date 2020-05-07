# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:27:56 2020

@author: wyckliffe
"""


import numpy as np 
import requests
from PIL import Image 
import cv2
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.utils.np_utils import to_categorical


class DNN(): 
    
    def __init__(self): 
        
        np.random.seed(0)
        (X_train, y_train) , (x_test, y_test) = mnist.load_data() 
        self.num_of_samples = []
        self.num_of_classes = 10 
        
        assert(X_train.shape[0] == y_train.shape[0]), "The number of images is equal to the numbe rof labels"
        assert(x_test.shape[0] == y_test.shape[0]), "The number of images is equal to the numbe rof labels"
        assert(X_train.shape[1:] == (28, 28)), "The dimensions of the images are not 28 x 28"
        assert(x_test.shape[1:] == (28, 28)), "The dimensions of the images are not 28 x 28"
        
        self.X_train = X_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.y_train_encoded = self.encodingClasses(y_train)
        self.y_test_encoded = self.encodingClasses(y_test)
        
        assert(self.y_train_encoded.shape == (60000, 10)), "Error in encoding the labels"
        
        self.x_train_normalized = self.normalizing(X_train)
        self.x_test_normalized  = self.normalizing(x_test)
        
        self.x_train_flattened = self.flatten(self.x_train_normalized, 60000)
        self.x_test_flattened  = self.flatten(self.x_test_normalized, 10000)
        
        assert (self.x_train_flattened.shape == (60000, 28, 28, 1)), "Error in flattenning the image"
        assert (self.x_test_flattened.shape == (10000, 28, 28, 1)), "Error in flattenning the image"
        
        self.leNet()
        self.model = tf.keras.models.load_model('model.h5')
        
    def show(self):  
        
        cols = 5 
        
        fig, ax = plt.subplots(nrows=self.num_of_classes ,ncols=cols, figsize=(5, 10))
        fig.tight_layout()
        
        for i in range(cols): 
            for j in range(self.num_of_classes): 
                x_selected = self.X_train[self.y_train == j]
                ax[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), : , : ], cmap= plt.get_cmap("gray"))
                ax[j][i].axis("off")
                
                if i == 2: 
                    ax[j][i].set_title(str(j))
                    self.num_of_samples.append(len(x_selected))
                    
    def showsamples(self): 
        
        # you have to run show()first
        plt.figure(figsize=(12,4))
        plt.bar(range(0, self.num_of_classes), self.num_of_samples)
        plt.title("Ditribution of the training dataset")
        plt.xlabel("Class number")
        plt.ylabel("Number of images")
        plt.show()
        
    def encodingClasses(self, data): 
        
        return to_categorical(data, 10)
            
    def normalizing(self, data): 
        
        return data/255
    
    def flatten(self, data, size): 
        
        return  data.reshape(size, 28 , 28 , 1)
    
    
    def leNet(self): 
        model = tf.keras.Sequential()
        model.add(Conv2D(30 , (5,5), input_shape=(28,28,1) , activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(15 , (3,3) , activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_of_classes , activation='softmax'))
        adam = tf.keras.optimizers.Adam
        model.compile(adam(lr=0.01), loss="categorical_crossentropy", metrics=['accuracy'])
        model.fit(self.x_train_flattened, self.y_train_encoded, validation_split=0.1, epochs=10, batch_size=400, verbose=1, shuffle=1)
        model.save('model.h5')
        print("Model saved")
        
    
    def progress(self): 
        hist = self.model
        
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history["val_accuracy"])
        
        plt.legend(['loss', 'validation loss', 'accuracy', 'validation accuracy'])
        plt.xlabel('epochs')
        plt.show()
        
    def checkModel(self): 
        
        model = self.model
        score = model.evaluate(self.x_test_flattened, self.y_test_encoded , verbose=1)
        
        return {"Test score ": score[0] , "Test accuraccy": score[1]}
    
    def testImage(self, url): 
        
        res = requests.get(url, stream='true')
        assert(res.status_code == 200), "Get request failed"
        
        img = Image.open(res.raw)
        img = np.asarray(img)
        img = cv2.resize(img, (28,28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        assert(img.shape == (28,28)), "Test image size not 28 x 28"
        
        img = cv2.bitwise_not(img)
        plt.imshow(img, cmap= plt.get_cmap("gray"))
        
        img = img / 255 # normalize 
        img = img.reshape(1,28,28,1)
        
        model = self.model        
        prediction = model.predict_classes(img)
        
        print("Predicted digit :", str(prediction))
        
d = DNN()
