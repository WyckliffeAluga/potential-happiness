# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:02:02 2020

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
from tensorflow.keras.layers import Dense
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
        
        self.x_train_flattened = self.flatten(self.x_train_normalized)
        self.x_test_flattened  = self.flatten(self.x_test_normalized)
        
        assert (self.x_train_flattened.shape == (60000, 784)), "Error in flattenning the image"
        assert (self.x_test_flattened.shape == (10000, 784)), "Error in flattenning the image"
        
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
    
    def flatten(self, data): 
        num_pixels = 784
        
        return  data.reshape(data.shape[0], num_pixels)
    
    def create_model(self): 
        
        model = tf.keras.Sequential()
        model.add(Dense(10, input_dim=784, activation="relu"))
        model.add(Dense(30, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.num_of_classes, activation="softmax"))
        adam = tf.keras.optimizers.Adam
        model.compile(adam(lr=0.01), loss="categorical_crossentropy", metrics=['accuracy'])
        
        return model
    
    def train(self): 
        
        model = self.create_model()
        hist = model.fit(self.x_train_flattened, self.y_train_encoded, validation_split=0.1, epochs=10, batch_size=200, verbose=1, shuffle=1)
        
        return hist
    
    def progress(self): 
        hist = self.train()
        
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history["val_accuracy"])
        
        plt.legend(['loss', 'validation loss', 'accuracy', 'validation accuracy'])
        plt.xlabel('epochs')
        plt.show()
        
    def checkModel(self): 
        model = self.create_model()
        model.fit(self.x_train_flattened, self.y_train_encoded, validation_split=0.1, epochs=10, batch_size=200, verbose=1, shuffle=1)
        score = model.evaluate(self.x_test_flattened, self.y_test_encoded , verbose=1)
        
        return {"Test score ": score[0] , "Test accuraccy": score[1]}
    
    def testImage(self): 
        
        url = "https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png"
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
        img = img.reshape(1,784)
        
        model = self.create_model()
        model.fit(self.x_train_flattened, self.y_train_encoded, validation_split=0.1, epochs=10, batch_size=200, verbose=1, shuffle=1)
        
        prediction = model.predict_classes(img)
        
        print("Predicted digit :", str(prediction))
        
    
         

d = DNN()