# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:59:09 2020

@author: wyckliffe 
"""


import numpy as np 
import pickle
import pandas as pd
import requests
from PIL import Image 
import cv2
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint



class CNN():
    
    def __init__(self): 
        
        self.num_of_classes = 43
        self.num_of_samples = []
        self.x_train = []
        self.y_train = []
        self.x_val   = []
        self.y_val   = []
        self.x_test  = []
        self.y_test  = []
        self.data    = []
        
        self.get_data()
        
        self.p_x_train = self.imgProcessingPipeline(self.x_train)
        self.p_x_train = self.p_x_train.reshape(self.p_x_train.shape[0] , 32 ,32, 1) # add depth
        self.p_y_train = self.encode(self.y_train)
        self.p_x_val   = self.imgProcessingPipeline(self.x_val)
        self.p_x_val   = self.p_x_val.reshape(self.p_x_val.shape[0] , 32 ,32, 1) # add depth
        self.p_y_val   = self.encode(self.y_val)
        self.p_x_test  = self.imgProcessingPipeline(self.x_test)
        self.p_x_test  = self.p_x_test.reshape(self.p_x_test.shape[0] , 32 ,32, 1) # add depth
        self.p_y_test  = self.encode(self.y_test)
        
        self.Net()
        self.model = tf.keras.models.load_model('traffic_model.h5')
    
    
    def get_data(self): 
        
        print("Fetching data.......")
        
        with open("german-traffic-signs/train.p", "rb") as f :
          train_data =  pickle.load(f)
 
         
        with open("german-traffic-signs/valid.p", "rb") as f :
          val_data =  pickle.load(f)
          

        with open("german-traffic-signs/test.p", "rb") as f :
          test_data =  pickle.load(f)
          
        self.data = pd.read_csv("german-traffic-signs/signnames.csv")
       
        self.x_train, self.y_train = train_data['features'], train_data['labels']
        self.x_val, self.y_val = val_data['features'], val_data['labels']
        self.x_test, self.y_test = test_data['features'], test_data['labels']
        
        print("Completed!")
        print("Checking assertions.........!")
        
        assert(self.x_train.shape[0] == self.y_train.shape[0]), "The number of images is not equal to the number of labels"
        assert(self.x_val.shape[0] == self.y_val.shape[0]), "The number of images is not equal to the number of labels"
        assert(self.x_test.shape[0] == self.y_test.shape[0]), "The number of images is not equal to the number of labels"
    
        assert(self.x_train.shape[1:] == (32, 32, 3)), "The dimensions of images are not 32 x 32 x 3"
        assert(self.x_val.shape[1:] == (32, 32, 3)), "The dimensions of images are not 32 x 32 x 3"
        assert(self.x_test.shape[1:] == (32, 32, 3)), "The dimensions of images are not 32 x 32 x 3"
        
        print("Everything checks out thank you !")
        
    def show(self):  
        
        cols = 5 
        
        fig, ax = plt.subplots(nrows=self.num_of_classes ,ncols=cols, figsize=(5, 50))
        fig.tight_layout()
        
        for i in range(cols): 
            for j , row in self.data.iterrows(): 
                x_selected = self.x_train[self.y_train == j]
                ax[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), : , : ], cmap= plt.get_cmap("gray"))
                ax[j][i].axis("off")
                
                if i == 2: 
                    ax[j][i].set_title(str(j) + "_" + row['SignName'])
                    self.num_of_samples.append(len(x_selected))
                    
    def showsamples(self): 
        
        # you have to run show()first
        plt.figure(figsize=(12,4))
        plt.bar(range(0, self.num_of_classes), self.num_of_samples)
        plt.title("Ditribution of the training dataset")
        plt.xlabel("Class number")
        plt.ylabel("Number of images")
        plt.show()
        
    def grayScale(self, img): 
        """
        Color on the traffic signs is not important
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def histogramNormalization(self, img): 
        
        """
        Standardize the lighgting of all images enhancing the contrast
    
        """
        return cv2.equalizeHist(self.grayScale(img))
    
    def preprocessing(self, img): 
        
        img = self.histogramNormalization(img)
        img = img/255
        
        return img
    
    def imgProcessingPipeline(self, data): 
        
        return np.array(list(map(self.preprocessing, data)))
        
    def encode(self, data):
        
        return to_categorical(data, self.num_of_classes)
    
    def augumentImages(self): 
        
        dataGen = ImageDataGenerator(width_shift_range = 0.1, 
                           height_shift_range = 0.1, 
                           zoom_range = 0.2, 
                           shear_range = 0.1 , 
                           rotation_range = 10)
        
        dataGen.fit(self.p_x_train)
        batches = dataGen.flow(self.p_x_train, self.p_y_train, batch_size=20)
        x_batch, y_batch = next(batches)
        
        fig, ax = plt.subplots(1, 15, figsize=(20, 5))
        fig.tight_layout()
        
        for i in range(15): 
            ax[i].imshow(x_batch[i].reshape(32,32))
            ax[i].axis("off")
            
        
    def Net(self): 
        
        
        model = tf.keras.Sequential()
        model.add(Conv2D(60 , (5,5), input_shape=(32,32,1) , activation='relu'))
        model.add(Conv2D(60 , (5,5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(30, (3,3) , activation='relu'))
        model.add(Conv2D(30, (3,3) , activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2))) 
        
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        
        model.add(Dropout(0.5))
        
        model.add(Dense(self.num_of_classes , activation='softmax'))
        adam = tf.keras.optimizers.Adam
        model.compile(adam(lr=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
        
        print(model.summary())
        
        dataGen = ImageDataGenerator(width_shift_range = 0.1, 
                           height_shift_range = 0.1, 
                           zoom_range = 0.2, 
                           shear_range = 0.1 , 
                           rotation_range = 10)
        
        dataGen.fit(self.p_x_train)
        
        
        model.fit_generator(dataGen.flow(self.p_x_train, self.p_y_train, batch_size=50),
                            steps_per_epoch=2000, epochs=10, validation_data=(self.p_x_val , self.p_y_val), 
                            verbose=1, shuffle=1)
        
        model.save('traffic_model.h5')
    
        print("Model saved")
        


    def checkModel(self): 
        
        model = self.model
        score = model.evaluate(self.p_x_test, self.p_y_test , verbose=1)
        
        return {"Test score ": score[0] , "Test accuraccy": score[1]}        
    
traffic = CNN()

