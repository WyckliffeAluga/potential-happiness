# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:34:59 2020

@author: wyckliffe
"""

import os
import numpy as np 
import ntpath
import cv2 
import pandas as pd 
import random 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.image as mpimg 
from imgaug import augmenters as iaa

class Car() : 
    
    def __init__(self): 
        
        self.datadir = 'data'
        
        self.data =  self.get_data()
        
        # Balance data 
        self.balanceData()
        
        # load image paths and steering 
        self.image_paths, self.steering = self.load_img_steering()
        
        # split the images into training and testing 
        self.x_train , self.x_test , self.y_train , self.y_test = self.splitImages()
        
        # preprocess images 
        # self.x_processed_train = self.preprocessPipeline(self.x_train)
        # self.x_processed_test  = self.preprocessPipeline(self.x_test)
        
        # run the model 
        self.nvidia_model()
        self.model = tf.keras.models.load_model('carModel.h5')
        
    def get_data(self): 
        print("Fetching data .............................")
        
        
        columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
        
        df = pd.read_csv(os.path.join(self.datadir, 'driving_log.csv') , names = columns)
        pd.set_option('display.max_colwidth', None)
        
        df['center'] = df['center'].apply(self.path_leaf)
        df['left'] = df['left'].apply(self.path_leaf)
        df['right'] = df['right'].apply(self.path_leaf)
        
        return df
 
        
    def path_leaf(self, path): 
        
        head, tail = ntpath.split(path)
        
        return tail 
    
    def histPlot(self): 
        
        num_bins = 25; 
        hist, bins = np.histogram(self.data['steering'] , num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5 
        plt.bar(center, hist, width=0.05)
        
        samples_per_bin = 400
        plt.plot((np.min(self.data['steering']) , 
                  np.max(self.data['steering'])), 
                 (samples_per_bin, samples_per_bin))
        
        plt.show()
        
    def balanceData(self): 
        
        print("Balancing the data.............")
        
        print('total data:', len(self.data))
        remove_list = []
        samples_per_bin = 400
        num_bins = 25;
        
        hist, bins = np.histogram(self.data['steering'] , num_bins)
      
        for j in range(num_bins): 
            list_ = [] 
            
            for i in range(len(self.data['steering'])): 
                if self.data['steering'][i] >= bins[j] and self.data['steering'][i] <= bins[j + 1]:
                    list_.append(i)
                    
            list_ = shuffle(list_)
            list_ = list_[samples_per_bin:]
            remove_list.extend(list_)
        
        print("Data removed :" , len(remove_list))
        self.data.drop(self.data.index[remove_list], inplace=True)
        print("Final data :" , len(self.data['steering']))
        
        hist, _ = np.histogram(self.data['steering'], (num_bins))
        
        center = (bins[:-1] + bins[1:]) * 0.5 
        plt.bar(center, hist, width=0.05)
        
        plt.plot((np.min(self.data['steering']) , 
                  np.max(self.data['steering'])), 
                 (samples_per_bin, samples_per_bin))
        
    def load_img_steering(self): 
        
        img_path = [] 
        steering = [] 
        datadir = "data/IMG"
        
        for i in  range(len(self.data)): 
            indexed_data = self.data.iloc[i]
            center, left , right = indexed_data[0], indexed_data[1], indexed_data[2]
            img_path.append(os.path.join(datadir, center.strip()))
            steering.append(float(indexed_data[3]))
        img_path = np.asarray(img_path)
        steering = np.asarray(steering)
        
        return img_path, steering
    
    def splitImages(self): 
        
        x_train , x_test, y_train, y_test = train_test_split(self.image_paths, 
                                                             self.steering, 
                                                             test_size = 0.2, 
                                                             random_state=123)
        
        print("Training Samples: {}\nValid Samples: {}".format(len(x_train), len(x_test)))
        
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        ax[0].hist(y_train, bins=25, width=0.05, color='blue')
        ax[0].set_title("Training Set")
        ax[1].hist(y_test, bins=25, width=0.05, color='red')
        ax[1].set_title("Testing Set")
        
        return x_train, x_test, y_train, y_test 
    
    def imagePreprocessing(self, img): 
        
       # img = mpimg.imread(img)
        img = img[60:135, :, :] # crop out top of the image and hood of car
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) # convert to YUV for nvidia model
        img = cv2.GaussianBlur(img,(3,3), 0 ) # reduce noise on the image 
        img = cv2.resize(img, (200,66)) # resize to fit nvidia model 
        img = img/255 # normalize
        
        return img
    
    def viewRandom(self): 
        image = self.image_paths[100]
        originalImage = mpimg.imread(image)
        preprocessedImage = self.imagePreprocessing(image)
        
        fig, ax = plt.subplots(1, 2, figsize=(15,10) )
        fig.tight_layout()
        ax[0].imshow(originalImage)
        ax[0].set_title('Original Image')
        ax[1].imshow(preprocessedImage)
        ax[1].set_title('preprocessed Image')
        
    def preprocessPipeline(self, data): 
        
        return np.array(list(map(self.imagePreprocessing, data)))
    
    def checkProcessing(self): 
        
        plt.imshow(self.x_processed_train[random.randint(0, len(self.x_processed_train) - 1)])
        plt.axis('off')
        print(self.x_processed_train.shape)
        
    def nvidia_model(self):
 
        model = tf.keras.Sequential()         
        model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), input_shape=(66,200,3),activation='elu'))
        model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='elu'))
        model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='elu'))
        model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
        
        model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
        model.add(Dropout(0.5))
         
        model.add(Flatten())
        
        model.add(Dense(100, activation='elu'))
        model.add(Dropout(0.5))
         
        model.add(Dense(50, activation='elu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(10, activation ='elu'))
        model.add(Dropout(0.5))
         
        model.add(Dense(1))
         
        adam = tf.keras.optimizers.Adam
        optimizer= adam(lr=1e-3)
        model.compile(loss='mse', optimizer=optimizer)
        
        model.fit_generator(self.batchGenerator(100, 1), 
                            steps_per_epoch=300, epochs=10, 
                            validation_data= self.batchGenerator(100, 0), 
                            validation_steps=200, verbose=1, shuffle=1)
        
        #model.fit(self.x_processed_train, 
        #          self.y_train, 
        #          validation_data=(self.x_processed_test, self.y_test), 
         #         epochs=30, batch_size=100, verbose=1, shuffle=1)
        
        model.save('carModel.h5')
        print("Model saved")
        
    def zoom(self, img): 
        zoom = iaa.Affine(scale=(1, 1.3))
        
        return zoom.augment_image(img)
    
    def pan(self, img): 
        pan = iaa.Affine(translate_percent= {"x" : (-0.1,0.1), "y" : (-0.1, 0.1)})
        
        return pan.augment_image(img)
    
    def img_random_brightness(self, img): 
        
        altered_brightness = iaa.Multiply((0.2, 1.2))
        
        return altered_brightness.augment_image(img)
    
    def flipping(self, img, steering_angle):
        
        img = cv2.flip(img, 1)
        steering = - steering_angle
        
        return img, steering
        
    
    def checkAugment(self):
        
        random_index =  random.randint(0, 1000)

        
        image = self.image_paths[random_index]
        steering = self.steering[random_index]
        original = mpimg.imread(image)
        changed , angle = self.flipping(original, steering)
        
        
        fig , ax = plt.subplots(1,2, figsize=(15, 10))
        fig.tight_layout()
        
        ax[0].imshow(original)
        ax[0].set_title('Original Image')
        ax[1].imshow(changed)
        ax[1].set_title("Augmented Image")
        
    def random_augment(self, img, steering_angle):
        
        img = mpimg.imread(img)
        
        if np.random.rand() < 0.5 : 
            img = self.pan(img)
            
        if np.random.rand() < 0.5 : 
            img = self.zoom(img)
            
        if np.random.rand() < 0.5 : 
            img = self.img_random_brightness(img)
            
        if np.random.rand() < 0.5 : 
            img , steering_angle = self.flipping(img, steering_angle)
        
        return img, steering_angle
    
    def viewRandomAugment(self): 
        ncol = 2 
        nrow = 10 
        fig , ax = plt.subplots(nrow, ncol, figsize=(15,50))
        fig.tight_layout()
        
        for i in range(10): 
            
            random_num = random.randint(0 , len(self.image_paths) - 1)
            
            random_image = self.image_paths[random_num]
            random_steer = self.steering[random_num]
            
            original_image = mpimg.imread(random_image)
            augment_image, augment_steering = self.random_augment(random_image, random_steer)
            
            ax[i][0].imshow(original_image)
            ax[i][0].set_title('Original image')
            
            ax[i][1].imshow(augment_image)
            ax[i][1].set_title('Augmented image')
            
    def batchGenerator(self, batch_size, istraining): 
        
        while True : 
            batch_img = []
            batch_str = []
            
            for i in range(batch_size): 
                random_index = random.randint(0, len(self.image_paths) -1)
                
                img, steering = self.random_augment(self.image_paths[random_index] ,
                                                    self.steering[random_index])
            else: 
                img = mpimg.imread(self.image_paths[random_index])
                steering = self.steering[random_index]
                
            
            img = self.imagePreprocessing(img)
            batch_img.append(img)
            batch_str.append(steering)
            
        yield (np.asarray(batch_img) , np.asarray(batch_str))
        
    
    def batchPlot(self): 
        x_train_gen, y_train_gen = next(self.batchGenerator(1,1))
        x_test_gen, y_test_gen = next(self.batchGenerator(1,0))
        
        fig , ax = plt.subplots(1, 2, figsize=(15,10))
        fig.tight_layout()        
        
        ax[0].imshow(x_train_gen[0])
        ax[0].set_title('Training image')
            
        ax[1].imshow(x_test_gen[0])
        ax[1].set_title('Validation image')        
        
                       
c = Car()