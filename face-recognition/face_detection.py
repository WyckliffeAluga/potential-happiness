# -*- coding: utf-8 -*-
"""
Created on Sun May 10 22:11:28 2020

@author: wyckliffe
"""

# HOG Face Detector 
# CNN Face Detector 

import cv2 
import face_recognition

class Detector(): 
    
    def __init__(self):
        
        self.img = self.load_image()
        
    
    def load_image(self): 
        
        return cv2.imread("img1.jpg")

    def HOG_detector(self): 
        
        face_location = face_recognition.face_locations(self.img, model='hog')
        
        return (face_location)
    
    def CNN(self): 
        face_location = face_recognition.face_locations(self.img, model='cnn')
        
        return len(face_location)
    
    def show_locations(self): 
        
        for index, location in enumerate(self.HOG_detector()): 
            top, right, bot, left = location 
            
            img = self.img[top:bot, left:right]
            
            cv2.imshow("Face No :" + str(index + 1), img)

d = Detector()