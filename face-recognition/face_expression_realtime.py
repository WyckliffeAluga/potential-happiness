# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:07:54 2020

@author: wyckliffe
"""

import numpy as np 
import cv2 
from keras.preprocessing import image 
from keras.models import model_from_json 
import face_recognition


class RTD(): 
    
    def __init__(self): 
        
        self.video = self.capture()
        model = model_from_json(open('models/expression_model.json', 'r').read())
        self.model = model.load_weights('models/expression_model.h5')
        self.label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'suprise', 'neutral')
    
    
    
    def capture(self): 
        
        return cv2.VideoCapture(0)     
    
    def preprocess(self, img): 
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(img, (48, 48))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/ 255
        return img
        
    def emotions(self): 
        locations = [] 
        
        while True: 
            ret , frame = self.video.read() 
            frame_small = cv2.resize(frame,(0,0) , fx=0.25, fy =0.25)
            locations = face_recognition.face_locations(frame_small,
                                                        number_of_times_to_upsample=2, 
                                                        model='hog')
            
            for index, location in enumerate(locations): 
                top, right, bot, left = location 
                right  = right * 4 
                bot  = bot * 4 
                left  = left * 4 
                top  = top * 4 
                cv2.rectangle(frame, (left, top), (right, bot), (0,0,255), 2)
                face =  frame[top:bot, left:right]
                face = self.preprocess(face)
                prediction = self.model.predict(face)
                
            cv2.imshow("Face", frame)
        
            if cv2.waitKey(1) & 0xFF == ord("q") : 
                break
        self.video.release() 
    
        cv2.destroyAllWindows()
        

r = RTD()