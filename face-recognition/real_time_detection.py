# -*- coding: utf-8 -*-
"""
Created on Sun May 10 22:36:54 2020

@author: wyckliffe
"""
import cv2 
import face_recognition


class RTD(): 
    
    def __init__(self): 
        
        self.video = self.capture()
    
    
    def capture(self): 
        
        return cv2.VideoCapture(0)
    
    
    def locations(self): 
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
            cv2.imshow("Face", frame)
        
            if cv2.waitKey(1) & 0xFF == ord("q") : 
                break
        self.video.release() 
    
        cv2.destroyAllWindows()
        
        
    def prerecoded(self, video_path):
        
        video = cv2.VideoCapture(video_path)
        
        locations = [] 
        
        while True: 
            ret , frame = video.read() 
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
            cv2.imshow("Face", frame)
        
            if cv2.waitKey(1) & 0xFF == ord("q") : 
                break
        self.video.release() 
    
        cv2.destroyAllWindows()
    
    def blurrFace(self): 
        
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
                
                face = frame[top:bot, left:right]
                
                face = cv2.GaussianBlur(face, (99,99), 30)
                
                frame[top:bot, left:right ] = face
                      
            cv2.imshow("Face", frame)
        
            if cv2.waitKey(1) & 0xFF == ord("q") : 
                break
            
        self.video.release() 
    
        cv2.destroyAllWindows()
        
        
r = RTD()