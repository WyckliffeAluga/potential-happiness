# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:28:38 2020

@author: wyckliffe
"""


import cv2 
import numpy as np


# edge detection (sharp changes in intensity in adjacent pixels )
# Gradient: measure of the change in brightness over adjacent pixels 

class EdgeDetection(): 
    
    def __init__(self, img): 
        self.img = img
        
    
    def grayImage(self): 
        
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
    
    def smoothening(self): 
        """
        Gaussian blur to remove noise 

        Returns
        -------
        A blurred image

        """
        return cv2.GaussianBlur(self.grayImage(), (5, 5), 0)
    
    def simpleDetector(self):
        """
        Measure the adjacent changes in intensity in all directions, x and y
        """
        return cv2.Canny(self.smoothening(), 50, 150)
    
    def interestRegion(self): 
        
        """
        Create a region of interest where we would like to detect the lanes
        """
      
        polygons = np.array([[(200, self.simpleDetector().shape[0]) , (1100 , self.simpleDetector().shape[0]) , (550 , 250)]])
        mask = np.zeros_like(self.simpleDetector())
        cv2.fillPoly(mask, polygons, 255)
        
        return mask
    
    def bitwise_cropping(self):
        """
        take two images and convert to binary then do bitwise_and 
        meaning compare two binaries and then take zero unless both are ones
        
        Computing the bitwise_and of both images, takes the bitwise and of each homologoous, 
        pixel in both arrays, ultimately masking the canny image to only show the region of interest 
        traced by the polygonial contour of the mask.
        """
        
        return cv2.bitwise_and(self.simpleDetector(), self.interestRegion())
    
    def line_detection(self): 
        
        """
        Hough transform 
        """
        return cv2.HoughLinesP(self.bitwise_cropping(), 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
        
    def display_lines(self, lines) : 
        """
        Displays lanes on a black image

        Returns
        -------
        line_image : TYPE
            DESCRIPTION.

        """
        line_image = np.zeros_like(self.img)
        
    
        
        if lines is not None: 
            # loop through the list
            for line in lines: 
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image , (x1, y1), (x2, y2), (255,0,0), 10)
        
        return line_image
    
    def lanes(self) :
        """
        

        Returns
        -------
        lanes on the colored image

        """
        
        return cv2.addWeighted(self.display_lines(self.line_detection()), 0.8, self.img, 1 , 1)
    
    def make_cordinates(self , line_parameters): 
        
        slope, intercept = line_parameters
        y1 = self.img.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
        return np.array([x1, y1, x2, y2])
  
        
    def average_slope_intercepts(self):
        
        left_fit = []
        right_fit = []
        
        for line in self.line_detection() : 
           x1, y1,x2, y2 =  line.reshape(4)
           parameters = np.polyfit((x1, x2), (y1, y2) , 1)
           slope = parameters[0]
           intercept = parameters[1]
           
           if slope < 0 : # negative slope 
               left_fit.append((slope, intercept))
           else:
               right_fit.append((slope, intercept))
                   
       
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        
        left_line  = self.make_cordinates(left_fit_average)
        right_line = self.make_cordinates(right_fit_average)
        
        return np.array([left_line, right_line])
        
    def averaged_lines_image(self): 
        
        averaged_lines = self.average_slope_intercepts()
        line_image = self.display_lines(averaged_lines)
       
        return cv2.addWeighted(line_image, 0.8, self.img, 1, 1)
        
        
    def show(self, imageMatrix):
      cv2.imshow("result", imageMatrix)
     
        
# video capture object 
cap = cv2.VideoCapture("test2.mp4")

while(cap.isOpened()): 
    _, frame = cap.read()       
    eyes = EdgeDetection(frame)
    eyes.show(eyes.averaged_lines_image())
    
    if cv2.waitKey(1) == ord("q"): 
        break 
    
cap.release()
cv2.destroyAllWindows()























