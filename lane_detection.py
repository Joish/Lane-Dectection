# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:12:18 2019

@author: Joish
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def blur_image(image):
    return cv2.GaussianBlur(image,(5,5),0)

def edge_detect(image):
    return cv2.Canny(image,50,150)

def mask_image(image):
    height = image.shape[0]
    mask = np.zeros_like(image)
    poly = np.array([[(200,height),(1100,height),(550,250)]])
    cv2.fillPoly(mask,poly,255)
    return mask

def region_intrest(mask,image):
    return cv2.bitwise_and(mask,image)

def display_lines(image,lines):
    copy = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(copy,(x1,y1),(x2,y2),(255,0,0),10)
    return copy

def find_cordi(image,lines):
    slope = lines[0]
    intercept = lines[1]
    
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)

    return np.array([x1,y1,x2,y2])

def average_slope(image,lines):
    left_line = []
    right_line = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        params = np.polyfit((x1,x2),(y1,y2),1)
        slope = params[0]
        intercept = params[1]
        
        if slope > 0:
            right_line.append((slope,intercept))
        else:
            left_line.append((slope,intercept))
            
    avg_left_line = np.average(left_line,axis=0)
    avg_right_line = np.average(right_line,axis=0)
    
    left = find_cordi(image,avg_left_line)
    right = find_cordi(image,avg_right_line)
    
    return np.array([left,right])

## code for lane dection in Picture ## 

#img = cv2.imread('test_image.jpg')
#img_copy = np.copy(img)
#gray = cv2.cvtColor(img_copy , cv2.COLOR_BGR2GRAY)
#blur = blur_image(gray)
#edge = edge_detect(blur)
#mask = mask_image(edge)
#roi = region_intrest(mask,edge)
#
#line = cv2.HoughLinesP(roi,2,np.pi/180,100,minLineLength = 40,maxLineGap = 5)
#average_line = average_slope(img_copy,line)
#line_image = display_lines(img_copy,average_line)
#
#combo = cv2.addWeighted(img_copy,0.8,line_image,1 ,1 ) #last parameter is a gamma paramert , not a big deal
#
#cv2.imshow('image',combo)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

## code for lane dection in Video ##

cap = cv2.VideoCapture('video.mp4')
while (cap.isOpened()):
    _,frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    blur = blur_image(gray)
    edge = edge_detect(blur)
    mask = mask_image(edge)
    roi = region_intrest(mask,edge)
    
    line = cv2.HoughLinesP(roi,2,np.pi/180,100,minLineLength = 40,maxLineGap = 5)
    average_line = average_slope(frame,line)
    line_image = display_lines(frame,average_line)
    
    combo = cv2.addWeighted(frame,0.8,line_image,1 ,1 ) #last parameter is a gamma paramert , not a big deal
    
    cv2.imshow('image',combo)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
