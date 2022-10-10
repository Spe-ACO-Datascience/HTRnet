# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:31:21 2022

@author: Emilie
"""

import cv2 
import numpy as np

# Load the image
path = 'C:\\Users\\CamPc\\Documents\\ACO\\3A\\HTRnet\\data\\all_data-7-1.png'
img = cv2.imread(path)


def cut_contours(img, minarea, maxarea):
    
    # Detect contours 
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
    #Going through every contours found in the image.
    approx = []
    for cnt in contours :
        approx.append(cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True))
    
    
    rects = []
    
    # Get coordinates of contours 
    for cnt in approx:
        area = cv2.contourArea(cnt)
        if (area > minarea and area < maxarea):
            x,y,w,h = cv2.boundingRect(cnt)
            rects.append([x,x+w,y,y+h])
    
    # Resize image on contours 
    for i in range(len(rects)):
        letter = img[rects[i][2]:rects[i][3], rects[i][0]:rects[i][1]]
        letter = cv2.resize(letter, (128,128))
        
        cv2.imshow(f'letter_{i}',letter)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

cut_contours(img, 15000, 30000)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    