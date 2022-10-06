#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 10:38:31 2022

@author: yannis
"""

import cv2
import numpy as np

# Load the image
img = cv2.imread('./data/data_test.jpg')



def extractContours(img):
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(img, 255/3,255, 3)
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = 10)
    thresh = cv2.erode(thresh,None,iterations = 10)

    # Find the contours
    contours,hierarchy = cv2.findContours(thresh,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

    return(contours)


rects = []
areas = []

allContours = extractContours(img)

#Calculate Area 
for cnt in allContours:
    areas.append(cv2.contourArea(cnt))
    
gridArea = max(areas)
gridAreaIndex = areas.index(gridArea)

gridContours = allContours[gridAreaIndex]

x,y,w,h = cv2.boundingRect(gridContours)

new_img = img[y:y+h, x:x+w]

lettersContour = extractContours(new_img)


# for cnt in lettersContour:
#     area = cv2.contourArea(cnt)
#     if (area > 6000 and area < 11000):
#         areas.append(area)
#         x,y,w,h = cv2.boundingRect(cnt)
#         rects.append([x,x+w,y,y+h])
        
#         cv2.rectangle(new_img,
#                       (x,y),(x+w,y+h),
#                       (0,255,0),
#                       2)
#         cv2.imshow('img',new_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#For each contour, find the bounding rectangle and draw it
for cnt in lettersContour:
    area = cv2.contourArea(cnt)
    if (area > 6000 and area < 10000):
        areas.append(area)
        x,y,w,h = cv2.boundingRect(cnt)
        rects.append([x,x+w,y,y+h])
        
        cv2.rectangle(new_img,
                      (x,y),(x+w,y+h),
                      (0,255,0),
                      2)

first_rect = rects[4]

first_img = img[first_rect[2]:first_rect[3], first_rect[0]:first_rect[1]]

cv2.imshow('img',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


for i in range(len(rects)):
    letter = new_img[rects[i][2]:rects[i][3], rects[i][0]:rects[i][1]]
    letter = cv2.resize(letter, (128,128))
   
    cv2.imshow(f'letter_{i}',letter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

