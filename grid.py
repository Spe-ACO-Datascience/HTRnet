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
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Test Emilie 
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


approx = []
#Going through every contours found in the image.
for cnt in contours :
  
    approx.append(cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True))
    # draws boundary of contours.
    #cv2.drawContours(img, [approx], 0, (0, 0, 255), 2) 
  
    # Used to flatted the array containing
    # the co-ordinates of the vertices.
    #n = approx.ravel() 
    #i = 0
  
    #for j in n :
    #    if(i % 2 == 0):
    #        x = n[i]
    #        y = n[i + 1]
  
    #    i = i + 1



  
rects = []
areas = []


# for cnt in approx:
#     #cnt = lettersContour[1]
#     area = cv2.contourArea(cnt)
#     if area > 15000 and area < 30000 :
#         areas.append(area)

# import matplotlib.pyplot as plt

# x = [i for i in range(1,len(areas)+1)]
# plt.plot(areas)
# plt.show()





for cnt in approx:
    #cnt = lettersContour[1]
    area = cv2.contourArea(cnt)
    if (area > 15000 and area < 60000):
        print(area)
        areas.append(area)
        x,y,w,h = cv2.boundingRect(cnt)
        rects.append([x,x+w,y,y+h])
        
        cv2.rectangle(img,
                      (x,y),(x+w,y+h),
                      (0,255,0),
                      2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#first_rect = rects[4]

#first_img = img[first_rect[2]:first_rect[3], first_rect[0]:first_rect[1]]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()