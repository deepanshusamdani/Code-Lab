#!/usr/bin/python3

import numpy as np
import cv2
 
from matplotlib import pyplot as plt
import tkinter
'''
img = cv2.imread('apple.jpeg',1)
plt.imshow(img,interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
'''

#read the image in to the variable 'img'
img = cv2.imread('apple.jpeg',0)

#taking thedifferent threshold of iamge
# taking thresh_binary threshold
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# taking thresh_binary_inv threshold
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

# taking thresh_trunc threshold
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)

## taking thresh_tozero threshold
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)

# taking thresh_tozero_inv threshold
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
