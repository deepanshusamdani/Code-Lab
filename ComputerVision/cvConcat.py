#!/usr/bin/python3

import cv2
import numpy as np

image = cv2.imread('apple.jpeg')

(h,w) = image.shape[:2]
print("h: ",h,"\n","w: ", w)

# I just resized the image to a quarter(.5,.5) / double(2,2) of its original size
#resize(src, dst, Size(), 0.5, 0.5, interpolation);
#behind_Calc: dsize(output_image_size) = Size(round(fx*src.cols), round(fy*src.rows))

image = cv2.resize(image, (0, 0), None, 2, 2)

(h,w) = image.shape[:2]
print("h1: ",h,"\n","w1: ", w)

grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print("grey: ",grey[0][1])

# Make the grey scale image have three channels
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_vertical = np.vstack((image, grey_3_channel))
print("numpy_vertical: ",numpy_vertical)
numpy_horizontal = np.hstack((image, grey_3_channel))

numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
print("numpy_vertical_concat: ",numpy_vertical_concat)

numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('Main', image)
#cv2.imshow('Numpy Vertical', numpy_vertical)
#cv2.imshow('Numpy Horizontal', numpy_horizontal)
cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)


cv2.waitKey(0)
cv2.destroyAllwindows()
