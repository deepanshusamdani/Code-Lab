#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#https://www.analyticsvidhya.com/blog/2019/03/opencv-functions-computer-vision-python/

#geometric transformation
#https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
"""
Created on Sun Apr 12 12:35:53 2020

@author: deepu
"""

#detect color in the Image using ArgumentPassing(img as parameter) 
#at the run run time

"""
Run This Program on the terminal using below command 
#python3 cvImageColorDetection.py  --image apple.jpeg

"""

import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

#image = cv2.imread('/home/deepu/Desktop/directory/Image/apple.jpg')
boundaries = [
	          ([17, 15, 100], [50, 56, 200]),
	          ([86, 31, 4],   [220, 88, 50]),
	          ([25, 146, 190],[62, 174, 250]),
	          ([103, 86, 65], [145, 133, 128])
             ]

for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
    


