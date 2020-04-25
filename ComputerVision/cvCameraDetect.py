#!/usr/bin/python3

import cv2
import time
import numpy as np


def call_camera():
	i=0
	data=[]
	
	while 1:
		
		# start the webcam and read the data into the frame
		cap = cv2.VideoCapture(0)
		frame = cap.read()[1]
		print(frame)
		# convert the image into grayscale
		gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		cv2.imshow('gray_img',gray_image)
		# code for saving the image in a loop
		res = 'image'+str(i)+'.jpg'
		cv2.imwrite(res,gray_image)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break		

		cv2.destroyAllWindows()
		# release the camera so that it can be again started
		cap.release()
		# camera wil start after 2 seconds 
		time.sleep(2)

		i=i+1


call_camera()
