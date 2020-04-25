#!/usr/bin/python3

import cv2 as cv2
import cv2 as cv
import numpy as np
import os

FILE_OUTPUT = 'ooo.avi'


#if the file with the same name iscreated already the itreplace the file with new name
if os.path.isfile(FILE_OUTPUT):
	os.remove(FILE_OUTPUT)


#loading camera in 'capture as cap' 
cap = cv2.VideoCapture(0)


#fourcc readthe byte type and XVID is a video supporter 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (640,480))

#checking for detection
while cap.isOpened():
	# real data of captured image stored in frame
	status, frame = cap.read()
	if status == True:
		frame = cv2.flip(frame,1)
		out.write(frame)
		cv2.imshow('frame1',frame)
		if cv2.waitKey(1) & 0xFF ==ord('q'):                      #0xFF --> handle keyboard instruction
	        	print("vfdv")


cap.release()
out.release()
cv2.destroyAllWindows()













