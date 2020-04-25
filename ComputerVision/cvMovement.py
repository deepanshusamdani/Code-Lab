#!/usr/bin/python3

import cv2
import time
import numpy as np 

def movement(x,y,z):
#taking the absolute difference of images
	df1=cv2.absdiff(x,y)
	df2=cv2.absdiff(y,z)

	img3=df1&df2
	return img3



#loading camera
cap=cv2.VideoCapture(0)

#read the image  into 'takepicture as tp'
tp1=cap.read()[1]
tp2=cap.read()[1]
tp3=cap.read()[1]

#convert into gray color scale
gray1=cv2.cvtColor(tp1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(tp2,cv2.COLOR_BGR2GRAY)
gray3=cv2.cvtColor(tp3,cv2.COLOR_BGR2GRAY)

#detect the movement at real time 
while True:
	final=movement(gray1,gray2,gray3)

	cv2.imshow('diff',final)
	cv2.imshow('color',tp1)
	
	status,frame=cap.read()
	
	g4=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #swap the images into one another
	gray1=gray2
	gray2=gray3
	gray3=g4

	# res = 'smj'+str(i)+'.jpg'
	# cv2.imwrite(res,g4)

	if cv2.waitKey(1) and 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
