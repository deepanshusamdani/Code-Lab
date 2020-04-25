#!/usr/bin/python3

import cv2

#starting camera
cam=cv2.VideoCapture(0)

#checking for web cam

#detect a only_color at real time 
while cam.isOpened():
	status,frame=cam.read()
#				starting color,   end color range
	only_red=cv2.inRange(frame,(40,0,0),(255,20,20))
	only_green=cv2.inRange(frame,(0,30,0),(20,255,20))
	cv2.imshow('window_red',only_red)
	cv2.imshow('window_green',only_green)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cv2.destroyAllWindows()
cam.release()


#detect a only_color at save images

'''
img=cv2.imread("red.jpg",1)
print(img.shape)
only_red=cv2.inRange(img,(0,0,40),(30,30,255))
cv2.imshow("window",only_red)
cv2.waitKey(0)
'''
