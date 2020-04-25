#!/usr/bin/python3

import cv2

# for reading the image
img =cv2.imread('cat.jpeg')

image=img[0:224,0:225]
newimg=img[0:122,0:112]

print("enter your choice:")
print("1.cut in half")
print("2.cut in one-fourth")
print("3.cut in one-eigtht")
 
choice =input()

if choice == '1':
	crop1=img[0:122,0:112]
	crop2=img[0:112,0:224]
	cv2.imshow('half image',crop1)
	cv2.imshow('half image2',crop2)
	cv2.waitKey(0)

if choice=='2':
	print("1.top-left \n 2.top-right \n 3. bottom-left \n 4.bottom-right")
	ch=input("which one-fourth portion")
	

	if ch=='1':
		cropped=img[0:112,0:112]
		
	if ch=='2':
		cropped=img[0:112,112:224]

	if ch=='3':
		cropped=img[112:224,0:112]

	if ch=='4':
		cropped=img[122:224,112:224]
	
	cv2.imshow('one fourth',cropped)
	cv2.waitKey(0)

cv2.destroyAllWindows()


