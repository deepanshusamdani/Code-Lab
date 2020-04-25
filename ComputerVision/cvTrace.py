#!/usr/bin/python3

import cv2

#loading camera
cap=cv2.VideoCapture(0)

#read the images 
img1=cap.read()[1]
img2=cap.read()[1]
img3=cap.read()[1]

#create difference between first and second
difference_np=img1-img2
difference1=cv2.subtract(img1,img2)

#most accurate or absoluate result by cv2
f_diff1=cv2.absdiff(img1,img2)
f_diff2=cv2.absdiff(img2,img3)

f_diff=cv2.bitwise_xor(f_diff1,f_diff2)
cut_diff=f_diff[120:320]

cv2.imshow("normal",f_diff1)
cv2.imshow("img1",img1)
#cv2.imshow("bitwise",f_diff)
#cv2.imshow("cut_diff",cut_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()



