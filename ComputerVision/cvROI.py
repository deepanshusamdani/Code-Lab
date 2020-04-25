#!/usr/bin/python3

import cv2
import numpy as np
 

# Read image
im = cv2.imread("apple.jpeg")
print(im.shape)  
print("type: ",type(im)) 
# Select ROI
r = cv2.selectROI(im)
print(r)
x1=r[0]
y1=r[1]
x2=r[2]
y2=r[3]

print(r[0],r[1],r[2],r[3])

x=im[y1:(y1+y2),x1:(x1+x2)]
print("type X:",type(x))
cv2.imshow("roi_window",x)
#cv2.imwrite('a.jpeg',x)
#print(x.shape)
	
    # Crop image
#imcrop = im[0:112,0:112]
#r = cv2.selectROI(imcrop)
    # Display cropped image

#print(x1,y1)
#cor=cv2.line(im,(y1,x1),(y2+y1,x2+x1),(0,0,255),4)
#print(cor.shape)
#cv2.imshow("Image", imcrop)
cv2.imshow("Imagess", im )

cv2.waitKey(0)

cv2.destoryAllWindows()
