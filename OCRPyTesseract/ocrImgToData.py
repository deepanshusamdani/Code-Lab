#!/usr/bin/python3
import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt

#img = cv2.imread('/home/deepu/Desktop/cat1.jpeg')
img = cv2.imread('/home/deepu/Desktop/directory/images/smile.jpg')

#image_copy = img.copy()

#Read the text of image using ImageToString
string = pytesseract.image_to_string(img)
print(string)

#output type dictionary => output_type=Output.DICT
d = pytesseract.image_to_data(img, output_type=Output.DICT)
#print(d)

#for i in range(len(d['line_num'])):
#    if d['line_num'][i] == 6:
#        print(d['text'][18])
        
n_boxes = len(d['level'])

#createing box(rectangle/lines) over the letters on the image
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #p1 = (x,y)
    #p2 = (x+w,y)
    #p3 = (x+w, y+h)
    #p4 = (x, y+h)
    
    #image_copy = cv2.line(image_copy, p1, p2, color=(255, 0, 0), thickness=5)
    #image_copy = cv2.line(image_copy, p2, p3, color=(255, 0, 0), thickness=5)
    #image_copy = cv2.line(image_copy, p3, p4, color=(255, 0, 0), thickness=5)
    #image_copy = cv2.line(image_copy, p4, p1, color=(255, 0, 0), thickness=5)


cv2.imshow('img', img)
cv2.waitKey(0)

#plt.imshow(image_copy)
#plt.show()
