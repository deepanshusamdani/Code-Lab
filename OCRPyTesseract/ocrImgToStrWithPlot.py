#!/usr/bin/python3

#importing libraries
import pytesseract as pt 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 

#read the image
image =cv2.imread("/home/deepu/Desktop/directory/images/smile.jpg")

#another way to read image
#filePath = '/home/deepu/Desktop/directory/images/smile.jpg'
# image = Image.open(filePath)

#read image text 
string = pt.image_to_string(image)
print(string)

#copy of the image 
image_copy = image.copy()

#target word to search for
target_word = "WORLD"

#get all data from image
data = pt.image_to_data(image,output_type=pt.Output.DICT)
# print(data)

#get all the occurences of that word
word_occurence = [i for i, word in enumerate(data['text']) if word == target_word ]
print("occurence of word at position: ",word_occurence)

# Another way of above for loop
# c= []
# for i , word in enumerate(data['text']):
# 	if(word == tgt_word):
# 		print("%d %s" %(i,word))
# 		c.append(i)

for occ in word_occurence:
	w = data['width'][occ]
	h = data['height'][occ]
	t = data['top'][occ]
	l = data['left'][occ]

	#define all the surrounding box points
	p1 = (l, t)
	p2 = (l + w, t)
	p3 = (l + w, t + h)
	p4 = (l, t + h)
	
	# draw the 4 lines (rectangular)
	image_copy = cv2.line(image_copy, p1, p2, color=(255, 0, 0), thickness=5)
	image_copy = cv2.line(image_copy, p2, p3, color=(255, 0, 0), thickness=5)
	image_copy = cv2.line(image_copy, p3, p4, color=(255, 0, 0), thickness=5)
	image_copy = cv2.line(image_copy, p4, p1, color=(255, 0, 0), thickness=5)
	
plt.imsave("/home/deepu/Desktop/directory/images/all_dog_words.png", image_copy)
plt.imshow(image_copy)
plt.show()