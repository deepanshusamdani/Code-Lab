#!/usr/bin/python3

#!/usr/bin/python3

#Program: Convert PDF file to String or may be save in another file format

#importig libraries
from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path 
import os 
  
 #read the pdf file
PDF_file = "/home/deepu/Desktop/d.pdf"
    
pages = convert_from_path(PDF_file, 500) 
  
image_counter = 1

for page in pages: 
    filename = "page_"+str(image_counter)+".jpg"
    page.save(filename, 'JPEG') 
    image_counter = image_counter + 1
  
#"out_text.odt"
#In any file format
outfile = "out_text.txt"

f = open(outfile, "a") 

for i in range(1, image_counter): 

    filename = "page_"+str(i)+".jpg"
    text = str(((pytesseract.image_to_string(Image.open(filename))))) 
    print(text)

    text = text.replace('-\n', '')     
    f.write(text) 
  
f.close()