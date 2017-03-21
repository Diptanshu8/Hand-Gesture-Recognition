import cv2
import os
import numpy as np
from PIL import Image
def image_loading(img_addr):
    #function for checking if image is loaded successfully
    if os.path.isfile(img_addr):
        img = cv2.imread(img_addr)
        if img is None:
            print("Exiting because image could not be read/loaded properly\n")
            exit()
        return img
    else:
            print("The input image path doesn't exist\n")
            exit()

def showing_image(img,title):
    if img is None:
        print("The image cannot be displayed. Please debug\n")
        exit()
    cv2.imshow(title,img)
    cv2.waitKey(0)

def closing_image(title):
    cv2.destroyWindow(title)

def left_side_of_hand(img):
    rows,cols,channels = img.shape
    #just a hack for now
    background = img[0][0]
    col = 0
    while col<cols:
        row = 0
        while row<rows:
            temp = []
            temp = [item for item in img[row][col] if item not in background]
            if temp != []:
                return col
            row+=1
        col+=1
    return -1

def right_side_of_hand(img):
    rows,cols,channels = img.shape
    #just a hack for now
    background = img[0][0]
    col = cols-1
    while col>0:
        row = 0
        while row<rows:
            temp = []
            temp = [item for item in img[row][col] if item not in background]
            if temp != []:
                return col
            row+=1
        col-=1
    return -1

def top_of_hand(img):
    rows,cols,channels = img.shape
    #just a hack for now
    background = img[0][0]
    row = 0
    while row<rows:
        col = 0
        while col<cols:
            temp = []
            temp = [item for item in img[row][col] if item not in background]
            if temp != []:
                return row
            col+=1
        row+=1
    return -1

def bottom_of_hand(img):
    rows,cols,channels = img.shape
    #just a hack for now
    background = img[0][0]
    row = rows-1
    while row>0:
        col = 0
        while col<cols:
            temp = []
            temp = [item for item in img[row][col] if item not in background]
            if temp != []:
                return row
            col+=1
        row-=1
    return -1

def generating_boundary_of_hand(img):
    rows,cols= img.shape
    blank_image = np.zeros((rows,cols,1), np.uint8)
    blank_image.fill(0)
    row,col = 0,0
    values = []
    while row<rows-1:
        col = 0
        while col<cols-1:
            if img[row][col+1]!=img[row][col]:
                blank_image[row][col]=255
            col+=1
        row+=1
    return blank_image

image_addr = "Chirag1F.jpg"
img = image_loading(image_addr)
#img_c = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,img_binary_otsu= cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((5,5),np.uint8)
img_eroded = cv2.erode(img_binary_otsu,kernel,iterations = 1)
closing = cv2.morphologyEx(img_eroded, cv2.MORPH_OPEN,  kernel)
hand_boundary = generating_boundary_of_hand(closing)

showing_image(img,"Original image")
#showing_image(img_gray,"Gray image")
#showing_image(img_binary_otsu,"Binary image")
showing_image(img_eroded,"eroded image")
showing_image(closing,"temp image")
showing_image(hand_boundary,"boundary image")

cv2.destroyAllWindow()
"""
left = left_side_of_hand(img)
print left
right = right_side_of_hand(img)
print right
top = top_of_hand(img)
print top
bottom = bottom_of_hand(img)
print bottom
"""
"""
showing_image(img,"original image")
showing_image(img_c,"transformed image")
closing_image("original image")
closing_image("transformed image")
"""
