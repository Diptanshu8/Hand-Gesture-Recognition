import cv2
import os

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


image_addr = "1.jpg"
img = image_loading(image_addr)
img_c = cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)
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
