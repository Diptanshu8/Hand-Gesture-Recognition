import cv2
import os
import time
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
    v = np.median(img)
    sigma = 0.33
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    hand_boundary = cv2.Canny(img, lower, upper)
    return hand_boundary

def vertical_identification(img):
    rows,cols = img.shape
    mid_row = rows/2
    mid_col = cols/2
    offset_width = 0.05 #10% mask width
    offset_height = 0.20 #40% mask height
    row_mask = range(mid_row-int(rows*offset_height),mid_row+int(rows*offset_height))
    col_mask = range(0,int(cols*offset_width))+range(cols-int(cols*offset_width),cols)
    total_pixels = len(row_mask)*len(col_mask)
    identified_pixels,boundary_intensity = 0,255
    string = ""
    for row in row_mask:
        for col in col_mask:
            if img[row][col] == 255:
                identified_pixels+=1
    return (identified_pixels/(total_pixels+0.0))

def horizontal_identification(img):
    rows,cols = img.shape
    mid_row = rows/2
    mid_col = cols/2
    offset_width = 0.05 #10% mask width
    offset_height = 0.20 #40% mask height
    col_mask = range(mid_col-int(cols*offset_height),mid_col+int(cols*offset_height))
    row_mask = range(0,int(rows*offset_width))+range(rows-int(rows*offset_width),rows)
    total_pixels = len(row_mask)*len(col_mask)
    identified_pixels,boundary_intensity = 0,255
    for row in row_mask:
        for col in col_mask:
            if img[row][col] == 255:
                identified_pixels+=1
    return (identified_pixels/(total_pixels+0.0))

def image_processing(image_addr):
    img = image_loading(image_addr)
    #img_c = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,img_binary_otsu= cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    img_eroded = cv2.erode(img_binary_otsu,kernel,iterations = 1)
    closing = cv2.morphologyEx(img_eroded, cv2.MORPH_OPEN,  kernel)
    hand_boundary=generating_boundary_of_hand(closing)
    vertical_ratio = vertical_identification(hand_boundary)
    horizontal_ratio = horizontal_identification(hand_boundary)
    if vertical_ratio>horizontal_ratio:
        print(image_addr +" is horizontally oriented image")
    elif vertical_ratio<horizontal_ratio:
        print(image_addr +" is vertically oriented image")
    else:
        print("ambigious data. DISCARD IT")
    #showing_image(img,"Original image")
    #showing_image(img_binary_otsu,"Binary image")
    #showing_image(img_binary_otsu,"Binary image")
    #showing_image(img_eroded,"eroded image")
    #showing_image(closing,"temp image")
    #showing_image(hand_boundary,"boundary image")

images = ["Chirag1F.jpg","Chirag2F.jpg","ChiragPF.jpg","Chirag5F.jpg"]
# Generate trackbar Window Name
TrackbarName = "Trackbar"
# Make Window and Trackbar
cv2.namedWindow("WindowName")
cv2.createTrackbar("Trackbar", "WindowName", 0, len(images)-1, image_processing)
# Loop for get trackbar pos and process it
prev = 0
while True:
# Get position in trackbar
    TrackbarPos = cv2.getTrackbarPos("Trackbar", "WindowName")
    if TrackbarPos!=prev:
        image_processing(images[TrackbarPos])
    ch = cv2.waitKey(5)
    if ch == 27:
            break
    prev = TrackbarPos
cv2.destroyAllWindows()
