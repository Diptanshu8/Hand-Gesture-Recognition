import cv2
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
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

def compute_centroid(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cont_area = [cv2.contourArea(item) for item in contours]
    cnt = contours[cont_area.index(max(cont_area))]
    M = cv2.moments(cnt)
    centroid = (M['m10']/M['m00'],M['m01']/M['m00'])
    return centroid

def generate_boundary_pixels(img):
    print "Generating the hand boundary pixels"
    rows,cols = img.shape
    row,col = 0,0
    boundary_pixels = []
    while row<rows:
        col=0
        while col<cols:
            if img[row][col]==255:
                boundary_pixels.append((col,row))
            col+=1
        row+=1
    print "Finished generating the hand boundary pixels"
    print "number of boundary pixels are "+str(len(boundary_pixels))
    return boundary_pixels

def vertical_image_peak_detection(boundary_pixels):
    mask_size = 7
    offset = int(7/2)
    i,peaks = offset,[]
    while i<len(boundary_pixels)-offset:
        mask = [boundary_pixels[i+item][1] for item in xrange(-(offset),offset+1)]
        if mask[0]-mask[1]>=0 and mask[2]-mask[3]>=0 and mask[1]-mask[2]>=0 and mask[2]-mask[3]>=0 and mask[3]-mask[4]<=0 and mask[4]-mask[5]<=0 and mask[5]-mask[6]<=0:
            peaks.append(boundary_pixels[i])
        i+=1
    return peaks

def horizontal_image_peak_detection(img):
    pass

def eucledian_distance(peak,centroid):
    return pow(((peak[0]-centroid[0])**2)+(peak[1]-centroid[1])**2,0.5)
def hand_pixel_count(img):
    rows,cols = img.shape
    count = 0
    row = 0
    while row<rows:
        col = 0
        while col<cols:
            if img[row,col]==255:
                count+=1
            col+=1
        row+=1
    return count

def vertical_thumb_detection(img):
    hand_pixels = hand_pixel_count(img)
    rows,cols = img.shape
    mid_col = cols/2
    row,col=0,0
    while row<rows:
        mask_left = [img[row,(mid_col/2)+item] for item in xrange(-15,15)]
        mask_right = [img[row,((3*mid_col)/2)+item] for item in xrange(-15,15)]
        if sum(mask_right)<0.00069*hand_pixels or sum(mask_left)<0.00069*hand_pixels:
            print "Thumb present"
            break
        row+=1


def image_processing(image_addr):
    img = image_loading(image_addr)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,img_binary_otsu= cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    img_eroded = cv2.erode(img_binary_otsu,kernel,iterations = 1)
    closing = cv2.morphologyEx(img_eroded, cv2.MORPH_OPEN,  kernel)
    hand_boundary=generating_boundary_of_hand(closing)
    vertical_ratio = vertical_identification(hand_boundary)
    #horizontal_ratio = horizontal_identification(hand_boundary)
    hand_boundary_pixels = generate_boundary_pixels(hand_boundary)
    centroid = compute_centroid(hand_boundary)

        #cv2.drawContours(img, contours, item, (0,255,0), 3)
        #showing_image(img,"{} contour image".format(item))
    #if vertical_ratio>horizontal_ratio:
    #    horizontal_image_peak_detection(hand_boundary_pixels)
    #elif vertical_ratio<horizontal_ratio:
    #    pass
    peaks = vertical_image_peak_detection(hand_boundary_pixels)
    distances_peak = [(peak,eucledian_distance(peak,centroid)) for peak in peaks]
    distances = [item[1] for item in distances_peak]
    max_distance = max(distances)
    threshold_distance = 0.75*max_distance
    #print centroid
    #print threshold_distance,max_distance
    tip_distances = [item for item in distances_peak if item[1]>threshold_distance ]
    #print len(tip_distances)
    #plt.plot([item[0][0] for item in tip_distances],[item[0][1] for item in tip_distances],'bo')
    peaks = [item[0] for item in tip_distances]
    y_peaks = [item[1] for item in peaks]
    print peaks
    plotter = []
    for item in peaks:
        plotter.append(item)
        plotter.append(centroid)
    top = max(y_peaks)
    t = 0.75*(top)
    plt.axhline(top,0,1)
    plt.axhline(t,0,1)
    plt.plot(plotter)
    plt.show()

    #vertical_thumb_detection(closing)

    #showing_image(img_binary_otsu,"Binary image")
    #showing_image(img_binary_otsu,"Binary image")
    #showing_image(img_eroded,"eroded image")
    #showing_image(closing,"temp image")
    #showing_image(img,"Original image")

images = ["Satish1F.jpg","Satish2F.jpg","Satish3F.jpg","Satish4F.jpg","Satish5F.jpg"]
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
