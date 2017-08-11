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
    rows,cols= img.shape
    #just a hack for now
    background = img[0][0]
    col = 0
    while col<cols:
        row = 0
        while row<rows:
            #temp = []
            #temp = [item for item in img[row][col] if item not in background]
            if img[row][col] != background:
                return col
            row+=1
        col+=1
    return -1

def right_side_of_hand(img):
    rows,cols = img.shape
    #just a hack for now
    background = img[0][0]
    col = cols-1
    while col>0:
        row = 0
        while row<rows:
            #temp = []
            #temp = [item for item in img[row][col] if item not in background]
            if img[row][col] != background:
                return col
            row+=1
        col-=1
    return -1

def top_of_hand(img):
    rows,cols = img.shape
    #just a hack for now
    background = img[0][0]
    row = 0
    while row<rows:
        col = 0
        while col<cols:
            #temp = []
            #temp = [item for item in img[row][col] if item not in background]
            if img[row][col] != background:
                return row
            col+=1
        row+=1
    return -1

def bottom_of_hand(img):
    rows,cols = img.shape
    #just a hack for now
    background = img[0][0]
    row = rows-1
    while row>0:
        col = 0
        while col<cols:
            #temp = []
            #temp = [item for item in img[row][col] if item not in background]
            if img[row][col] != background:
                return row
            col+=1
        row-=1
    return -1

def generating_boundary_of_hand(img):
    v = np.median(img)
    sigma = 0.10
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
        i+=offset
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
    left,right = left_side_of_hand(img),right_side_of_hand(img)
    top,bottom = top_of_hand(img),bottom_of_hand(img)
    right_sum,left_sum = 0,0
    rows,cols = img.shape
    row = top
    while row<rows:
        mask_left = [img[row,left+item]/255 for item in xrange(0,30)]
        mask_right = [img[row,right-item]/255 for item in xrange(0,30)]
        right_sum+=sum(mask_right)
        left_sum += sum(mask_left)
        row+=1
    if right_sum<hand_pixels*0.00069 or left_sum<hand_pixels*0.00069:
        print "thumb preset"
    else:
        print "thumb absent"

def image_processing(image_addr):
    img = image_loading(image_addr)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,img_binary_otsu= cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    #img_eroded = cv2.erode(img_binary_otsu,kernel,iterations = 1)
    closing = cv2.morphologyEx(img_binary_otsu, cv2.MORPH_OPEN,  kernel)
    hand_boundary=generating_boundary_of_hand(closing)
    hand_boundary_pixels = generate_boundary_pixels(hand_boundary)
    centroid = compute_centroid(img_gray)

    showing_image(img,"Original image")
    showing_image(closing,"closed image")
    showing_image(hand_boundary,"boundary image")
    peaks = vertical_image_peak_detection(hand_boundary_pixels)
#    vertical_thumb_detection(closing)
    distances_peak = [(peak,eucledian_distance(peak,centroid)) for peak in peaks]
    distance_gradient = []
    for i in xrange(len(distances_peak)-1):
        distance_gradient.append(distances_peak[i+1][1]-distances_peak[i][1])
    distance_gradient.append(0)
    plot_points = []
    for i in xrange(len(distance_gradient)):
        if (distance_gradient[i]<0 and distance_gradient[i+1]>0 ) or  (distance_gradient[i]>0 and distance_gradient[i+1]<0 ):
            plot_points.append(distances_peak[i-1])
    distancematrix = {}
    for item in plot_points:
        distancematrix[item] = []
        for i in plot_points:#plot_points[:plot_points.index(item)]+plot_points[plot_points.index(item)+1:]:
            distancematrix[item].append(eucledian_distance(i[0],item[0]))
    final_plot_points = []
    #threshold_distance = 100
    #threshold_distance = 150
    threshold_distance = 200
    for item in distancematrix.keys():
        for distance in distancematrix[item]:
            if distance > threshold_distance:
                final_plot_points.append(plot_points[distancematrix[item].index(distance)][0])
    temp =final_plot_points
    final_plot_points = []
    final_plot_points = [item for item in temp if item not in final_plot_points]
    c = sorted(final_plot_points)
    x,y = zip(*c)
    rows,cols = hand_boundary.shape
    y = [rows - item for item in y]
    top = max(y)
    t = 0.75*(top)
    plt.axhline(top,0,1)
    plt.axhline(t,0,1)
    plt.plot(x,y,'-o')
    plt.show()

images= []
for item in os.listdir(os.getcwd()):
    if ".jpg" in item:
        images.append(item)
print images
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
