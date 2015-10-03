#!/usr/bin/python
# USAGE
# python transform_example.py --image images/example_01.png --outputfile='example_01.warped.png"
# python transform_example.py --image images/example_02.png --outputfile='example_02.warped.png"
# python transform_example.py --image images/example_03.png --outputfile='example_03.warped.png" 

# import the necessary packages
from pyimagesearch.transform import four_point_transform
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-o", "--outputfile")
args = vars(ap.parse_args())

# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
image = cv2.imread(args["image"])

MORPH = 9
CANNY = 84
HOUGH = 25

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.GaussianBlur(img, (3,3), 0, img)

# this is to recognize white on white
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
dilated = cv2.dilate(img, kernel)

edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)

lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
for line in lines[0]:
	cv2.line(edges, (line[0], line[1]), (line[2], line[3]), (255,0,0), 2, 8)

# finding contours
_, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

# simplify contours down to polygons
rects = []
for cont in contours:
    rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
    rects.append(rect)

pts = np.array(rects[0], dtype = "float32")

# apply the four point tranform to obtain a "birds eye view" of
# the image
warped = four_point_transform(image, pts)

# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)

cv2.imwrite(args["outputfile"], warped)
cv2.waitKey(0)
