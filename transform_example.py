#!/usr/bin/python
# USAGE
# python transform_example.py --image images/example_01.png --outputfile='example_01.warped.png"
# python transform_example.py --image images/example_02.png --outputfile='example_02.warped.png"
# python transform_example.py --image images/example_03.png --outputfile='example_03.warped.png" 

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
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

ratio = image.shape[0] / 500.0
if ratio > 1:
	orig = image.copy()
	image = imutils.resize(image, height = 500)

MORPH = 9
CANNY = 84
HOUGH = 25

img = cv2.GaussianBlur(image, (5,5), 0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros((gray.shape),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
# div = np.float32(gray)/(close)
res = np.uint8(cv2.normalize(gray,gray,0,255,cv2.NORM_MINMAX))
res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
edges = cv2.Canny(res, 15, 25)

_, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# loop over the contours
max_area = 0
min_area = 10000
for c in contours:
	# approximate the contour
	area = cv2.contourArea(c)
	if area > min_area:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if area > max_area:
			rect = approx
			max_area = area

# cv2.drawContours(image, [rect], -1, (0, 255, 0), 3)
# cv2.drawContours(image, [rect],-1,(0,255,0),-1)

# apply the four point tranform to obtain a "birds eye view" of
# the image
# warped = four_point_transform(image, rect.reshape(4, 2))

warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 300, offset = 10)
warped = warped.astype("uint8") * 255

# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)

cv2.imwrite(args["outputfile"], warped)
cv2.waitKey(0)
