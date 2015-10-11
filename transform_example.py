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
def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew

image = cv2.imread(args["image"])

ratio = image.shape[0] / 500.0
if ratio > 1:
	orig = image.copy()
	image = imutils.resize(image, height = 500)

#preprocessing
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),2 )
kernel = np.ones((11,11),'uint8')
# dilated = cv2.dilate(gray,kernel, iterations = 2)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
ret,thresh = cv2.threshold(opening,127,255,0)
edges = cv2.Canny(opening, 150, 250, apertureSize=3)

_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = filter(lambda cont: cv2.arcLength(cont, False), contours)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
approximated_contours = []
for c in contours:
	area = cv2.contourArea(c)
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True).reshape(-1, 2)
	approximated_contours.append(approx)

approximated_contours = sorted(approximated_contours, key=cv2.contourArea,reverse=True)[:1]
warped = four_point_transform(image, approximated_contours[0].reshape(4, 2))
image = cv2.drawContours(image, approximated_contours, -1, (255, 0, 0), 2)

cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)