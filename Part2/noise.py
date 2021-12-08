import cv2
import numpy as np




# Read image
image = cv2.imread('VilleneuveMonaco1979.jpg')

# Convert from BGR to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
