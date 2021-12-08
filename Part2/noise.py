import random
import cv2
import numpy as np

# Read image
image = cv2.imread('Subaru555.jpg')

# Convert from BGR to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Salt and Pepper with probability 10% (10% white, 10% black)
spnoisy_img = np.zeros(gray_image.shape, np.uint8)

# Select each pixel of the original image
for colIdx in range(gray_image.shape[0]):
    for rowIdx in range(gray_image.shape[1]):
        # Generate a random number in the semi-open range [0.0 1.0)
        rand = random.random()
        # If the number is less than 0.1 (10% possibility), turn the pixel black 0
        if rand < 0.1:
            spnoisy_img[colIdx][rowIdx] = 0
        # Else if the number is greater than 0.9 (10% possibility), turn the pixel white 255
        elif rand > 0.9:
            spnoisy_img[colIdx][rowIdx] = 255
        # Else just keep the original value
        else:
            spnoisy_img[colIdx][rowIdx] = gray_image[colIdx][rowIdx]

cv2.namedWindow('Salt and Pepper',cv2.WINDOW_NORMAL)
cv2.imshow('Salt and Pepper', spnoisy_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

