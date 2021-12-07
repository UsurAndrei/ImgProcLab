import cv2
import numpy as np


# Solarize function
def solarize(originalImage, thresValue):
    # Create solarized image, same resolution as original, all pixels value 0 of type uint8 (0-255)
    solarized_image = np.zeros(originalImage.shape, np.uint8)
    # shape returns a tuple of the number of rows, columns, and channels (if the image is color)
    # Select each pixel of the original image
    for colIdx in range(originalImage.shape[0]):
        for rowIdx in range(originalImage.shape[1]):
            # Check how the value of each pixel compares to our threshold value
            if originalImage[colIdx][rowIdx] >= thresValue:
                solarized_image[colIdx][rowIdx] = originalImage[colIdx][rowIdx]
            elif originalImage[colIdx][rowIdx] < thresValue:
                solarized_image[colIdx][rowIdx] = 255 - originalImage[colIdx][rowIdx]
    return solarized_image


# Read image
image = cv2.imread('VilleneuveMonaco1979.jpg')

# Convert from BGR to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show original and  gray scale image
cv2.namedWindow('Orig',cv2.WINDOW_NORMAL)
cv2.imshow('Orig', image)
cv2.namedWindow('Gray',cv2.WINDOW_NORMAL)
cv2.imshow('Gray', gray_image)

# Wait for any key, destroy all windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Solarize using thresValue = 64
solarized_img64 = solarize(gray_image, 64)

# Solarize using thresValue = 128
solarized_img128 = solarize(gray_image, 128)

# Solarize using thresValue = 192
solarized_img192 = solarize(gray_image, 192)

# Show solarized images
cv2.namedWindow('Solarized image [64]',cv2.WINDOW_NORMAL)
cv2.imshow('Solarized image [64]', solarized_img64)
cv2.namedWindow('Solarized image [128]',cv2.WINDOW_NORMAL)
cv2.imshow('Solarized image [128]', solarized_img128)
cv2.namedWindow('Solarized image [192]',cv2.WINDOW_NORMAL)
cv2.imshow('Solarized image [192]', solarized_img192)

# Wait for any key, destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
