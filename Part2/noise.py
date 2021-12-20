import random
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# Read image
image = cv2.imread('Subaru555.jpg')

# Convert from BGR to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.imshow('Gray Image', gray_image)
cv2.imwrite('GrayScaleImg.jpg', gray_image)

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

# Show Salt and Pepper noisy image, save in same directory
cv2.namedWindow('Salt and Pepper',cv2.WINDOW_NORMAL)
cv2.imshow('Salt and Pepper', spnoisy_img)
cv2.imwrite('SaltAndPepper.jpg', spnoisy_img)

# Create noise using numpy.random poisson function, type uint8 so pixels get 0-255 values
noise = np.random.poisson(gray_image).astype(np.uint8)
# Add noise to initial grayscale image
poisson_img = (gray_image + noise)

# Show image with Shot noise, save in same directory
cv2.namedWindow('Shot Noise', cv2.WINDOW_NORMAL)
cv2.imshow('Shot Noise', poisson_img)
cv2.imwrite('ShotNoise.jpg', poisson_img)

# Use mean filter on images, kernel size 5x5
meanFilterSP = cv2.blur(spnoisy_img, (5, 5))
meanFilterPoisson = cv2.blur(poisson_img, (5, 5))

# Use median filter on images, kernel size 5x5
medianFilterSP = cv2.medianBlur(spnoisy_img, 5)
medianFilterPoisson = cv2.medianBlur(poisson_img, 5)

# Use gaussian filter on images, kernel size 5x5
gaussianFilterSP = cv2.GaussianBlur(spnoisy_img, (5, 5), 0, 0)
gaussianFilterPoisson = cv2.GaussianBlur(poisson_img, (5, 5), 0, 0)

# Show image after mean filter, save in same directory
cv2.namedWindow('Mean Filter (Salt & Pepper)', cv2.WINDOW_NORMAL)
cv2.imshow('Mean Filter (Salt & Pepper)', meanFilterSP)
cv2.imwrite('MeanFilteredSP.jpg', meanFilterSP)
cv2.namedWindow('Mean Filter (Shot)', cv2.WINDOW_NORMAL)
cv2.imshow('Mean Filter (Shot)', meanFilterPoisson)
cv2.imwrite('MeanFilteredPoisson.jpg', meanFilterPoisson)

# Show image after median filter, save in same directory
cv2.namedWindow('Median Filter (Salt & Pepper)', cv2.WINDOW_NORMAL)
cv2.imshow('Median Filter (Salt & Pepper)', medianFilterSP)
cv2.imwrite('MedianFilteredSP.jpg', medianFilterSP)
cv2.namedWindow('Median Filter (Shot)', cv2.WINDOW_NORMAL)
cv2.imshow('Median Filter (Shot)', medianFilterPoisson)
cv2.imwrite('MedianFilteredPoisson.jpg', medianFilterPoisson)

# Show image after gaussian filter, save in same directory
cv2.namedWindow('Gaussian Filter (Salt & Pepper)', cv2.WINDOW_NORMAL)
cv2.imshow('Gaussian Filter (Salt & Pepper)', gaussianFilterSP)
cv2.imwrite('GaussianFilteredSP.jpg', gaussianFilterSP)
cv2.namedWindow('Gaussian Filter (Shot)', cv2.WINDOW_NORMAL)
cv2.imshow('Gaussian Filter (Shot)', gaussianFilterPoisson)
cv2.imwrite('GaussianFilteredPoisson.jpg', gaussianFilterPoisson)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Print Similarity Scores
sim_score = ssim(gray_image, meanFilterSP)
mse_score = mean_squared_error(gray_image, meanFilterSP)
print('Salt & Pepper: Original - Mean Filtered SSID:{:.3f}'.format(sim_score))
print('Salt & Pepper: Original - Mean Filtered MSE:{:.3f}'.format(mse_score))
sim_score = ssim(gray_image, medianFilterSP)
mse_score = mean_squared_error(gray_image, medianFilterSP)
print('Salt & Pepper: Original - Median Filtered SSID:{:.3f}'.format(sim_score))
print('Salt & Pepper: Original - Median Filtered MSE:{:.3f}'.format(mse_score))
sim_score = ssim(gray_image, gaussianFilterSP)
mse_score = mean_squared_error(gray_image, gaussianFilterSP)
print('Salt & Pepper: Original - Gauss Filtered SSID:{:.3f}'.format(sim_score))
print('Salt & Pepper: Original - Gauss Filtered MSE:{:.3f}'.format(mse_score))

sim_score = ssim(gray_image, meanFilterPoisson)
mse_score = mean_squared_error(gray_image, meanFilterPoisson)
print('Shot: Original - Mean Filtered SSID:{:.3f}'.format(sim_score))
print('Shot: Original - Mean Filtered MSE:{:.3f}'.format(mse_score))
sim_score = ssim(gray_image, medianFilterPoisson)
mse_score = mean_squared_error(gray_image, medianFilterPoisson)
print('Shot: Original - Median Filtered SSID:{:.3f}'.format(sim_score))
print('Shot: Original - Median Filtered MSE:{:.3f}'.format(mse_score))
sim_score = ssim(gray_image, gaussianFilterPoisson)
mse_score = mean_squared_error(gray_image, gaussianFilterPoisson)
print('Shot: Original - Gauss Filtered SSID:{:.3f}'.format(sim_score))
print('Shot: Original - Gauss Filtered MSE:{:.3f}'.format(mse_score))
