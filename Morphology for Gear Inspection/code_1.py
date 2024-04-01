import cv2
import numpy as np
from matplotlib import pyplot as plt


original_image = cv2.imread('D:/66_2/652 comvision/6408_Case Study#1/case1.jpg', cv2.IMREAD_GRAYSCALE) #Load image


contours, _ = cv2.findContours(original_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Find contours in the image
filled_image = np.zeros_like(original_image) #Create an empty image to fill the contours
cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED) #Draw filled contours on the empty image


median_filtered_image = cv2.medianBlur(filled_image, 5)  #Reduce Noise


mask = np.zeros_like(original_image, dtype=np.uint8) #Create a blank mask
cv2.circle(mask, (266, 284),195, 255, -1)  #Draw circle on mask
cv2.circle(mask, (669, 271),194, 255, -1)  #Draw circle on mask
result_image = cv2.bitwise_and(median_filtered_image, median_filtered_image, mask=255-mask) #⨁ Hole mask


radius = 10 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
dilated_image = cv2.dilate(result_image, kernel, iterations=1) #dilation with the circular kernel


holering = np.zeros_like(original_image, dtype=np.uint8) #Create a blank mask
cv2.circle(holering, (266, 284), 200, 255, thickness=3) #Draw circle ring on mask
cv2.circle(holering, (669, 271), 200, 255, thickness=3) #Draw circle ring on mask


defects_image = holering - dilated_image


# Display the original and filled images
plt.subplot(231), plt.imshow(original_image, cmap='gray'), plt.title('Original Image')
plt.subplot(232), plt.imshow(filled_image, cmap='gray'), plt.title('Filled Contours')
plt.subplot(233), plt.imshow(median_filtered_image, cmap='gray'), plt.title('Median Filtered')
plt.subplot(234), plt.imshow(result_image, cmap='gray'), plt.title('⨁ Hole mask')
plt.subplot(235), plt.imshow(dilated_image, cmap='gray'), plt.title('Dilated')
plt.subplot(236), plt.imshow(defects_image, cmap='gray'), plt.title('Defected Image')
plt.show()
