from helper import kmeans, resize, showImage, huMonents
import cv2
import numpy as np


image = cv2.imread("0cos015.png")
kernel = np.ones((5,5), np.uint8)

img_erosion = cv2.erode(image, kernel, iterations=1)
img_dilation = cv2.dilate(image, kernel, iterations=1)

cv2.imshow("Erosion", img_erosion)
cv2.imshow("Dilasion", img_dilation)
cv2.imshow("Original", image)
cv2.waitKey(0)
# showImage(image)
