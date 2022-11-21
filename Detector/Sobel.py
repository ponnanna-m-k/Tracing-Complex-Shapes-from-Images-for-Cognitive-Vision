# Python program to Edge detection
# using OpenCV in Python
# using Sobel edge detection
# and Laplacian method
import cv2

# Take each frame
img = cv2.imread("Test Images/eyewear2.jpg")

# Convert to HSV for simpler calculations
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Calcution of Sobelx
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

# Calculation of Sobely
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# Calculation of Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)

cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)
cv2.imshow('laplacian', laplacian)

k = cv2.waitKey(0) & 0xFF

cv2.destroyAllWindows()