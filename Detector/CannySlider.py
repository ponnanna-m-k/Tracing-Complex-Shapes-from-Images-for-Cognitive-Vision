# script for tuning Parameters
import cv2

# reads the image
img = cv2.imread("Test Images/eyewear2.jpg")

# empty callback function for creating tracker
def callback(foo):
    pass


# create windows and trackbar
cv2.namedWindow('Parameters', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('Threshold1', 'Parameters', 0, 255, callback)  # Can be changed
cv2.createTrackbar('Threshold2', 'Parameters', 0, 255, callback)  # Can be changed

while (True):
    # get Threshold value from trackbar
    th1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
    th2 = cv2.getTrackbarPos('Threshold2', 'Parameters')

    edge = cv2.Canny(img, th1, th2)
    cv2.imshow('Original Image', img)
    cv2.imshow('Canny', edge)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()