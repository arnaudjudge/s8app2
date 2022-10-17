import cv2
import matplotlib.pyplot as plt
from skimage import color as skic
from skimage import io as skiio
import numpy as np


def edge_detection(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Using the Canny filter to get contours
    edges = cv2.Canny(gray, 20, 30)
    # Using the Canny filter with different parameters
    edges_high_thresh = cv2.Canny(gray, 60, 120)
    # Stacking the images to print them together
    # For comparison
    images = np.hstack((gray, edges_high_thresh))

    # Display the resulting frame
    cv2.imshow('Frames', images)
    cv2.waitKey()

    print(edges_high_thresh.mean())


def detect_leaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # find the brown color
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow and green color in the leaf
    mask_yellow_green = cv2.inRange(hsv, (40, 30, 30), (180, 255, 255))
    # find any of the three colors(green or brown or yellow) in the image
    #mask = cv2.bitwise_or(mask_yellow_green, mask_brown)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask_yellow_green)
    return hsv, res

if __name__ == "__main__":
    frame = skiio.imread("./baseDeDonneesImages/street_urb983.jpg")
    edge_detection(frame)
    hsv, r = detect_leaf(frame)
    images = np.hstack((frame, hsv, r))

    cv2.imshow('f', images)
    cv2.waitKey()
    print(np.count_nonzero(r)/(r.flatten().size))

