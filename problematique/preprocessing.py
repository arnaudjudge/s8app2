import numpy as np
import cv2


def edge_coefficient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Using the Canny filter with different parameters
    edges_high_thresh = cv2.Canny(gray, 60, 120)
    # cv2.imshow('Frames', edges_high_thresh)
    # cv2.waitKey()
    return edges_high_thresh.mean()


def leaf_color_coef(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # find the yellow and green color in the leaf
    mask_yellow_green = cv2.inRange(hsv, (40, 30, 30), (180, 255, 255))
    # find any of the three colors(green or brown or yellow) in the image
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask_yellow_green)
    return np.count_nonzero(res)/res.flatten().size


# def excess_green(image):
#     r = np.sum(image[:, :, 0])
#     g = np.sum(image[:, :, 1])
#     b = np.sum(image[:, :, 2])
#
#     return g / (r + g + b)

def normalize(image):
    total = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2])
    total = np.where(total == 0, 1, total)
    return image[:, :, 0] / total, image[:, :, 1] / total, image[:, :, 2] / total


def excess_green_index(image):
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]#normalize(image)
    r = np.sum(red)
    g = np.sum(green)
    b = np.sum(blue)
    return g / (r + b + g)


def excess_blue_index(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r = np.sum(r)
    g = np.sum(g)
    b = np.sum(b)
    return b / (r + g + b)


def excess_red_index(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]#normalize(image)
    r = np.sum(r)
    g = np.sum(g)
    b = np.sum(b)
    return 1.4 * r - b


# def excess_greenred_index(image):
#     image = normalize(image)
#     r = np.sum(image[:, :, 0])
#     g = np.sum(image[:, :, 1])
#     b = np.sum(image[:, :, 2])
#     return (3 * g) - ((2.4 * r) - b)