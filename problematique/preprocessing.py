import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import hog


def edge_coefficient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Using the Canny filter with different parameters
    edges_high_thresh = cv2.Canny(gray, 60, 120)
    # cv2.imshow('Frames', edges_high_thresh)
    # cv2.waitKey()
    return edges_high_thresh.mean()


def warm_cold_color_ratio(image):
    # TODO: ocean (présente de couleurs très chaudes dans la partie supérieure:
    #  en divisant verticalement l'image, la saturation des pixels de la moitié
    #  supérieure est au moins 30% plus basse et l'intensité lumineuse est 25% plus
    #  élevée que la moitié inférieure seulement si l'image est un océan
    pass


def asphalt_pixels(image):
    # TODO: ville (présence de tons de gris dans le tier inférieur:
    #  présence d’un grand nombre de pixels à basse saturation et valeur
    #  variante dans le tier le plus bas dans le domaine HSV)
    pass


def hog_factor(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return np.sum(hog_image)


def top_down_luminosity_ratio(image):
    shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    top = image[:int(shape[0] / 2), :, :]
    bottom = image[int(shape[0] / 2):, :, :]
    return np.average(top[:, :, 2]) / np.average(bottom[:, :, 2])


def left_right_color_difference_ratio(image):
    shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    left = image[:, :int(shape[0] / 2), :]
    right = image[:, int(shape[0] / 2):, :]
    return max(np.average(left[:, :, 0]), np.average(right[:, :, 0])) / min(np.average(left[:, :, 0]),
                                                                            np.average(right[:, :, 0]))


def h_dominant_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dx = np.gradient(gray, axis=0)
    dy = np.gradient(gray, axis=1)
    # ims = np.hstack((dx, dy))
    # cv2.imshow('D', ims)
    # cv2.waitKey()
    return abs(dx).sum() / abs(dy).sum()

def very_grey(image, limit=15):
    # Grey is where r, g and b are all equal, this functions gives a little wiggle room
    # make this with numpy plz
    # t = 0
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         r = range(image[i, j, 0] - 20, image[i, j, 0] + 20)
    #         if image[i, j, 1] in r and image[i, j, 2] in r:
    #             t += 1
    # return t / (image.shape[0] * image.shape[1])

    d1 = np.logical_and(np.where(image[:, :, 1] >= (image[:, :, 0] - limit), True, False),
                        np.where(image[:, :, 1] <= (image[:, :, 0] + limit), True, False))

    d2 = np.logical_and(np.where(image[:, :, 2] >= (image[:, :, 0] - limit), True, False),
                        np.where(image[:, :, 2] <= (image[:, :, 0] + limit), True, False))

    return np.logical_and(d1, d2).sum() / (image.shape[0] * image.shape[1])


def leaf_color_coef(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # find the yellow and green color in the leaf
    mask_yellow_green = cv2.inRange(hsv, (40, 30, 30), (180, 255, 255))
    # find any of the three colors(green or brown or yellow) in the image
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask_yellow_green)
    return np.count_nonzero(res) / res.flatten().size


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
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]  # normalize(image)
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
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]  # normalize(image)
    r = np.sum(r)
    g = np.sum(g)
    b = np.sum(b)
    return r / (r + g + b)

# def excess_greenred_index(image):
#     image = normalize(image)
#     r = np.sum(image[:, :, 0])
#     g = np.sum(image[:, :, 1])
#     b = np.sum(image[:, :, 2])
#     return (3 * g) - ((2.4 * r) - b)
