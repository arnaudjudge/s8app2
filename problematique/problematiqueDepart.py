"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from preprocessing import *
from ImageCollection import ImageCollection
import helpers.analysis as an

#######################################
def main():
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E3 et problématique
    # N = 6
    # im_list = np.sort(random.sample(range(np.size(ImageCollection.image_list, 0)), N))
    # print(im_list)
    # ImageCollection.images_display(im_list)
    # ImageCollection.view_histogrammes(im_list)
    # plt.show()
    IC = ImageCollection
    dims = np.zeros((len(IC.images), 7), dtype=float)
    for i in range(len(IC.images)):
        image = IC.images[i]
        dims[i, 0] = excess_green_index(image)  # ok split on forest
        dims[i, 1] = edge_coefficient(image)  # very good
        dims[i, 2] = leaf_color_coef(image)
        dims[i, 3] = excess_blue_index(image)  # mid separation
        dims[i, 4] = excess_red_index(image)
        dims[i, 5] = h_dominant_gradient(image)  # good, split on coast
        dims[i, 6] = very_grey(image, limit=15)  # ok split on street

    #an.view3D(dims, IC.targets, 'dims 1 2 3')
    import matplotlib
    colors = ['red', 'green', 'blue']
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(dims[:, 5], dims[:, 1], c=IC.targets, cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    loc = np.arange(0, max(IC.targets), max(IC.targets) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels([0, 1, 2])
    plt.show()

    print("allo")


######################################
if __name__ == '__main__':
    main()
