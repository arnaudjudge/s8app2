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
    data = []
    for c in IC.images:
        dims = np.zeros((len(c), 6), dtype=float)
        for i in range(len(c)):
            image = c[i]
            dims[i, 0] = excess_green_index(image)
            dims[i, 1] = h_dominant_gradient(image)  # good, split on coast
            dims[i, 2] = leaf_color_coef(image)
            dims[i, 3] = excess_blue_index(image)  # mid separation
            dims[i, 4] = edge_coefficient(image)  # very good
            dims[i, 5] = very_grey(image, limit=15)  # ok split on street
            # dims[i, 6] = excess_red_index(image)  # ok split on forest
        data.append(dims)
    dims = np.vstack((data[0], data[1], data[2]))

    #an.view3D(dims, IC.targets, 'dims 1 2 3')
    # import matplotlib
    # colors = ['red', 'green', 'blue']
    # fig = plt.figure(figsize=(8, 8))
    # plt.scatter(dims[:, 5], dims[:, 1], c=IC.targets, cmap=matplotlib.colors.ListedColormap(colors))
    #
    # cb = plt.colorbar()
    # loc = np.arange(0, max(IC.targets), max(IC.targets) / float(len(colors)))
    # cb.set_ticks(loc)
    # cb.set_ticklabels([0, 1, 2])
    # plt.show()

    test = []
    ndonnees = 5000
    for dim in range(dims.shape[1]):
        d = dims[:, dim]
        test.append((np.max(d) - np.min(d)) * np.random.random(ndonnees) + np.min(d))
    test = np.transpose(np.array(test))

    from helpers import classifiers

    classifiers.full_Bayes_risk(data, IC.targets, test, 'Bayes risque #1',
                                an.Extent(ptList=dims), dims, IC.targets)


    # classifiers.full_ppv(1, dims, IC.targets, dims,
    #                      '1-PPV avec données orig comme représentants', an.Extent(ptList=dims), dims, IC.targets)

    cluster_centers, cluster_labels = classifiers.full_kmean(10, data, IC.targets,
                                                             'Représentants des 1-moy', an.Extent(ptList=dims))
    classifiers.full_ppv(7, cluster_centers, cluster_labels, test, '1-PPV sur le 1-moy', an.Extent(ptList=dims),
                         dims, IC.targets)


    n_hidden_layers = 5
    n_neurons = 10
    classifiers.full_nn(n_hidden_layers, n_neurons, dims, IC.targets, test,
                        f'NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche',
                        an.Extent(ptList=dims), dims, IC.targets)
    plt.show()


######################################
if __name__ == '__main__':
    main()
