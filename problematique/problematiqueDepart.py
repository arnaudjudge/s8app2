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
from helpers import classifiers
from helpers.visualisation import view_dimension_histograms

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
            dims[i, 4] = excess_green_index(image)
            dims[i, 1] = h_dominant_gradient(image)  # good, split on coast
            dims[i, 2] = leaf_color_coef(image)
            # dims[i, 3] = excess_blue_index(image)  # mid separation
            dims[i, 0] = edge_coefficient(image)  # very good
            dims[i, 5] = very_grey(image, limit=15)  # ok split on street
            dims[i, 3] = excess_red_index(image)  # ok split on forest
        data.append(dims)
    dims = np.vstack((data[0], data[1], data[2]))
    lens = (len(data[0]), len(data[1]), len(data[2]))

    # Normalize data and replace in array
    dims, _ = an.scaleData(dims)
    data = [dims[:lens[0]], dims[lens[0]:lens[1]+lens[0]], dims[lens[1]+lens[0]:]]

    # split en train-test
    min_len = min(len(data[0]), len(data[1]), len(data[2]))
    idx = np.arange(min_len)[:int(min_len*0.8)]
    train_data = []
    test_data = []
    train_targets = []
    test_targets = []
    for i in range(len(data)):
        mask = np.zeros(len(data[i]), dtype=bool)
        mask[idx,] = True

        train_data.append(data[i][mask])
        train_targets.append(IC.targets[i][mask])
        test_data.append(data[i][~mask])
        test_targets.append(IC.targets[i][~mask])

    # format acceptable pour classifieurs
    test_data = np.concatenate((test_data[0], test_data[1], test_data[2]), axis=0)
    train_targets = np.concatenate((train_targets[0], train_targets[1], train_targets[2]), axis=0)
    test_targets = np.concatenate((test_targets[0], test_targets[1], test_targets[2]), axis=0)

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


    # an.calcModeleGaussien(d, "Allo")
    # corr = np.corrcoef(d.T, rowvar=True)

    test = []
    ndonnees = 5000
    for dim in range(train_data[0].shape[1]):
        d = train_data[0][:, dim]
        test.append((1 - (-1)) * np.random.random(ndonnees) + (-1))
    test = np.transpose(np.array(test))

    view_dimension_histograms(train_data)

    exectute = ['bayes', 'kppv', 'nn']
    if 'bayes' in exectute:
        classifiers.full_Bayes_risk(train_data, train_targets, test, 'Bayes risque #1',
                                    an.Extent(ptList=dims), test_data, test_targets)

    if 'kppv' in exectute:
        cluster_centers, cluster_labels = classifiers.full_kmean(9, train_data, train_targets,
                                                                 'Représentants des 1-moy', an.Extent(ptList=dims))
        classifiers.full_ppv(1, cluster_centers, cluster_labels, test, '1-PPV sur le 1-moy', an.Extent(ptList=dims),
                             test_data, test_targets)

    if 'nn' in exectute:
        n_hidden_layers = 5
        n_neurons = 10
        classifiers.full_nn(n_hidden_layers, n_neurons, np.vstack((train_data[0], train_data[1], train_data[2])), train_targets, test,
                            f'NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche',
                            an.Extent(ptList=dims), test_data, test_targets)
    plt.show()


######################################
if __name__ == '__main__':
    main()
