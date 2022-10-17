import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    data1 = np.array([[0, 1], [0, -1]]).T
    data2 = np.array([[1, 0], [-1, 0]]).T

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(data1[0], data1[1], c='b')
    ax1.scatter(data2[0], data2[1], c='r')
    plt.show()

    # A = np.array([[0.1, 10], [0.1, 10]]) # toujours pas possible de separer
    # d1t= np.matmul(A, data1)
    # d2t = np.matmul(A, data2)

    d1t = -1 / (1 + np.exp(np.abs(data1[0]) / 2)), 1 / (1 + np.exp(-data1[1] / 2))

    d2t = -1 / (1 + np.exp(np.abs(data2[0]) / 2)), 1 / (1 + np.exp(-data2[1] / 2))
    print(d1t)
    print(d2t)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(d1t[0], d1t[1], c='b')
    ax1.scatter(d2t[0], d2t[1], c='r')
    plt.show()