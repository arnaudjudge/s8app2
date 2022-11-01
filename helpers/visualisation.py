from matplotlib import pyplot as plt

def view_dimension_histograms(data):
    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    for idx, c in enumerate(data):
        for i in range(c.shape[1]):
            axs[idx, i].hist(c[:, i], bins='auto')
            axs[idx, i].set_title(f'C {idx}, dim {i}')
            axs[idx, i].set_xticks([])
            axs[idx, i].set_yticks([])
    plt.show()
