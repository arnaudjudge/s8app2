import matplotlib.pyplot as plt
import numpy as np
from skimage import color as skic
from skimage import io as skiio
from scipy.stats import norm
from pathlib import Path
import cv2
from preprocessing import *


class CustomImageCollection:
    def __init__(self, image_folder: Path, methods=None):
        if methods is None:
            methods = {}
        self.methods = methods
        self.image_folder = image_folder
        self.image_list = self.image_folder.rglob("*.jpg")

    def coefficients(self, method_to_run):
        coeffs = {"coast": [], "forest": [], "street": []}
        for im_path in self.image_list:
            im = cv2.imread(im_path.__str__())
            coeff = method_to_run(im)
            if "coast" in im_path.__str__():
                coeffs["coast"].append(coeff)
            elif "forest" in im_path.__str__():
                coeffs["forest"].append(coeff)
            elif "street" in im_path.__str__():
                coeffs["street"].append(coeff)
        return coeffs

    def run_coefficients(self, visual: bool = False):
        for name, method in self.methods.items():
            coeffs = self.coefficients(method_to_run=method)
            average_coast = np.average(coeffs["coast"])
            average_forest = np.average(coeffs["forest"])
            average_street = np.average(coeffs["street"])
            std_coast = np.std(coeffs["coast"]).__float__()
            std_forest = np.std(coeffs["forest"]).__float__()
            std_street = np.std(coeffs["street"]).__float__()
            if visual:
                minimum = min(average_coast - (3.5 * std_coast),
                              average_forest - (3.5 * std_forest),
                              average_street - (3.5 * std_street))
                maximum = max(average_coast + (3.5 * std_coast),
                              average_forest + (3.5 * std_forest),
                              average_street + (3.5 * std_street))
                x = np.arange(minimum, maximum, 0.001)
                plt.title(f'Distributions for {name}')
                plt.plot(x, norm.pdf(x, average_coast, std_coast), color='blue',
                         label=f'coast: μ: {round(average_coast, 3)}, σ: {round(std_coast, 3)}')
                plt.plot(x, norm.pdf(x, average_forest, std_forest), color='green',
                         label=f'forest: μ: {round(average_forest, 3)}, σ: {round(std_forest, 3)}')
                plt.plot(x, norm.pdf(x, average_street, std_street), color='red',
                         label=f'street: μ: {round(average_street, 3)}, σ: {round(std_street, 3)}')
                plt.legend(title='Parameters')
                plt.show()
            print(
                f"METHOD: {name}\n"
                f"COAST:\t{round(average_coast, 3)} ± {round(std_coast, 3)}\n"
                f"FOREST:\t{round(average_forest, 3)} ± {round(std_forest, 3)}\n"
                f"STREET:\t{round(average_street, 3)} ± {round(std_street, 3)}\n")


if __name__ == "__main__":
    image_coll = CustomImageCollection(image_folder=Path("./baseDeDonneesImages"),
                                       methods={"top down": hog_factor})
    image_coll.run_coefficients(visual=True)
