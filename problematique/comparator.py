import numpy as np
from skimage import color as skic
from skimage import io as skiio
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

    def run_coefficients(self):
        for name, method in self.methods.items():
            coeffs = self.coefficients(method_to_run=method)
            average_coast = np.average(coeffs["coast"])
            average_forest = np.average(coeffs["forest"])
            average_street = np.average(coeffs["street"])
            std_coast = np.std(coeffs["coast"])
            std_forest = np.std(coeffs["forest"])
            std_street = np.std(coeffs["street"])
            print(
                f"METHOD: {name}\n"
                f"COAST:\t{round(average_coast, 3)} ± {round(std_coast, 3)}\n"
                f"FOREST:\t{round(average_forest, 3)} ± {round(std_forest, 3)}\n"
                f"STREET:\t{round(average_street, 3)} ± {round(std_street, 3)}\n")


if __name__ == "__main__":
    image_coll = CustomImageCollection(
        image_folder=Path("./baseDeDonneesImages"), methods={"Excess green": excess_green_index})
    image_coll.run_coefficients()