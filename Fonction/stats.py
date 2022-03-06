import numpy as np
from typing import Callable, Union, List
from numbers import Number
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections import OrderedDict



def moyenne(x: np.ndarray):
    return np.mean(x).item()


def ecart_type(x: np.ndarray):
    return np.std(x).item()


def mediane(x: np.ndarray):
    return np.median(x).item()


def correlation(data: np.ndarray, correler_colonnes: bool = True):
    return np.corrcoef(data, rowvar=not correler_colonnes)


def covariance(data, covariance_colonnes: bool = True):
    return np.cov(data, rowvar=not covariance_colonnes)


#  échelle log
#  Kolmogorov–Smirnov test (général)
#  Relative entropy (distribtuon discrete)
#  p-value! (test permutations)
#  Test ajustement khi-carré
#  Test indépendance variables aléatoires catégoriques



class ImageManip:

    def __init__(self, image: np.ndarray):
        self.__pixels = image.copy()
        self.__originale = self.__pixels.copy()

    def rogner(self, x_min: int = None, x_max: int = None, y_min: int = None, y_max: int = None):
        self.__pixels = self.__pixels[y_min:y_max, x_min:x_max]

    def restaurer_originale(self):
        self.__pixels = self.__originale.copy()


def generer_sinus_avec_bruit(x_min: float, x_max: float, nb_x: int, amplitude_sinus: float = 1,
                             translation_horizontale_sinus: float = 0, translation_verticale_sinus: float = 0,
                             moyenne_bruit: float = 0, ecart_type_bruit: float = 0.1):
    x = np.linspace(x_min, x_max, nb_x)
    y = amplitude_sinus * np.sin(x + translation_horizontale_sinus) + translation_verticale_sinus
    bruit = np.random.normal(moyenne_bruit, ecart_type_bruit, y.shape)
    return x, y, bruit


def generer_polynome_avec_bruit(x_min: float, x_max: float, nb_x: int, coefficients: tuple, moyenne_bruit: float = 0,
                                ecart_type_bruit: float = 0.1):
    x = np.linspace(x_min, x_max, nb_x)
    y = np.polyval(coefficients, x)
    bruit = np.random.normal(moyenne_bruit, ecart_type_bruit, y.shape)
    return x, y, bruit


if __name__ == '__main__':
    x = np.linspace(0, 10, 100)
