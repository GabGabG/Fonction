"""
Fichier permettant de générer des données, que ce soit sous la forme de tuples (x,y) ou (x,y,z), ou sous la forme
de distributions de probabilité, ou finalement sous la forme d'images. Les méthodes utilisées ici ne sont pas inclues
dans le cadre des capsules, mais des commentaires seront présents pour expliquer ce qui se passe pour ceux que ça
intéresse! Amusez-vous avec le code si vous voulez :)

Code initialement écrit par Gabriel Genest
"""
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from warnings import warn
from typing import Union, Collection


class RK4:
    """
    Classe permettant de résoudre un système d'équations différentielles ordinaires selon l'algorithme de Runge-Kutta
    d'ordre 4.
    """

    def __init__(self, fonction: callable, conditions_initiales: tuple, t_min: float, t_max: float, nombre_t: int,
                 *fargs):
        """
        Constructeur de la classe RK4. Sert à intégrer numériquement un système d'équations différentielles ordinaires.
        La dimension du système importe peu, la nature "vectorisée" du code fait en sorte qu'autant un système en 1D
        qu'un système en 5D peut se résoudre. Utilise l'algorithme Runge-Kutta d'ordre 4.
        :param fonction: Fonction (i.e. système d'équations) à résoudre. Doit être sous la forme d'un "callable", soit
        un objet qu'on appelle avec des paramètres et qui donne une sortie. Il doit prendre au moins 2 paramètres:
        la "position" (i.e. un vecteur d'état) ainsi que le pas de temps courant.
        :param conditions_initiales: Tuple contenant les conditions initiales. Par exemple, cela peut être la position
        et la vitesse au temps 0.
        :param t_min: Temps minimal d'intégration.
        :param t_max: Temps maximal d'intégration.
        :param nombre_t: Nombre de pas de temps pour l'intégration numérique.
        :param fargs: Tout autre argument devant être passé aux équations différentielles lors de la résolution.
        """
        self.fonction = fonction
        self.conditons_init = conditions_initiales
        self.t_min = t_min
        self.t_max = t_max
        self.nombre_t = nombre_t
        self.h = (self.t_max - self.t_min) / self.nombre_t
        self.t_points = np.linspace(self.t_min, self.t_max, self.nombre_t)
        self.fargs = fargs

    def resoudre(self):
        """
        Méthode permettant de résoudre numériquement le système d'équations différentielles.
        :return: x_points, un vecteur contenant les différents états en fonction du temps.
        """
        nb_conditions_init = len(self.conditons_init)
        x_points = np.zeros((nb_conditions_init, len(self.t_points)), float)
        x = self.conditons_init
        f = lambda x, t: self.fonction(x, t, *self.fargs)
        for index, t in enumerate(self.t_points):
            x_points[:, index] = x
            k1 = self.h * f(x, t)
            k2 = self.h * f(x + 0.5 * k1, t + 0.5 * self.h)
            k3 = self.h * f(x + 0.5 * k2, t + 0.5 * self.h)
            k4 = self.h * f(x + k3, t + self.h)
            x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return x_points


class EDOs:

    def __init__(self, conditions_initiales, nom_variables: tuple = None):
        self.conditions_initiales = conditions_initiales
        n_dim = len(conditions_initiales)
        self.x_points = None
        self.t_points = None
        self.nom_variables = nom_variables if nom_variables is not None else tuple([f"v{i}" for i in range(n_dim)])

    def equations_differentielles(self, x: np.ndarray, t: float):
        raise NotImplementedError("À implémenter dans les sous-classes spécifiques.")

    def resoudre_EDO(self, t_min: float, t_max: float, nombre_points_t: int):
        """
        Méthode permettant de résoudre les équations différentielles à l'aide de la
        classe RK4.
        :param t_min: Temps minimal à considérer pour la résolution numérique.
        :param t_max: Temps maximal à considérer pour la résolution numérique.
        :param nombre_points_t: Nombre de pas de temps à considérer pour la résolution numérique.
        :return: etats, tuple contenant les états et les pas de temps associés.
        """
        rk4 = RK4(self.equations_differentielles, self.conditions_initiales, t_min, t_max, nombre_points_t)
        self.x_points = rk4.resoudre()
        self.t_points = rk4.t_points
        etats = (self.x_points, self.t_points)
        return etats

    def enregistrer_resolution(self, nom_fichier: str, ecrire_colonnes: bool = True):
        """
        Méthode permettant de sauvegarder la résolution numérique courante sous le format CSV. La première colonne
        contient l'information sur les pas de temps, alors que les autres contienent l'information sur les états.
        :param nom_fichier: Nom du fichier dans lequel sauvegarder les informations. Si le fichier existe déjà, il sera
        écrasé.
        :param ecrire_colonnes: Booléen spécifiant si on veut écrire le nom des colonnes en en-tête du fichier. Par
        défaut, on les écrit.
        :return: Liste contenant le nom des colonnes. Permet d'avoir une référence sur quelle colonne contient quoi.
        """
        if self.x_points is None:
            raise ValueError("Il faut résoudre le système avant de l'enregistrer")

        X = self.x_points.T
        T = self.t_points
        colonnes = ",".join(str(var) for var in self.nom_variables)
        colonnes = "t," + colonnes

        with open(nom_fichier, "w") as fichier:
            if ecrire_colonnes:
                fichier.write(colonnes + "\n")
            for vars, t in zip(X, T):
                ligne = ",".join(str(valeur) for valeur in ([t] + list(vars)))
                fichier.write(ligne + "\n")

        return colonnes


class OscillateurHarmonique(EDOs):
    """
    Classe modélisant un oscillateur harmonique (comme un système masse-ressorte). Hérite de EDOs.
    """

    def __init__(self, conditions_initiales: tuple, frequence: float):
        """
        Constructeur de la classe OscillateurHarmonique. Sert à modéliser un oscillateur harmonique tel qu'un système
        mass-ressort.
        :param conditions_initiales: Tuple contenant les conditions initiales du système: la position et la vitesse au
        temps 0.
        :param frequence: Fréquence de l'oscillateur (i.e. nombre "d'allers retours" effectués en 1 seconde).
        """
        super(OscillateurHarmonique, self).__init__(conditions_initiales, nom_variables=("x", "x'"))
        self.frequence = frequence

    def equations_differentielles(self, x: np.ndarray, t: float):
        """
        Méthode permettant de résoudre le système d'équations différentielles de l'oscillateur harmonique.
        :param x: Vecteur d'état courant [position, vitesse]. Doit être un ndarray de NumPy.
        :param t: Pas de temps courant (inutile dans le cas présent, mais doit être inclus pour des raisons de
        compatibilité).
        :return: etat_suivant, vecteur d'état du prochain pas de temps.
        """
        x_1 = x[0]
        x_2 = x[1]
        d_x_1 = x_2
        d_x_2 = -(self.frequence ** 2) * x_1
        etat_suivant = np.array([d_x_1, d_x_2])
        return etat_suivant


class LorenzAttractor(EDOs):
    """
    Classe modélisant un attracteur de Lorenz (très cool et beau à voir!). Hérite de EDOs.
    """

    def __init__(self, conditions_initiales: tuple, rho: float, sigma: float, beta: float):
        """
        Constructeur de la classe LorenzAttractor. Sert à modéliser un attracteur de Lorenz. Ce concept est très
        intéressant et pertinent en chaos et en analyse non linéaire.
        :param conditions_initiales: Tuple des conditions initiales, soit x, y, z au temps 0.
        :param rho: Paramètre (peut être essentiellement n'importe quoi).
        :param sigma: Paramètre (peut être essentiellement n'importe quoi).
        :param beta: Paramètre (peut être essentiellement n'importe quoi).
        """
        super(LorenzAttractor, self).__init__(conditions_initiales, nom_variables=("x", "y", "z"))
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    def equations_differentielles(self, X: np.ndarray, t: float):
        """
        Méthode permettant de résoudre le système d'équations différentielles de l'attracteur de Lorenz.
        :param X: Vecteur d'état courant [x, y, z]. Doit être un ndarray de NumPy.
        :param t: Pas de temps courant (inutile dans le cas présent, mais doit être inclus pour des raisons de
        compatibilité).
        :return: etat_suivant, vecteur d'état du prochain pas de temps.
        """
        x = X[0]
        y = X[1]
        z = X[2]
        d_x = self.sigma * (y - x)
        d_y = x * (self.rho - z) - y
        d_z = x * y - self.beta * z
        etat_suivant = np.array([d_x, d_y, d_z])
        return etat_suivant


class NuageDePointsBase:

    def __init__(self, x_s: np.ndarray, y_s: np.ndarray):
        if x_s.shape != y_s.shape:
            raise ValueError("La forme des vecteurs doit être la même.")
        self.__x_s = x_s
        self.__y_s = y_s

    def enregistrer(self, nom_fichier: str, ecrire_colonnes: bool = True):
        x = self.__x_s
        y = self.__y_s
        colonnes = "x,y"

        with open(nom_fichier, "w") as fichier:
            if ecrire_colonnes:
                fichier.write(colonnes + "\n")
            for i, j in zip(x, y):
                ligne = f"{i},{j}"
                fichier.write(ligne + "\n")

        return colonnes


class NuageDePointsDistributionNormale(NuageDePointsBase):

    def __init__(self, x_min: float, x_max: float, nb_points: int, moyenne_normale: float = 0,
                 ecart_type_normale: float = 1, seed: int = None):
        if seed is not None:
            np.seed(seed)
        x_s = np.random.uniform(x_min, x_max, nb_points)
        if seed is not None:
            np.seed(seed)
        y_s = np.random.normal(moyenne_normale, ecart_type_normale, nb_points)
        super(NuageDePointsDistributionNormale, self).__init__(x_s, y_s)


class NuageDePointsDistributionPoisson(NuageDePointsBase):

    def __init__(self, x_min: float, x_max: float, nb_points: int, moyenne_poisson: float, seed: int = None):
        if seed is not None:
            np.seed(seed)
        x_s = np.random.uniform(x_min, x_max, nb_points)
        if seed is not None:
            np.seed(seed)
        y_s = np.random.poisson(moyenne_poisson, nb_points)
        super(NuageDePointsDistributionPoisson, self).__init__(x_s, y_s)


class Distribution:
    """
    Class générale permettant de modéliser une distribution de probabilité, autant discrète que continue.
    """

    def __init__(self):
        """
        Constructeur de la classe Distribution. Ne prend aucun paramètre.
        """
        self._y = None
        self._x = None

    @property
    def x(self):
        if self._x is None:
            return None
        return self._x.copy()

    @property
    def y(self):
        if self._y is None:
            return None
        return self._y.copy()

    def fonction(self, *args, **kwargs):
        """
        Méthode permettant de retourner la fonction de probabilité évaluée à un seul point ou à plusieurs points.
        :param x: Point(s) où évaluer la fonction.
        :return: La valeur de probabilité associée à chaque points. Si un seul point est donné, retourne un scalaire. Si
        plusieurs points, retourne un vecteur.
        """
        msg = "La fonction de probabilité est spécifique à chaque loi. À implémenter dans les sous-classes."
        raise NotImplementedError(msg)

    def distribution_probabilite(self, *args, **kwargs):
        raise NotImplementedError("Implémentée de manière différente selon distribution discrète ou continue.")

    def enregistrer(self, nom_fichier: str, ecrire_colonnes: bool = True):
        """
        Méthode permettant d'enregistrer une distribution générée.
        :param nom_fichier: Nom du fichier dans lequel enregistrer la distribution. Si le fichier existe déjà, il sera
        écrasé.
        :param ecrire_colonnes: Booléen spécifiant si on veut écrire le nom des colonnes en en-tête du fichier. Par
        défaut, on les écrit.
        :return: colonnes, indicateur de quelle colonne contient quoi. Ici, la première est la colonne des valeurs x,
        alors que la seconde est la colonne des valeurs y, donc des probabilités associées.
        """
        if self._x is None:
            raise ValueError("Il faut générer une distribution avant de l'enregistrer.")

        x = self._x
        y = self._y
        colonnes = "x,y"

        with open(nom_fichier, "w") as fichier:
            if ecrire_colonnes:
                fichier.write(colonnes + "\n")
            for i, j in zip(x, y):
                ligne = f"{i},{j}"
                fichier.write(ligne + "\n")

        return colonnes


class DistributionDiscrete(Distribution):

    def fonction(self, x: Union[int, Collection[int]]):
        if not np.all(x % 1 == 0):
            raise TypeError("La variable 'x' doit être composée de valeurs entières.")

    def distribution_probabilite(self, x_min: int, x_max: int, pas: int):
        self._x = np.arange(x_min, x_max + 1, pas)
        self._y = self.fonction(self._x)

        return self._x, self._y


class DistributionContinue(Distribution):

    def distribution_probabilite(self, x_min: float, x_max: float, nb_x: int):
        self._x = np.linspace(x_min, x_max, nb_x)
        self._y = self.fonction(self._x)

        return self._x, self._y

    def fonction(self, x: Union[float, Collection[float]]):
        return None


class DistributionBinomiale(DistributionDiscrete):

    def __init__(self, probabilite_succes: float, nombre_essais: int):
        super(DistributionBinomiale, self).__init__()
        if not (0 <= probabilite_succes <= 1):
            raise ValueError("La probabilité de succès doit être entre 0 et 1 inclusivement.")
        self.__p = probabilite_succes
        if nombre_essais < 0:
            raise ValueError("Le nombre d'essais doit être positif (mais peut être 0).")
        self.__n = nombre_essais

    @property
    def p(self):
        return self.__p

    @property
    def q(self):
        return 1 - self.__p

    @property
    def n(self):
        return self.__n

    @p.setter
    def p(self, probabilite_succes: float):
        if not (0 <= probabilite_succes <= 1):
            raise ValueError("La probabilité de succès doit être entre 0 et 1 inclusivement.")
        self.__p = probabilite_succes

    @n.setter
    def n(self, nombre_essais: int):
        if nombre_essais < 0:
            raise ValueError("Le nombre d'essais doit être positif (mais peut être 0).")
        self.__n = nombre_essais

    def fonction(self, x: Union[int, Collection[int]]):
        super(DistributionBinomiale, self).fonction(x)
        n = self.__n
        p = self.__p
        x = np.ravel(x)
        res = np.zeros_like(x, dtype=float)
        for i, x_ in enumerate(x):
            if 0 <= x_ <= n:
                res[i] = coefficient_binomial(n, x_) * p ** x_ * (1 - p) ** (n - x_)
        if res.size == 1:
            res = res.item()
        return res


class DistributionPoisson(DistributionDiscrete):

    def __init__(self, moyenne: float):
        super(DistributionPoisson, self).__init__()
        if moyenne < 0:
            raise ValueError("La moyenne doit être positive (mais peut être 0).")
        self.__mu = moyenne

    @property
    def moyenne(self):
        return self.__mu

    @moyenne.setter
    def moyenne(self, moyenne: float):
        if moyenne < 0:
            raise ValueError("La moyenne doit être positive (mais peut être 0).")
        self.__mu = moyenne

    def fonction(self, x: Union[int, Collection[int]]):
        super(DistributionPoisson, self).fonction(x)
        mu = self.__mu
        x = np.ravel(x)
        res = np.zeros_like(x, dtype=float)
        for i, x_ in enumerate(x):
            if x_ >= 0:
                res[i] = mu ** x_ * np.exp(-mu) / factorielle(x_)
        if res.size == 1:
            res = res.item()
        return res


class DistributionUniforme(DistributionContinue):

    def __init__(self, alpha: float, beta: float):
        super(DistributionUniforme, self).__init__()
        if alpha > beta:
            warn("Le paramètre 'alpha' devrait être plus petit que 'beta'. Les deux sont donc échangés.")
            t = alpha
            alpha = beta
            beta = t
        if alpha == beta:
            raise ValueError("Les bornes doivent être différentes.")
        self.alpha = alpha
        self.beta = beta

    def fonction(self, x):
        a = self.alpha
        b = self.beta
        x = np.ravel(x)
        res = np.zeros_like(x, dtype=float)
        res[(a <= x) & (x <= b)] = 1 / (b - a)
        if res.size == 1:
            res = res.item()
        return res


class DistributionExponentielle(DistributionContinue):

    def __init__(self, taux: float):
        super(DistributionExponentielle, self).__init__()
        if taux < 0:
            raise ValueError("Le taux doit être positif (mais peut être 0).")
        self.__taux = taux

    @property
    def taux(self):
        return self.__taux

    @taux.setter
    def taux(self, taux: float):
        if taux < 0:
            raise ValueError("Le taux doit être positif (mais peut être 0).")
        self.__taux = taux

    def fonction(self, x):
        x = np.ravel(x)
        res = np.zeros_like(x, dtype=float)
        lambda_ = self.__taux
        x_inf_0 = x >= 0
        res[x_inf_0] = lambda_ * np.exp(-lambda_ * x[x_inf_0])
        if res.size == 1:
            res = res.item()
        return res


class DistributionNormale(DistributionContinue):

    def __init__(self, moyenne: float, ecart_type: float):
        super(DistributionNormale, self).__init__()
        self.mu = moyenne
        if ecart_type == 0:
            raise ValueError("L'écart type ne peut être nul.")
        if ecart_type < 0:
            warn("L'écart type est négatif. La valeur absolue sera prise.")
            ecart_type = -ecart_type
        self.sigma = ecart_type

    def fonction(self, x):
        s = self.sigma
        mu = self.mu
        return 1 / (s * (2 * np.pi) ** .5) * np.exp(-.5 * ((x - mu) / s) ** 2)


class EnregistrerImage:

    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels

    def enregistrer_image(self, nom_fichier: str, format: str = "tiff", normalise: bool = False):
        nom_complet = nom_fichier + "." + format
        pixels = self.pixels
        if normalise:
            pixel_max = np.max(self.pixels)
            pixels = pixels.astype(float) / pixel_max
        if format.lower() in ["tif", "tiff"]:
            tifffile.imwrite(nom_complet, pixels)
        else:
            plt.imsave(nom_complet, pixels)


class GenererSinus(EnregistrerImage):

    def __init__(self, taille_image: tuple, freq: int = 1, horizontal: bool = True):
        taille = taille_image
        taille_x = taille[0]
        taille_y = taille[1]
        x_s = np.linspace(0, 1, taille_x) * np.pi * freq * 2
        y_s = np.linspace(0, 1, taille_y) * np.pi * freq * 2
        xx, yy = np.meshgrid(x_s, y_s)
        if horizontal:
            pixels = np.sin(xx)
        else:
            pixels = np.sin(yy)
        super(GenererSinus, self).__init__(pixels)


class GenererSpecklesLaser(EnregistrerImage):

    def __init__(self, taille_image: tuple, diametre_simulation: float):
        diam = diametre_simulation
        rayon = diam / 2
        xx, yy = np.indices(taille_image)
        xx -= taille_image[0] // 2
        yy -= taille_image[1] // 2
        mask = (xx ** 2 + yy ** 2 - rayon ** 2) <= 0
        phases = np.exp(1j * np.random.uniform(-np.pi, np.pi, taille_image))
        simulation = mask * phases
        simulation = np.abs(np.fft.fftshift(np.fft.fft2(simulation))) ** 2
        simulation = simulation.real
        simulation /= np.max(simulation)
        super(GenererSpecklesLaser, self).__init__(simulation)


def factorielle(x):
    if x % 1 != 0 or x < 0:
        raise ValueError("La fonction factorielle n'est définie que pour des entiers positifs.")
    if x <= 1:
        return 1
    return x * factorielle(x - 1)


def coefficient_binomial(n, k):
    if k > n:
        return 0
    if k == 0:
        return 1
    num = factorielle(n)
    denom_1 = factorielle(k)
    denom_2 = factorielle(n - k)
    return num / (denom_1 * denom_2)


if __name__ == '__main__':
    import scipy.stats as st
    st.truncnorm()
    lorenz = LorenzAttractor((10, 0, 5), 28, 10, 8 / 3)
    lorenz.resoudre_EDO(0, 50, 100_000)
    lorenz.enregistrer_resolution("data/lorenz.csv")
    plt.plot(lorenz.t_points, lorenz.x_points[0])
    plt.show()
    # oh = OscillateurHarmonique((0, 10), 0.25)
    # oh.resoudre_EDO(0, 100, 1000)
    # plt.plot(oh.t_points, oh.x_points[0])
    # plt.show()
    # oh.enregistrer_resolution("oscillateur_harmonique.csv")
    # exit()
    # nuage_points = NuageDePointsDistributionNormale(0, 100, 50)
    # nuage_points.enregistrer("data/nuageGauss.csv")

    distribution_gauss = NuageDePointsDistributionNormale(0, 10_000, 10_000)
    distribution_gauss.enregistrer("data/nuageGauss_big.csv")
    #
    # fit_gauss = DistributionNormale(0, 1)
    # fit_gauss.distribution_probabilite(-5, 5, 1000)
    # fit_gauss.enregistrer("data/distributionGauss.csv")
    # exit()
    speckles = GenererSpecklesLaser((1000, 1000), 1000 / 10)
    speckles.enregistrer_image("data/testSpeckles", "tiff", normalise=True)

    sinus = GenererSinus((1000, 1000), 5)
    sinus.enregistrer_image("data/testSinus", "png")
