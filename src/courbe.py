import numpy as np
import matplotlib.pyplot as plt
from numbers import Number
from typing import *
from .style import Marqueurs, Couleurs, Lignes, Style, StyleCourbe


class Courbe:
    __courbe_courante = 0

    def __init__(self, x: Iterable, y: Iterable, err_x: Union[Number, Iterable] = None,
                 err_y: Union[Number, Iterable] = None, label: str = None):
        self.__x = np.ravel(x)
        self.__y = np.ravel(y)
        if isinstance(err_x, Number):
            err_x = np.full_like(x, err_x, type(err_x))
        if isinstance(err_y, Number):
            err_y = np.full_like(y, err_y, type(err_y))
        self.__err_x = np.ravel(err_x)
        self.__err_y = np.ravel(err_y)
        Courbe.__courbe_courante += 1
        if label is None:
            label = f"Courbe {Courbe.__courbe_courante}"
        self.label = label
        self.__fig_crees = []

    @property
    def x(self):
        return self.__x.copy()

    @property
    def y(self):
        return self.__y.copy()

    @property
    def err_x(self):
        err_x = self.__err_x
        if err_x is None:
            return None
        return err_x.copy()

    @property
    def err_y(self):
        err_y = self.__err_y
        if err_y is None:
            return None
        return err_y.copy()

    @x.setter
    def x(self, nouveau_x: Iterable):
        self.__x = nouveau_x

    @y.setter
    def y(self, nouveau_y: Iterable):
        self.__y = nouveau_y

    @err_x.setter
    def err_x(self, nouveau_err_x: Union[Number, Iterable]):
        if isinstance(nouveau_err_x, Number):
            nouveau_err_x = np.full_like(self.__err_x, nouveau_err_x, type(nouveau_err_x))
        self.__err_x = nouveau_err_x

    @err_y.setter
    def err_y(self, nouveau_err_y: Union[Number, Iterable]):
        if isinstance(nouveau_err_y, Number):
            nouveau_err_y = np.full_like(self.__err_y, nouveau_err_y, type(nouveau_err_y))
        self.__err_y = nouveau_err_y

    def construire_graphique(self, avec_erreurs_x: bool = True, avec_erreurs_y: bool = True,
                             bande_erreur_y: bool = False, style: Style = Style(), ax: plt.Axes = None):
        if ax is None:
            fig, ax = plt.subplots()
            self.__fig_crees.append(fig)
        if style is not None:
            style_dict = style.format_dictionnaire()
        else:
            return None
        err_x = self.__err_x if avec_erreurs_x else None
        err_y = self.__err_y if avec_erreurs_y and not bande_erreur_y else None
        ax.errorbar(self.__x, self.__y, xerr=err_x, yerr=err_y, label=self.label, **style_dict)
        if bande_erreur_y and err_y is not None:
            couleur = style_dict["color"]
            ax.fill_between(self.__x, self.__y - err_y, self.__y + err_y, alpha=0.5, color=couleur)
        return ax

    def afficher_courbe(self, avec_erreurs_x: bool = True, avec_erreurs_y: bool = True, bande_erreur_y: bool = False,
                        style: Style = Style(), nom_x: str = "x", nom_y: str = "y", ax: plt.Axes = None):
        self.construire_graphique(avec_erreurs_x, avec_erreurs_y, bande_erreur_y, style, ax)
        plt.legend()
        plt.xlabel(nom_x)
        plt.ylabel(nom_y)
        plt.show()

    def vider_figures_crees(self):
        for fig in self.__fig_crees:
            plt.close(fig)
        self.__fig_crees.clear()

    def __del__(self):
        self.vider_figures_crees()
        self.__courbe_courante -= 1

    def __eq__(self, other):
        if isinstance(other, Courbe):
            x_egaux = np.array_equal(self.x, other.x)
            y_egaux = np.array_equal(self.y, other.y)
            err_x_egaux = np.array_equal(self.err_x, other.err_x)
            err_y_egaux = np.array_equal(self.err_y, other.err_y)
            return x_egaux and y_egaux and err_x_egaux and err_y_egaux


class Courbes:

    def __init__(self, courbes: List[Courbe], styles: List[Style] = None, generer_styles: bool = True,
                 *arguments_generer_styles, **arguments_generer_styles_dict):
        self.__courbes = courbes
        nb_courbes = len(courbes)
        if styles is not None:
            if len(styles) != nb_courbes:
                raise ValueError("Le nombre de styles doit être égal au nombre de courbes.")
        elif generer_styles:
            styles = self.generer_styles_pour_courbes(nb_courbes, *arguments_generer_styles,
                                                      **arguments_generer_styles_dict)
        else:
            styles = [None] * nb_courbes
        self.__styles = styles

    @property
    def courbes(self):
        return self.__courbes

    @property
    def styles(self):
        return self.__styles

    def ajouter_courbe(self, courbe: Courbe, style: Style = Style()):
        self.__courbes.append(courbe)
        self.__styles.append(style)

    def enelver_courbe(self, index: int = None, courbe: Courbe = None):
        if index is None and courbe is None:
            return None
        if index is not None and courbe is not None:
            raise ValueError("Spécifier l'index ou la courbe, mais pas les deux.")
        if index is not None:
            self.__courbes.pop(index)
            self.__styles.pop(index)
        if courbe is not None:
            index = self.__courbes.index(courbe)
            self.__courbes.remove(courbe)
            self.__styles.pop(index)

    def changer_courbe(self, nouvelle_courbe: Courbe, index: int = None, ancienne_courbe: Courbe = None,
                       nouveau_style: Style = None):
        if index is None and ancienne_courbe is None:
            return None
        if index is not None and ancienne_courbe is not None:
            raise ValueError("Spécifier l'index ou la courbe, mais pas les deux.")
        if index is None:
            index = self.__courbes.index(ancienne_courbe)
        self.__courbes[index] = nouvelle_courbe
        if nouveau_style is not None:
            self.__styles[index] = nouveau_style

    def changer_style(self, nouveau_style: Style, index: int = None, ancien_style: Style = None):
        if index is None and ancien_style is None:
            return None
        if index is not None and ancien_style is not None:
            raise ValueError("Spécifier l'index ou le style, mais pas les deux.")
        if index is not None:
            self.__styles[index] = nouveau_style
        if ancien_style is not None:
            index = self.__styles.index(ancien_style)
            self.__styles[index] = nouveau_style

    def preparer_graphique(self, arguments_par_courbe: List[dict] = None):
        fig, ax = plt.subplots()
        nb_courbes = len(self.__courbes)
        args = [dict() for _ in range(nb_courbes)]
        if arguments_par_courbe is not None:
            if len(arguments_par_courbe) == 1:
                args = [arguments_par_courbe[0] for _ in range(nb_courbes)]
            elif len(arguments_par_courbe) != nb_courbes:
                msg = "Le paramètre `arguments_par_courbe` peut avoir un seul élément qui sera utilisé pour chaque " \
                      "courbe ou bien avoir autant d'éléments que de courbes. Il peut aussi être None pour n'avoir " \
                      "aucun argument."
                raise ValueError(msg)
            else:
                args = arguments_par_courbe
        for i, courbe in enumerate(self.__courbes):
            args_courants = args[i]
            args_courants["ax"] = ax
            style = self.__styles[i]
            if style is not None:
                args_courants["style"] = style
            courbe.construire_graphique(**args_courants)
        return ax

    def afficher_graphique(self, arguments_par_courbe: List[dict] = None, nom_x: str = "x", nom_y: str = "y"):
        self.preparer_graphique(arguments_par_courbe)
        plt.legend()
        plt.xlabel(nom_x)
        plt.ylabel(nom_y)
        plt.show()

    @classmethod
    def generer_style_pour_courbe(cls, marqeur: bool = True, ligne: bool = True, index: int = 0):
        if not marqeur and not ligne:
            return None
        m = None if not marqeur else cls.__generer_marqueur(index)
        l = None if not ligne else cls.__generer_ligne(index)
        c = cls.__generer_couleur(index)
        return StyleCourbe(c, m, l)

    @classmethod
    def generer_styles_pour_courbes(cls, nombre_courbes: int, marqueurs: Union[bool, List[int]] = True,
                                    lignes: Union[bool, List[int]] = True):
        styles = []
        marqueurs_ = marqueurs
        lignes_ = lignes
        if marqueurs is True:
            marqueurs_ = list(range(1, nombre_courbes + 1))
        if lignes is True:
            lignes_ = list(range(1, nombre_courbes + 1))
        if marqueurs is False and lignes is False:
            return [None] * nombre_courbes
        marqueurs = marqueurs_
        lignes = lignes_
        for courbe in range(1, nombre_courbes + 1):
            s = cls.generer_style_pour_courbe(courbe in marqueurs, courbe in lignes, courbe - 1)
            styles.append(s)
        return styles

    @classmethod
    def __generer_marqueur(cls, index: int, modulo: bool = True):
        if modulo:
            index = index % Marqueurs.nombre_marqueurs()
        marqueur = Marqueurs.marqueurs_valeurs()[index]
        return marqueur

    @classmethod
    def __generer_ligne(cls, index: int, modulo: bool = True):
        if modulo:
            index = index % Lignes.nombre_lignes()
        ligne = Lignes.lignes_valeurs()[index]
        return ligne

    @classmethod
    def __generer_couleur(cls, index: int, modulo: bool = True):
        if modulo:
            index = index % Couleurs.nombre_couleurs()
        couleur = Couleurs.couleurs_valeurs()[index]
        return couleur
