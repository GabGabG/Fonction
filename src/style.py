from typing import *

class Couleurs:
    __noms = ["noir", "rouge", "orange", "vert", "bleu", "magenta", "rose", "cyan", "lime",
              "ciel", "maron", "jaune", "argent"]

    __valeurs = ["black", "red", "orange", "green", "blue", "magenta", "pink", "cyan",
                 "lime", "deepskyblue", "maroon", "yellow", "silver"]

    __styles = dict(zip(__noms, __valeurs))

    @classmethod
    def ajouter_couleur_tuple_ou_hex_RGB(cls, nom_couleur: str, RGB: Union[tuple, str]):
        cls.__styles[nom_couleur] = RGB
        cls.__noms.append(nom_couleur)
        cls.__valeurs.append(RGB)

    @classmethod
    def ajouter_couleur(cls, nom_couleur: str, nom_matplotlib: str):
        cls.__styles[nom_couleur] = nom_matplotlib
        cls.__noms.append(nom_couleur)
        cls.__valeurs.append(nom_matplotlib)

    @classmethod
    def enlever_couleur(cls, nom: str):
        pop = cls.__styles.pop(nom, None)
        if pop is not None:
            cls.__noms.remove(nom)
            cls.__valeurs.remove(pop)
        return pop

    @classmethod
    def couleur(cls, nom: str):
        couleur = cls.__styles[nom]
        return couleur

    @classmethod
    def couleurs(cls):
        return cls.__styles

    @classmethod
    def couleurs_noms(cls):
        return cls.__noms

    @classmethod
    def couleurs_valeurs(cls):
        return cls.__valeurs

    @classmethod
    def nombre_couleurs(cls):
        return len(cls.__styles)


class Lignes:
    __noms = ["pleine", "pointillée", "pointillée variante", "pointillée alterne", "pointillée espacé"]

    __valeurs = ["-", ":", "--", "-.", (0, (1, 10))]

    __styles = dict(zip(__noms, __valeurs))

    @classmethod
    def ajouter_ligne_custom(cls, nom_ligne: str, offset_initial: int = 0, sequence: tuple = (1, 5)):
        val = (offset_initial, sequence)
        cls.__styles[nom_ligne] = val
        cls.__noms.append(nom_ligne)
        cls.__valeurs.append(val)

    @classmethod
    def enelver_ligne(cls, nom: str):
        pop = cls.__styles.pop(nom, None)
        if pop is not None:
            cls.__noms.remove(nom)
            cls.__valeurs.remove(pop)
        return pop

    @classmethod
    def ligne(cls, nom: str):
        ligne = cls.__styles.get(nom, None)
        return ligne

    @classmethod
    def lignes(cls):
        return cls.__styles

    @classmethod
    def lignes_noms(cls):
        return cls.__noms

    @classmethod
    def lignes_valeurs(cls):
        return cls.__valeurs

    @classmethod
    def nombre_lignes(cls):
        return len(cls.__styles)


class Marqueurs:
    __noms = ['point', 'pixel', 'cercle', 'triangle bas', 'triangle haut', 'triangle gauche', 'triangle droit',
              'tri bas', 'tri haut', 'tri gauche', 'tri droit', 'octogone', 'carré', 'pentagone', 'plus variante',
              'étoile', 'hexagone', 'hexagone variante', 'plus', 'x', 'x variante', 'diamand', 'diamant variante',
              'ligne vertical', 'ligne horizontal', 'ligne gauche', 'ligne droit', 'ligne haut', 'ligne bas',
              'caret gauche', 'caret droit', 'caret haut', 'caret bas', 'caret gauche variante', 'caret droit variante',
              'caret haut variante', 'caret bas variante']

    __valeurs = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
                 'X',
                 'd', 'D', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    __styles = dict(zip(__noms, __valeurs))

    @classmethod
    def ajouter_marqueur_custom_forme(cls, nom_marqueur: str, nb_cotes: int = 3, style: int = 0,
                                      angle_rotation: float = 0):
        if style not in [0, 1, 2]:
            raise ValueError("Le style ne peut être que 0, 1 ou 2")
        val = (nb_cotes, style, angle_rotation)
        cls.__styles[nom_marqueur] = val
        cls.__noms.append(nom_marqueur)
        cls.__valeurs.append(val)

    @classmethod
    def ajouter_marqueur_custom_lettre(cls, nom_marqueur: str, lettre: str):
        val = f"${lettre}$"
        cls.__styles[nom_marqueur] = val
        cls.__noms.append(nom_marqueur)
        cls.__valeurs.append(val)

    @classmethod
    def enlever_marqueur(cls, nom: str):
        pop = cls.__styles.pop(nom, None)
        if pop is not None:
            cls.__noms.remove(nom)
            cls.__valeurs.remove(pop)
        return pop

    @classmethod
    def marqueur(cls, nom: str):
        mar = cls.__styles.get(nom, None)
        return mar

    @classmethod
    def marqueurs(cls):
        return cls.__styles

    @classmethod
    def marqueurs_noms(cls):
        return cls.__noms

    @classmethod
    def marqueurs_valeurs(cls):
        return cls.__valeurs

    @classmethod
    def nombre_marqueurs(cls):
        return len(cls.__styles)


class Style:

    def __init__(self, couleur: Union[str, tuple] = None):
        self.couleur = couleur

    def format_dictionnaire(self):
        d = {"color": self.couleur}
        return d

    def __eq__(self, other):
        if isinstance(other, Style):
            return self.couleur == other.couleur
        return False


class StyleCourbe(Style):

    def __init__(self, couleur: Union[str, tuple] = None, marqueurs: Union[str, int] = None,
                 ligne: Union[str, tuple] = Lignes.ligne("pleine")):
        super(StyleCourbe, self).__init__(couleur)
        self.marqueurs = marqueurs
        self.ligne = ligne if ligne is not None else ""

    def format_dictionnaire(self):
        d = super(StyleCourbe, self).format_dictionnaire()
        d.update({"marker": self.marqueurs, "linestyle": self.ligne})
        return d

    def __eq__(self, other):
        if isinstance(other, StyleCourbe):
            return self.couleur == other.couleur and self.marqueurs == other.marqueurs and self.ligne == other.ligne
        elif isinstance(other, Style):
            return self.couleur == other.couleur and self.marqueurs is None and self.ligne == "-"
        return False
