import warnings
from typing import Union, Tuple, Iterable, Type, Sequence
from src.regression_interpolation import _Liaison, LiaisonMixte
from src.courbe import Courbe
from src.style import Style
from src.variables import VariablesDependantes, VariablesIndependantes
import numpy as np
from matplotlib.pyplot import Axes
from numbers import Number


class Fonction:

    # TODO: mettre un from_csv_file

    def __init__(self, x: Iterable, y: Iterable, label: str = "Fonction"):
        """
        Constructeur de la classe `Fonction`. Permet de créer une version "informatique" de ce qu'est une fonction
        mathématique.
        :param x: iterable. Observable indépendant, souvent appelé "variable x". Doit être de même taille que `y`.
        :param y: iterable. Observable dépendant, souvent appelé "variable y". Doit être de même taille que `x`.
        :param label: str. Nom donné à la fonction courante. "Fonction" par défaut.
        """
        if not isinstance(y, VariablesDependantes):
            y = VariablesDependantes(y)
        if not isinstance(x, VariablesIndependantes):
            x = VariablesIndependantes(x)
        if len(x) != len(y):
            raise ValueError("Les variables dépendantes et indépendantes doivent avoir la même taille.")
        self._x = x
        self._y = y
        self._x._bloquer_modification_taille = True
        self._y._bloquer_modification_taille = True
        self._liaison = None
        self.label = label

    @property
    def info(self) -> dict:
        """
        Propriété de l'objet courant. Permet d'obtenir les informations de l'objet `Fonction` courant.
        :return: info, un dictionnaire contenant l'observable `x`, l'observable `y` et l'information importante de la
        liaison (si elle existe).
        """
        liaison_info = None
        if self._liaison is not None:
            liaison_info = self._liaison.info
        info = {"observable x": self.x, "observable y": self.y, "liaisons": liaison_info}
        return info

    @property
    def liaison(self) -> _Liaison:
        """
        Propriété de l'objet courant. Permet d'obtenir la liaison courante (`None` si elle n'existe pas).
        :return: la liaison courante.
        """
        return self._liaison

    @property
    def x(self) -> VariablesIndependantes:
        """
        Propriété de l'objet courant. Permet d'obtenir une *copie* de la variable indépendante `x`.
        :return: Une copie de l'attribut `x` (la variable indépendante).
        """
        return self._x.copie()

    @property
    def y(self) -> VariablesDependantes:
        """
        Propriété de l'objet courant. Permet d'obtenir une *copie* de la variable dépendante `y`.
        :return: Une copie de l'attribut `y` (la variable dépendante).
        """
        return self._y.copie()

    def __len__(self) -> int:
        """
        Méthode permettant de faire `len(self)` ou `self` est l'objet courant. Retourne le nombre d'éléments dans la
        variable indépendante (égal au nombre d'éléments dans la variable dépendante).
        :return: le nombre d'éléments dans la variable dépendante.
        """
        return len(self._x)

    def __call__(self, x: Union[int, float, complex, Iterable]) -> VariablesDependantes:
        """
        Méthode permettant d'appeler l'objet courant (i.e. faire `self(...)` où `self` est l'objet courant). Lance une
        exception si la liaison entre les variables indépendantes et dépendantes n'existe pas.
        :param x: nombre ou itérable. Valeur(s) à donner à la liaison (si elle existe).
        :return: un objet `VariablesDependantes` correspondant à l'application de la liaison courante (si elle existe)
        sur le paramètre donné `x`.
        """
        if self._liaison is None:
            raise ValueError("Veuillez spécifier une manière de 'lier' les valeurs.")
        valeurs = self._liaison(x)
        return VariablesDependantes(valeurs)

    def __getitem__(self, item: Union[int, slice]) -> Tuple[
        Union[int, float, complex, Iterable], Union[int, float, complex, Iterable]]:
        """
        Méthode permettant d'accéder à certains éléments des variables indépendantes et dépendantes.
        :param item: entier ou slice. Clé(s) de l'/des élément(s) auquel/auxquels on veut accéder.
        :return: le tuple (valeur(s) x, valeur(s) y).
        """
        y = self._y[item]
        x = self._x[item]
        return x, y

    def __setitem__(self, key: Union[int, slice, Iterable],
                    values: Tuple[Union[int, float, complex, Iterable, None], Union[
                        int, float, complex, Iterable, None]]) -> None:
        """
        Méthode permettant d'accéder et modifier certains éléments des variables indépendantes et dépendantes. Rend la
        liaison courante invalide.
        :param key: entier, slice ou itérable. Clé(s) de l'/des élément(s) auquel/auxquels on veut accéder et modifier.
        :param values: tuple de nombres ou d'itérables. Nouvelle(s) valeur(s) dont on veut modifier l'/les ancienne(s).
        L'élément 0 du tuple correspond à la/aux nouvelle(s) valeur(s) pour la variable indépendante `x`, alors que
        l'élément 1 du tuple correspond à la/aux nouvelle(s) valeur(s) pour la variable dépendante `y`. Dans les deux
        cas, `None` signifie ne pas changer les valeurs courantes (par exemple, si on veut seulement changer en `y`).
        :return: Rien.
        """
        nouvelles_x = values[0]
        nouvelle_y = values[1]
        if nouvelle_y is not None:
            self._y[key] = nouvelle_y
        if nouvelles_x is not None:
            self._x[key] = nouvelles_x
        if self._liaison is not None:
            warnings.warn("Liaison maintenant invalide.")
            self._liaison = None

    def changer_variables(self, key: Union[int, slice, Iterable],
                          values: Tuple[Union[int, float, complex, Iterable], Union[
                              int, float, complex, Iterable]] = (None, None)) -> None:
        """
        Méthode similaire à `__setitem__`. En fait, c'est une encapsulation.
        :param key: entier, slice ou itérable. Clé(s) de l'/des élément(s) auquel/auxquels on veut accéder et modifier.
        :param values: tuple de nombres ou d'itérables. Nouvelle(s) valeur(s) dont on veut modifier l'/les ancienne(s).
        L'élément 0 du tuple correspond à la/aux nouvelle(s) valeur(s) pour la variable indépendante `x`, alors que
        l'élément 1 du tuple correspond à la/aux nouvelle(s) valeur(s) pour la variable dépendante `y`. Dans les deux
        cas, `None` signifie ne pas changer les valeurs courantes (par exemple, si on veut seulement changer en `y`).
        :return: Rien.
        """
        self[key] = values

    def ajouter_variables(self,
                          valeurs: Tuple[Union[int, float, complex, Sequence], Union[int, float, complex, Sequence]],
                          positions: Union[int, slice, Iterable] = -1) -> bool:
        """
        Méthode permettant d'ajouter des variables indépendantes et dépendantes. Rend la liaison courant invalide.
        :param valeurs: tuple de nombres ou séquences. Valeur(s) à ajouter. L'élément 0 du tuple représente la/les
        valeurs à ajouter à la variable indépendante. Le second élément représente la/les valeurs à ajouter à la
        variable dépendante.
        :param positions: entier, slice ou itérable. Position(s) où ajouter la/les valeur(s). -1 par défaut, ce qui
        signifie à la fin.
        :return: ret, bool. Booléen qui dit si l'ajout a été fait.
        """
        self._x._bloquer_modification_taille = False
        self._y._bloquer_modification_taille = False
        ajout_x = valeurs[0]
        ajout_y = valeurs[1]
        if isinstance(ajout_x, Number):
            l_x = 1
        else:
            l_x = len(ajout_x)
        if isinstance(ajout_y, Number):
            l_y = 1
        else:
            l_y = len(ajout_y)
        if l_x != l_y:
            raise ValueError("Les variables dépendantes et indépendantes ajoutées doivent avoir la même taille.")
        ret = self._x.ajouter_variables(ajout_x, positions)
        if ret:  # Au cas où quelque chose arrive et qu'on ne puisse pas ajouter de variables.
            ret &= self._y.ajouter_variables(ajout_y, positions)
        self._x._bloquer_modification_taille = True
        self._y._bloquer_modification_taille = True
        if self._liaison is not None and ret:
            warnings.warn("Liaison maintenant invalide.")
            self._liaison = None
        return ret

    def enlever_variables(self, positions: Union[int, slice, Iterable] = None,
                          valeurs_x: Union[int, float, complex, Iterable] = None) -> Union[int, slice, Iterable, None]:
        """
        Méthode permettant d'enlever des valeurs aux variables indépendantes et dépendantes. Rend la liaison invalide.
        :param positions: int, slice ou Iterable. Index où enlever les valeurs. `None` par défaut, signifie
        qu'on ne retire pas par l'index.
        :param valeurs_x: nombre ou itérable. Valeurs en `x` à retirer (les valeurs `y` correspondantes sont aussi
        enlevées). `None` par défaut, signifie qu'on ne retire pas par l'index.
        :return: pos, int, slice, Iterable ou None. Positions où les variables ont été enlevées. `None` si aucun
        retrait.
        """
        self._x._bloquer_modification_taille = False
        self._y._bloquer_modification_taille = False
        pos = self._x.enlever_variables(positions, valeurs_x)
        if pos is not None:
            self._y.enlever_variables(pos)
        self._x._bloquer_modification_taille = True
        self._y._bloquer_modification_taille = True
        if self._liaison is not None and pos is not None:
            warnings.warn("Liaison maintenant invalide.")
            self._liaison = None
        return pos

    def ajouter_liaison(self, type_liaison: Type[_Liaison], borne_inf: float = None, borne_sup: float = None,
                        label: str = None, discontinuites_permises: bool = False, epsilon_continuite: float = None,
                        executer: bool = True, *execution_args, **execution_kwargs) -> None:
        """
        Méthode permettant d'ajouter une objet `_Liaison` pour "lier" les variables indépendantes aux variables
        dépendantes.
        :param type_liaison: type de `_Liaison` (peut être classe fille). Cet argument ne peut pas être une liaison
        mixte.
        :param borne_inf: float. Borne inférieure de la liaison à ajouter. `None` par défaut, ce qui signifie aucune
        borne (techniquement -infini).
        :param borne_sup:  float. Borne supérieure de la liaison à ajouter. `None` par défaut, ce qui signifie aucune
        borne (techniquement infini).
        :param label: str. Nom / label de la liaison. `None` par défaut, prend le nom par défaut du constructeur.
        :param discontinuites_permises: bool. Booléen spécifiant si on accepte les discontinuités dans la liaison à
        créer. `False` par défaut.
        :param epsilon_continuite: float. Dans le cas où les discontinuités ne sont pas permises, on utilise cette
        valeur comme écart absolu maximal qu'on considère pour la continuité. `None` par défaut (on ne considère aucun
        écart possible).
        :param executer: bool. Booléen spécifiant si on doit exécuter la liaison après sa création. Une exécution
        signifie qu'on fait, par exemple, la régression dans le cas de la régression. `True` par défaut.
        :param execution_args: arguments pour l'exécution (voir les méthodes propres aux liaisons pour plus de détails).
        :param execution_kwargs: mots-clés d'arguments pour l'exécution (voir les méthodes propres aux liaisons pour
        plus de détails).
        :return: Rien.
        """
        if type_liaison == LiaisonMixte:
            msg = "Spécifier une liaison mixte est ambiguë. Veuillez spécifier chaque liaison interne une à la " \
                  "fois ou utiliser `ajouter_liaisons` avec toutes ses liaisons internes."
            raise TypeError(msg)
        if borne_sup is None:
            borne_sup = np.inf
        if borne_inf is None:
            borne_inf = -np.inf
        args = (self._x <= borne_sup) & (self._x >= borne_inf)
        x_obs = self._x[args]
        y_obs = self._y[args]
        creation_args = (x_obs, y_obs, label)
        if label is None:
            creation_args = creation_args[:-1]
        liaison_courante = type_liaison(*creation_args)
        if executer:
            liaison_courante.executer(*execution_args, **execution_kwargs)
        if self._liaison is None:
            self._liaison = liaison_courante
        else:
            self._liaison = self._liaison.concatener_a_courant(discontinuites_permises, epsilon_continuite, None,
                                                               liaison_courante)

    def ajouter_liaisons(self, types_liaisons: Sequence[Type[_Liaison]], bornes_inf: Sequence[float],
                         bornes_sup: Sequence[float], labels: Sequence[str] = None,
                         discontinuites_permises: bool = False, epsilon_continuite: float = None,
                         executer: bool = True, execution_kwargs: dict = None) -> None:
        """
        Méthode permettant d'ajouter plusieurs liaisons à l'objet courant.
        :param types_liaisons: séquence. Types de liaisons. Un type ne peut être `_LiaisonMixte`.
        :param bornes_inf: séquence de floats. Séquence de bornes inférieures pour les liaisons. Dans le cas où un
        élément est `None`, on ne met pas de borne inférieure. La séquence doit être de même longueur que le nombre de
        liaisons.
        :param bornes_sup: séquence de floats. Séquence de bornes supérieures pour les liaisons. Dans le cas où un
        élément est `None`, on ne met pas de borne supérieure. La séquence doit être de même longueur que le nombre de
        liaisons.
        :param labels: séquence de strings. Noms/labels des liaisons. `None` par défaut (prend le nom par défaut
        du constructeur).
        :param discontinuites_permises: bool. Booléen spécifiant si on permet les discontinuités entre les liaisons.
        `False` par défaut.
        :param epsilon_continuite: float. Dans le cas où les discontinuités ne sont pas permises, on utilise cette
        valeur comme écart absolu maximal qu'on considère pour la continuité. `None` par défaut (on ne considère aucun
        écart possible).
        :param executer: bool. Booléen spécifiant si on doit exécuter la liaison après sa création. Une exécution
        signifie qu'on fait, par exemple, la régression dans le cas de la régression. `True` par défaut.
        :param execution_kwargs: dict. Dictionnaires dont les clés sont les arguments pour l'exécution et les valeurs
        sont les valeurs des arguments. `None` par défaut, donc on prend les arguments par défaut de l'exécution. Voir
        les méthodes spécifiques de l'exécution des liaisons pour plus de détails sur les mots-clés.
        :return: Rien.
        """
        if len(types_liaisons) != len(bornes_inf) or len(types_liaisons) != len(bornes_sup):
            raise ValueError("Il doit y avoir autant de liaisons que de bornes inférieures et supérieures")
        if labels is None:
            labels = [None] * len(types_liaisons)
        # Si on n'a pas fini de créer la LiaisonMixte finale, on ne se soucie pas des discontinuités possibles.
        discontinuites_permises_temp = True
        for i, type_liaison in enumerate(types_liaisons):
            if i == len(types_liaisons) - 1:
                discontinuites_permises_temp = discontinuites_permises
            self.ajouter_liaison(type_liaison, bornes_inf[i], bornes_sup[i], labels[i], discontinuites_permises_temp,
                                 epsilon_continuite, False)
        if executer:
            self._liaison.executer(execution_kwargs)

    def afficher_fonction(self, style: Style = Style(), ax: Axes = None):
        c = Courbe(self._x, self._y, label=self.label)
        c.afficher_courbe(False, False, False, style, self._x.label, self._y.label, ax)
