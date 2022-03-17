import numpy as np
from scipy.optimize import curve_fit
from typing import Union, Callable, Iterable, List, Tuple
from scipy.interpolate import interp1d
from warnings import warn


# TODO: Finir documentation

class _Liaison:
    # Variable statique stipulant si l'addition de _Liaison permet les discontinuités. Par défaut, c'est non.
    __add_permet_discontinuites__ = False

    # Variable statique stipulant quel est l'écart maximal permis entre l'intervalle observable des liaisons mixtes.
    __discontinuites_epsilon__ = None

    def __init__(self, x_obs: Iterable, y_obs: Iterable, label: str):
        """
        Constructeur de la classe de base `_Liaison`. Sert à représenter une "liaison" mathématique entre des variables
        indépendantes et des variables dépendantes. On peut penser à lier les variables à l'aide d'une régression
        polynomiale ou une interpolation quadratique, mais aussi à l'aide d'une fonction si on la connaît déjà.
        :param x_obs: Iterable. Variables indépendantes observées, c'est-à-dire celles qu'on utilise si on a à effectuer
        une régression ou interpolation.
        :param y_obs: Iterable. Variables dépendantes observées, c'est-à-dire celles qu'on utilise si on a à effectuer
        une régression ou interpolation. Doit être de même taille que x_obs.
        :param label: str. Nom donné à la liaison.
        """
        self._x_obs = np.ravel(x_obs).copy()
        self._y_obs = np.ravel(y_obs).copy()
        if len(self._x_obs) != len(self._y_obs):
            raise ValueError("Les observables (x et y) doivent avoir la même longueur.")
        self.label = label
        self._extrapolation_permise = False
        self._fonction = None

    def __call__(self, eval_x: Union[int, float, Iterable], afficher_warning: bool = True) -> np.ndarray:
        """
        Méthode permettant d'appeler l'objet courant avec des arguments pour évaluer la liaison à certain(s) point(s).
        Voir `evaluer_aux_points` pour plus d'information.
        :param eval_x: int, float ou iterable. Point(s) où évaluer la liaison.
        :param afficher_warning: bool. Booléen stipulant si on doit afficher les warnings. Ceux-ci peuvent être
        dérangeants à la longue, c'est pourquoi l'option de ne pas les afficher existe.
        :return: eval_y, l'évaluation de la liaison aux points eval_x.
        """
        eval_y = self.evaluer_aux_points(eval_x, afficher_warning)
        return eval_y

    def __add__(self, other) -> 'LiaisonMixte':
        """
        Méthode permettant de concatener deux _Liaison pour en former une dite "Mixte". Voir la classe `LiaisonMixte`
        pour plus d'informations. Voir `concatener_a_courant` pour l'algorithme de concaténation. Par défaut, on ne
        permet pas de discontinuités dans la liaison finale, mais ces paramètres peuvent être changés à l'aide des
        variables statiques `__add_permet_discontinuites__` et `__discontinuites_epsilon__`.
        :param other: object. Autre "objet" à concatener.
        :return: lm, un objet `LiaisonMixte` permettant de partitionner la manière dont on lie les variables.
        """
        lm = self.concatener_a_courant(_Liaison.__add_permet_discontinuites__, _Liaison.__discontinuites_epsilon__,
                                       None, other)
        return lm

    @property
    def info(self) -> dict:
        """
        Propriété donnant des informations sur l'objet courant. Pour cette classe mère, on ne retourne que la fonction
        de liaison. Les classes filles ajoutent plus d'éléments.
        :return: un dictionnaire d'informations. Pour l'instant, seulement la fonction de liaison est présente.
        """
        return {"fonction": self._fonction}

    @property
    def fonction(self) -> Callable:
        """
        Méthode retournant l'attribut de classe `_fonction`. Étant donné que cet attribut ne doit pas être modifié, on
        utilise la notion de @property pour s'assurer d'avoir un comportement "read only".
        :return: fct, attribut de classe `_fonction`.
        """
        fct = self._fonction
        return fct

    @property
    def x_obs(self) -> np.ndarray:
        """
        Méthode retournant l'attribut de classe `_x_obs` (copie). Étant donné que cet attribut ne doit pas être modifié,
        on utilise la notion de @property pour s'assurer d'avoir un comportement "read only". De plus, on en retourne
        une copie, car on peut tout de même changer les éléments internes sinon. Cela briserait complètement la
        cohérence de l'objet courant.
        :return: x_obs, copie de l'attribut de classe `_x_obs`.
        """
        x_obs = self._x_obs.copy()
        return x_obs

    @property
    def y_obs(self) -> np.ndarray:
        """
        Méthode retournant l'attribut de classe `_y_obs` (copie). Étant donné que cet attribut ne doit pas être modifié,
        on utilise la notion de @property pour s'assurer d'avoir un comportement "read only". De plus, on en retourne
        une copie, car on peut tout de même changer les éléments internes sinon. Cela briserait complètement la
        cohérence de l'objet courant.
        :return: y_obs, copie de l'attribut de classe `_y_obs`.
        """
        y_obs = self._y_obs.copy()
        return y_obs

    @property
    def pret(self) -> bool:
        """
        Méthode retournant si la liaison courante est prête à être utilisée. Pour qu'une liaison soit prête, il faut :
        - En cas de régression, celle-ci doit être effectuée, donc les paramètres optimisés pour pouvoir les utiliser.
        - En cas d'extrapolation, celle-ci doit être effectuée, donc les splines sont déterminées pour pouvoir les
        utiliser.
        :return: pret, booléen représentant si la liaison est prête ou non (i.e. on regarde si la fonction interne de
        liaison est spécifiée, donc différent de None).
        """
        pret = self._fonction is not None
        return pret

    def executer(self, *args, **kwargs) -> None:
        """
        Méthode permettant la cohésion entre les différents objets mathématiques pouvant être utilisés comme "liaison".
        Elle permet d'exécuter la préparation de la liaison, c'est-à-dire régresser ou interpoler. Étant donné que les
        deux cas sont assez différents, on utilise cette méthode qui généralise le concept et permet la compatibilité.
        :param args: Arguments à passer à la fonction plus spécifique. Par exemple, si l'objet courant est une liaison
        de type régression polynomiale, on doit spécifier obligatoirement le degré du polynôme.
        :param kwargs: Arguments (sous forme de mots-clés, par exemple degre=2). Voir `args` ci-dessus.
        :return: None
        """
        raise NotImplementedError("À implémenter dans les sous-classes.")

    def evaluer_aux_points(self, eval_x: Union[int, float, Iterable], afficher_warning: bool = True) -> np.ndarray:
        """
        Méthode permettant d'évaluer la liaison courante à certains points. La liaison doit être prête. Dans le cas où
        la liaison ne permet pas d'extrapoler, un warning peut être affiché.
        :param eval_x: int, float, Iterable. Point(s) où l'on veut évaluer la liaison.
        :param afficher_warning: bool. Booléen spécifiant si on veut afficher un warning dans le cas où une
        extrapolation est effectuée ou lorsqu'elle n'est pas permise et que des points à évaluer se retrouvent en-dehors
        de l'intervalle des valeurs observées. `True` par défaut.
        :return: eval, array NumPy. Évaluation de la liaison aux points voulus. Cas spéciaux:

        - Si un point est hors intervalle observé, mais que l'extrapolation est possible, on obtient la valeur
        extrapolée.
        -Si un point est hors intervalle observé, mais que l'extrapolation n'est pas possible, on obtien `nan`.
        -Si un point est dans l'intervalle observé, l'évaluation se fait normalement.
        """
        if not self.pret:
            msg = "Veuillez vous assurer que la fonction de liaison est prête. " \
                  "Veuillez faire la régression ou l'interpolation."
            raise ValueError(msg)
        positions_valides = self._validation_valeurs_a_evaluer(eval_x, afficher_warning)
        eval = np.where(positions_valides, self._fonction(eval_x), np.nan)
        return eval

    def concatener_a_courant(self, discontinuites_permises: bool = False, epsilon_continuite: float = None,
                             label: str = None, *autres_liaisons: '_Liaison') -> 'LiaisonMixte':
        """
        Méthode permettant de concatener une ou plusieurs liaisons à l'objet courant de sorte à obtenir une liaison par
        parties. Voir `concatener` pour plus d'informations sur l'algorithme de concaténation.
        :param discontinuites_permises: bool. Booléen spécifiant si les discontinuités sont permises. `False` par
        défaut.
        :param epsilon_continuite: float. Dans le cas où les discontinuités ne sont pas permises, on utilise cette
        valeur comme écart absolu maximal qu'on considère pour la continuité. `None` par défaut (on ne considère aucun
        écart possible).
        :param label: str. Nom donné à la nouvelle liaison mixte créée. `None` par défaut (valeur par défaut du
        constructeur).
        :param autres_liaisons: _Liaison. Autres objets _Liaison à concatener à celui courant.
        :return: lm, un objet `LiaisonMixte` permettant de partitionner la manière dont on lie les variables.
        """
        lm = _Liaison.concatener(discontinuites_permises, epsilon_continuite, label, self, *autres_liaisons)
        return lm

    @staticmethod
    def concatener(discontinuites_permises: bool = False, epsilon_continuite: float = None, label: str = None,
                   *liaisons: '_Liaison') -> 'LiaisonMixte':
        """
        Méthode (statique) permettant de concatener une ou plusieurs liaisons à l'objet courant de sorte à obtenir une
        liaison par parties.
        :param discontinuites_permises: bool. Booléen spécifiant si les discontinuités sont permises. `False` par
        défaut.
        :param epsilon_continuite: float. Dans le cas où les discontinuités ne sont pas permises, on utilise cette
        valeur comme écart absolu maximal qu'on considère pour la continuité. `None` par défaut (on ne considère aucun
        écart possible).
        :param label: str. Nom donné à la nouvelle liaison mixte créée. `None` par défaut (valeur par défaut du
        constructeur).
        :param autres_liaisons: _Liaison. Autres objets _Liaison à concatener à celui courant.
        :return: lm, un objet `LiaisonMixte` permettant de partitionner la manière dont on lie les variables.
        """
        liaisons_temp = []
        for liaison in liaisons:
            if isinstance(liaison, LiaisonMixte):
                liaisons_temp.extend(liaison._liaisons)
            else:
                liaisons_temp.append(liaison)
        if label is None:
            lm = LiaisonMixte(liaisons_temp, discontinuites_permises, epsilon_continuite)
        else:
            lm = LiaisonMixte(liaisons_temp, discontinuites_permises, epsilon_continuite, label)
        return lm

    def _validation_valeurs_a_evaluer(self, eval_x: Union[int, float, Iterable],
                                      afficher_warning: bool = True) -> np.ndarray:
        """
        Méthode utilitaire permettant de trouver les positions où les valeurs à évaluer sont valides. Par valide, on
        entend celles qui sont dans l'intervalle observé lorsque l'extrapolation n'est pas permise. Si elle est permise,
        toutes les valeurs sont valides.
        :param eval_x: int, float, Iterable. Point(s) dont on désire valider pour ensuite évaluer.
        :param afficher_warning: bool. Booléen spécifiant si on veut afficher un warning lorsque soit l'extrapolation
        est permise et il y a des points hors intervalle observé, soit lorsque l'extrapolation n'est pas permise et que
        des points sont aussi hors intervalle. Dans le premier cas, le warning sert à mettre en garde l'utilisateur
        contre des valeurs possiblement fausses, car l'extrapolation peut être mauvaise. Dans le second cas, on explique
        à l'utilisateur que des points ne seront pas considérés (evaluation à nan).
        :return: positions_valides, array NumPy de booléens ayant la même taille que `eval_x`. La valeur à l'indice i
        indique si `eval_x[i]` est un point valide.
        """
        eval_x = np.ravel(eval_x)
        min_x_obs = np.min(self._x_obs)
        max_x_obs = np.max(self._x_obs)
        positions_valides = (eval_x <= max_x_obs) & (eval_x >= min_x_obs)
        x_valides = np.sum(positions_valides)
        invalides = eval_x.shape != x_valides
        msg = "Extrapolation permise, donc les points en-dehors de l'intervalle `x_obs` seront extrapolés. " \
              "Veuillez considérer que les valeurs peuvent être loin de la vérité."
        if not self._extrapolation_permise:
            msg = "Extrapolation non permise, " \
                  "donc les points en-dehors de l'intervalle `x_obs` ne sont pas pris en compte."
        else:
            positions_valides = np.full_like(positions_valides, True, bool)
        if afficher_warning and invalides:
            warn(msg, UserWarning)
        return positions_valides


class LiaisonGenerale(_Liaison):

    def __init__(self, fonction: Callable, x_obs: Iterable = None, label: str = "Liaison générale"):
        # On fournit une fonction. Si x_obs est None, les bornes sont -inf à inf. y_obs n'est pas présent. On le calcule
        # dans le constructeur à partir de x_obs. Si x_obs est [-inf, inf], on fait quoi? Laisse []?
        raise NotImplementedError("Classe pas terminée")
        if not callable(fonction):
            msg = f"La fonction '{fonction}' n'est pas valide. Elle doit être appelable (c'est-à-dire pourvoir " \
                  f"utiliser '()' dessus)"
            raise TypeError(msg)
        if x_obs is None:
            x_obs = [-np.inf, np.inf]
            y_obs = []
        else:
            y_obs = fonction(x_obs)
        super(LiaisonGenerale, self).__init__(x_obs, y_obs, label)
        self._fonction = fonction

    def executer(self, *args, **kwargs):
        # Ne sert qu'à redéfinir pour ne pas avoir d'erreur.
        return None


class LiaisonMixte(_Liaison):
    # TODO: Faire de quoi pour LiaisonGenerale (avec variables [])
    def __init__(self, liaisons: Union[_Liaison, List[_Liaison], Tuple[_Liaison, ...]],
                 discontinuites_permises: bool = False, epsilon_continuite: float = None, label: str = "Liaison mixte",
                 permettre_overlap_bornes: bool = False):
        """
        Constructeur de la classe `LiaisonMixte`. Sert à modéliser des liaisons par parties, c'est-à-dire dont la
        fonction de liaison change au cours des valeurs observées.
        :param liaisons: _Liaison, List[_Liaison], Tuple[_Liaison, ...]. Objet(s) `_Liaison` qui sert/servent à
        partitionner.
        :param discontinuites_permises: bool. Booléen spécifiant si les discontinuités entre les valeurs observées sont
        permises. Voir `_analyse_continuite` pour voir ce qu'on entend par "discontinuité". `False` par défaut.
        :param epsilon_continuite: float. Écart maximal permis pour considérer la continuité des valeur observées. None`
        par défaut.
        :param label: str. Nom de la liaison mixte. "Liaison mixte" par défaut.
        :param permettre_overlap_bornes: bool. Booléen spécifiant si on permet l'overlap de bornes (donc overlap de
        liaisons). `False` par défaut.
        """
        x_obs = []
        y_obs = []
        self._pret = True
        if isinstance(liaisons, _Liaison):
            liaisons = [liaisons]
        for liaison in liaisons:
            if not isinstance(liaison, _Liaison):
                raise TypeError("Seulement les objets instances de '_Liaison' (ou ses dérivées) sont permis.")
            if not liaison.pret: self._pret = False
            x_obs.append(liaison.x_obs)
            y_obs.append(liaison.y_obs)
        self._liaisons = liaisons
        self._bornes, continuite, overlap = self._trouver_bornes_nested_lists(x_obs, epsilon_continuite, True)
        if not continuite and not discontinuites_permises:
            epsilon = epsilon_continuite if epsilon_continuite is not None else 0
            msg = f"Les liaisons ne sont pas continues selon une différence absolue maximale de {epsilon}. " \
                  f"Veuillez vous assurer qu'elles sont continues ou que les discontinuités sont permises."
            raise ValueError(msg)
        if overlap and not permettre_overlap_bornes:
            msg = "Il y a une superposition des bornes (donc des liaisons)."
            raise ValueError(msg)
        x_obs_concat = np.concatenate(x_obs)
        y_obs_concat = np.concatenate(y_obs)
        self._x_obs_all = x_obs
        self._y_obs_all = y_obs
        super(LiaisonMixte, self).__init__(x_obs_concat, y_obs_concat, label)
        self._fonction = self._creer_fonction(self._liaisons, self._bornes)

    @property
    def info(self) -> dict:
        """
        Propriété donnant des informations sur l'objet courant.
        :return: un dictionnaire d'informations. Pour cette classe, on retrouve les informations relatives à chaque
        liaison présente dans la liaison mixte courante. Voir `info` des autres classes.
        """
        info = {liaison.label: liaison.info for liaison in self._liaisons}
        return info

    @property
    def liaisons(self) -> Union[List[_Liaison], Tuple[_Liaison, ...]]:
        """
        Méthode permettant d'accéder aux liaisons internes de l'objet courant.
        :return: liaisons, objets `_Liaisons` internes à l'objet courant.
        """
        liaisons = self._liaisons[:]
        return liaisons

    @property
    def bornes(self) -> np.ndarray:
        """
        Méthode permettant d'accéder à une copie des bornes des liaisons internes.
        :return: bornes, array NumPy. Cet array est en deux dimensions. La première ligne indique les bornes minimales
        de chaque liaison, alors que la seconde ligne indique les bornes maximales.
        """
        bornes = self._bornes.copy()
        return bornes

    @property
    def pret(self) -> bool:
        """
        Méthode indiquant si la liaison mixte courante est prête à l'évaluation. Une liaison mixte prête indique que
        toutes les liaisons internes sont prêtes aussi.
        :return: pret, booléen indiquant si la liaison est prête.
        """
        pret = self._pret
        return pret

    def __len__(self) -> int:
        """
        Méthode permettant d'accéder au nombre de liaisons internes de l'objet courant. Permet d'utiliser len(self).
        :return: len_, entier indiquant le nombre de liaisons internes.
        """
        len_ = len(self._liaisons)
        return len_

    def __getitem__(self, item) -> Tuple[
        Union[_Liaison, List[_Liaison], Tuple[_Liaison, ...]], np.ndarray, Tuple[
            np.ndarray, np.ndarray]]:
        """
        Méthode permettant d'indexer l'objet courant.
        :param item: objet servant d'index.
        :return: valeur, tuple contenant (en ordre) : la/les liaisons indexée(s), la/les bornes associée(s), tuple des
        observables x et y associées.
        """
        tuple_obs = (self._x_obs_all[item], self._y_obs_all[item])
        bornes = self._bornes[:, item]
        if bornes.ndim != 2:
            bornes = bornes.reshape(2, 1)
        valeur = self._liaisons[item], bornes, tuple_obs
        return valeur

    @staticmethod
    def _trouver_bornes_nested_lists(listes: List[Iterable], epsilon_continuite: float = None,
                                     retour_all: bool = False) -> np.ndarray:
        """
        Méthode statique permettant d'extraire les bornes à partir d'une liste d'itérables de `_Liaison`.
        :param listes: list. Liste contenant une liste ou tuple d'objets `_Liaison`.
        :param epsilon_continuite: float. Écart maximal à considérer pour la continuité des bornes. `None` par défaut,
        donc aucun écart considéré.
        :param retour_all: bool. Booléen indiquant si on doit retourner un booléen indiquant si les liaisons
        internes sont continues, ainsi qu'un autre booléen indiquant s'il y a un overlap de bornes. `False` par défaut.
        :return: ret. On retourne toujours au moins les bornes. Ces bornes sont un array NumPy 2D dont la première ligne
        est les bornes minimales de chaque intervalle observé des liaisons internes. La seconde ligne est les bornes
        maximales. Dans le cas où `retour_all` est `True`, on retourne aussi `True` si les liaisons sont
        considérées continues. `False` sinon, ainsi que `True` s'il n'y a pas d'overlap de liaisons, `False` sinon.
        """
        bornes = []
        for iterable in listes:
            minimum = np.min(iterable)
            maximum = np.max(iterable)
            bornes.append((minimum, maximum))
        continuite, overlap = LiaisonMixte._analyse_continuite_et_overlap(bornes, epsilon_continuite)
        bornes = np.array(bornes).T
        ret = bornes
        if retour_all:
            ret = bornes, continuite, overlap
        return ret

    @staticmethod
    def _analyse_continuite_et_overlap(bornes: List[tuple], epsilon: float = None) -> Tuple[bool, bool]:
        """
        Méthode statique permettant d'évaluer la continuité et si les bornes (donc les liaisons) sont superposées.
        :param bornes: liste de tuples. Bornes des liaisons, dans le format `(min_i, max_i)`.
        :param epsilon: float. Argument spécifiant quelle est la tolérance de discontinuité. `None` par défaut, cela
        veut dire aucune tolérance.
        :return: un tuple de booléens. Le premier spécifie s'il y a continuité (`True` si c'est le cas), alors que le
        second spécifie s'il y a overlap (`True` si c'est le cas).
        """
        minimums = []
        maximums = []
        for b in bornes:
            minimums.append(b[0])
            maximums.append(b[1])
        minimums.sort()
        maximums.sort()
        if epsilon is None:
            continuite = np.array_equal(minimums[1:], maximums[:-1])
        else:
            continuite = np.allclose(minimums[1:], maximums[:-1], atol=epsilon, rtol=0)
        overlap = not all(np.greater_equal(minimums[1:], maximums[:-1]))
        return continuite, overlap

    @staticmethod
    def _creer_fonction(liaisons: Union[List[_Liaison], Tuple[_Liaison, ...]], bornes: np.ndarray) -> Callable:
        """
        Méthode statique permettant de créer la fonction de liaisons générale entre les liaisons présentes dans l'objet
        courant. Ainsi, si on a deux liaisons dont les bornes sont respectivement (0, 10) et (10, 20), si on veut
        évaluer à 11, la fonction créée ici saura quelle liaison utiliser.
        :param liaisons: liste, tuple. Liste ou tuple de liaisons présentes dans l'objet courant.
        :param bornes: array 2D de bornes. La première ligne correspond aux bornes minimales, alors que la seconde
        correspond au bornes maximales.
        :return: f, un objet "callable". Cet objet prend en argument un/des point(s) où évaluer la liaison mixte
        courante.
        """
        borne_min_arg = bornes[0].argmin()
        borne_min = bornes[0, borne_min_arg]
        borne_max_arg = bornes[1].argmax()
        borne_max = bornes[1, borne_max_arg]
        liaison_borne_min = liaisons[borne_min_arg]
        liaison_borne_max = liaisons[borne_max_arg]
        extrapolation_min = liaison_borne_min._extrapolation_permise
        extrapolation_max = liaison_borne_max._extrapolation_permise
        nb_bornes = bornes.shape[1]

        def f(x: Union[int, float, complex, Iterable]):
            x_s = np.ravel(x).copy()
            y_s = np.full_like(x_s, np.nan, dtype=float)
            if extrapolation_min:
                y_s = np.where(x_s < borne_min, liaison_borne_min(x_s), y_s)
            if extrapolation_max:
                y_s = np.where(x_s > borne_max, liaison_borne_max(x_s), y_s)
            for i in range(nb_bornes):
                b_min, b_max = bornes[:, i]
                temp = liaisons[i].evaluer_aux_points(x_s, False)
                y_s = np.where((x_s <= b_max) & (x_s >= b_min), temp, y_s)
            return y_s

        return f

    def executer(self, liaison_execution_args: dict = None) -> None:
        """
        Méthode permettant de rendre la liaison mixte courant prête à être utilisée. Par prête, on entend que les
        diverses liaisons internes, comme les régressions ou interpolations, sont prêtes, donc qu'on peut évaluer
        à différents points.
        :param liaison_execution_args: dict. Dictionnaire contenant les arguments ou "key words arguments" pour exécuter
        les diverses liaisons. Ce dictionnaire peut prendre deux formes. La première est dans le format
        `{0 :(argument 0, argument 1, ...), 2 : (argument 0, argument 1,...)}`, on a donc essentiellement un format
        clé : valeur où la clé est l'index de la liaison à l'interne et la valeur est un tuple d'arguments pour
        l'exécution de la liaison `i`. Faire attention à l'ordre des arguments. Si une exécution particulière n'a pas
        besoin d'arguments, ne pas l'inclure, comme dans l'exemple du format où la liaison `1` n'est pas dans le
        dictionnaire.

        Le second format est `{0 : {"nom_argument_0" : argument 0, ...}, 2 : {"nom_argument_1" : argument 1, ...}}`
        soit un format où la clé est encore l'index de la liaison à l'interne, mais la valeur est maintenant un
        dictionnaire. Les clés de ce dictionnaire interne sont les noms d'arguments et les valeurs sont les valeurs que
        prennent les arguments. Encore une fois, ne pas mettre d'élément dans le dictionnaire si une liaison n'a pas
        besoin d'arguments spécifiques pour son exécution.

        Important : on peut mélanger les deux formats.

        Une fois cette méthode appelée, on peut utiliser la liaison mixte courante pour évaluer à certains points.
        :return: Rien
        """
        # liaison_execution_args, dictionnaire d'arguments pouvant avoir deux formes:
        # Forme 1: {0:(argument 0, argument 1, ...), 2:(argument 0, argument 1,...)}
        # où clé est index liaison. Si pas besoin d'argument, pas inclure, comme ^, 1 n'est pas présent
        # Forme 2: {0:{"nom_argument_0":argument 0, ...}, 2:{"nom_argument_1": argument 1, ...}}
        # version "kwargs", où encore une fois clé = index liaison. Si pas d'argument, ne pas le mettre.
        # On peut mixer les deux formats.
        if liaison_execution_args is None:
            liaison_execution_args = dict()
        for i, liaison in enumerate(self._liaisons):
            arguments = liaison_execution_args.get(i, tuple())
            if liaison.pret and arguments == tuple():
                continue
            if isinstance(arguments, dict):
                liaison.executer(**arguments)
            else:
                liaison.executer(*arguments)
        self._pret = True


class _InterpolationBase(_Liaison):

    def __init__(self, x_obs: Iterable, y_obs: Iterable, label: str):
        super(_InterpolationBase, self).__init__(x_obs, y_obs, label)

    @property
    def info(self) -> dict:
        return {"fonction interpolation": self._fonction}

    def interpolation(self, permettre_extrapolation: bool = False):
        raise NotImplementedError("Doit être implémentée dans les sous-classes.")

    def executer(self, permettre_extrapolation: bool = False):
        return self.interpolation(permettre_extrapolation)


class InterpolationLineaire(_InterpolationBase):

    def __init__(self, x_obs: Iterable, y_obs: Iterable, label: str = "Interpolation linéaire"):
        super(InterpolationLineaire, self).__init__(x_obs, y_obs, label)

    def interpolation(self, permettre_extrapolation: bool = False):
        fill_value = np.nan
        if permettre_extrapolation:
            fill_value = "extrapolate"
            self._extrapolation_permise = True
        self._fonction = interp1d(self._x_obs, self._y_obs, "linear", fill_value=fill_value)
        return self.fonction


class InterpolationQuadratique(_InterpolationBase):

    def __init__(self, x_obs: Iterable, y_obs: Iterable, label: str = "Interpolation quadratique"):
        super(InterpolationQuadratique, self).__init__(x_obs, y_obs, label)

    def interpolation(self, permettre_extrapolation: bool = False):
        fill_value = np.nan
        if permettre_extrapolation:
            fill_value = "extrapolate"
            self._extrapolation_permise = True
        self._fonction = interp1d(self._x_obs, self._y_obs, "quadratic", fill_value=fill_value)
        return self.fonction


class InterpolationCubique(_InterpolationBase):

    def __init__(self, x_obs: Iterable, y_obs: Iterable, label: str = "Interpolation cubique"):
        super(InterpolationCubique, self).__init__(x_obs, y_obs, label)

    def interpolation(self, permettre_extrapolation: bool = False):
        fill_value = np.nan
        if permettre_extrapolation:
            fill_value = "extrapolate"
            self._extrapolation_permise = True
        self._fonction = interp1d(self._x_obs, self._y_obs, "cubic", fill_value=fill_value)
        return self.fonction


class _RegressionBase(_Liaison):

    def __init__(self, x_obs: Iterable, y_obs: Iterable, label: str):
        super(_RegressionBase, self).__init__(x_obs, y_obs, label)
        self._reg_info = {"paramètres optimisés": None, "sigma paramètres": None, "SSe": None,
                          "fonction": None}

    @property
    def info(self) -> dict:
        return self._reg_info

    def regression(self, *args, **kwargs):
        raise NotImplementedError("Doit être implémentée dans les sous-classes.")

    def executer(self, *args, **kwargs):
        return self.regression(*args, **kwargs)

    @classmethod
    def generer_sigma_parametres(cls, matrice_covariance: np.ndarray):
        return np.sqrt(np.diag(matrice_covariance))


class RegressionPolynomiale(_RegressionBase):
    __warning_covariance_matrice__ = False

    def __init__(self, x_obs: Iterable, y_obs: Iterable, label: str = "Regression polynomiale"):
        super(RegressionPolynomiale, self).__init__(x_obs, y_obs, label)
        self._reg_info["degré"] = None

    def regression(self, degre: int, permettre_extrapolation: bool = False):
        self._extrapolation_permise = permettre_extrapolation
        V = self.generer_matrice_Vandermonde(self._x_obs, degre)
        coefs_estimes, autres = np.polynomial.polynomial.polyfit(self._x_obs, self._y_obs, degre, full=True)
        coefs_estimes = coefs_estimes[::-1]
        SSe = autres[0]
        if len(SSe) == 0:
            msg = "Impossible de calculer la somme des erreurs au carré (SSe), " \
                  "donc impossible d'estimer la matrice de covariance"
            warn(msg, RuntimeWarning)
            SSe = None
            sigma_coefs = np.full_like(coefs_estimes, np.nan, dtype=float)
        else:
            SSe = SSe.item()
            cov = self.generer_matrice_covariance(V, SSe, len(self._x_obs), degre)
            sigma_coefs = self.generer_sigma_parametres(cov)
        self._fonction = lambda x: np.polynomial.polynomial.polyval(x, coefs_estimes[::-1])
        self._reg_info.update({"paramètres optimisés": coefs_estimes, "sigma paramètres": sigma_coefs, "SSe": SSe,
                               "fonction": self._fonction, "degré": degre})
        return coefs_estimes.copy(), sigma_coefs.copy(), SSe

    @classmethod
    def generer_matrice_Vandermonde(cls, x: np.ndarray, degre: int, puissance_croissante: bool = False):
        return np.vander(x, degre + 1, puissance_croissante)

    @classmethod
    def generer_matrice_covariance(cls, matrice_Vandermonde: np.ndarray, SSe: float, nb_obs_x: int, degre: int,
                                   afficher_warning: bool = True):
        if afficher_warning and not cls.__warning_covariance_matrice__:
            msg = "La matrice de covariance n'est peut-être pas exacte. " \
                  "Elle est toutefois proportionnelle à un facteur multiplicatif " \
                  "près et peut servir de bonne d'approximation."
            warn(msg, UserWarning)
            cls.__warning_covariance_matrice__ = True
        denom = nb_obs_x - (degre + 1)
        if denom <= 0:
            raise ValueError("Il n'y a pas assez de points pour bien regresser. Veuillez en utiliser plus.")
        sigma = SSe / denom
        cov = sigma * np.linalg.inv(np.dot(matrice_Vandermonde.T, matrice_Vandermonde))
        return cov


class RegressionGenerale(_RegressionBase):

    def __init__(self, x_obs: Iterable, y_obs: Iterable, label: str = "Regression"):
        super(RegressionGenerale, self).__init__(x_obs, y_obs, label)

    def regression(self, fonction: Callable, estimation_initiale_parametres: tuple = None,
                   limites_parametres: tuple = None, permettre_extrapolation: bool = False):
        self._extrapolation_permise = permettre_extrapolation
        if limites_parametres is None:
            limites_parametres = (-np.inf, np.inf)
        parametres, cov = curve_fit(fonction, self._x_obs, self._y_obs, estimation_initiale_parametres,
                                    bounds=limites_parametres)
        sigma_params = self.generer_sigma_parametres(cov)
        self._reg_info.update({"paramètres optimisés": parametres, "sigma paramètres": sigma_params, "SSe": None,
                               "fonction": fonction})
        self._fonction = lambda x: fonction(x, *parametres)
        return parametres.copy(), sigma_params.copy(), None

    # ATTENTION: Les méthodes suivantes permettent de modéliser des fonctions communes, donc on peut les utiliser pour
    # effectuer des régressions. Or, il se peut que la régression soit vraiment mauvaise à cause des paramètres.
    # Par exemple, si on sait que nos données suivent une loi exponentielle négative définie par
    # f(x) = A*e^(-a*x)
    # Il est préférable de se définir une telle fonction au lieu d'utiliser celle définie plus bas. La méthode curve_fit
    # est itérative, donc inexacte, et si on met des paramètres qui ne devraient pas avoir d'influence, on peut se
    # retrouver totalement perdu avec une régression qui n'a aucun sens.

    @staticmethod
    def fonction_gaussienne(x: np.ndarray, a: float, mu: float, sigma: float, b: float):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b

    @staticmethod
    def fonction_exponentielle(x: np.ndarray, a: float, b: float, c: float, d: float):
        return a * np.exp(x * b + c) + d

    @staticmethod
    def fonction_sinus(x: np.ndarray, a: float, b: float, c: float, d: float):
        return a * np.sin(x * b + c) + d

    @staticmethod
    def fonction_ln(x: np.ndarray, a: float, b: float, c: float, d: float):
        return a * np.log(x * b + c) + d
