from typing import Union, Callable, Tuple, Iterable, Sized, Type, Sequence
from src.regression_interpolation import _Liaison, LiaisonMixte
from numbers import Integral
import numpy as np
from numbers import Number


class _Variables:
    __label__ = "Variables (base)"

    def __init__(self, valeurs: Iterable, bloquer_ajout_modification_taille: bool = True,
                 label: str = None):
        # TODO: Constructeur copie avec __new__ ?
        """
        Constructeur de la classe _Variables.
        :param valeurs: Iterable. Conteneur initial des valeurs de la variable à créer.
        :param bloquer_ajout_modification_taille: bool. Booléen spécifiant si on veut bloquer la modification de la
        taille de l'objet créé. Dans la très grande majorité des cas, ce booléen doit être `True` pour garder le concept
        cohérent (en effet, avoir un couple VariablesDependantes et VariablesIndependantes de tailles différentes n'est
        pas acceptable et cohérent). Par contre, on laisse la possibilité de le mettre à `False` (modification de la
        taille permise) pour des cas possibles où ça devrait être permis.
        """
        self.__cls = type(self)
        bloque_temp = bloquer_ajout_modification_taille
        label_temp = label if label is not None else self.__cls.__label__
        if isinstance(valeurs, _Variables):
            # Genre de "constructeur par copie".
            # Peut paraître inutile, mais sauve la construction d'un autre array...!
            bloque_temp = valeurs.modification_taille_est_bloquee
            self._valeurs = valeurs.valeurs
            label_temp = valeurs.label
        else:
            self._valeurs = np.ravel(valeurs).copy()  # Sinon on modifie l'original aussi.
        dtype = self._valeurs.dtype
        if not issubclass(dtype.type, Number):
            raise TypeError(f"Le type de données '{dtype}' n'est pas supporté.")
        self._iteration = 0
        self._len = len(self._valeurs)
        self._bloquer_modifcation_taille = bloque_temp
        # ^ Ne pas accéder à cette variable le plus possible, garde le concept cohérent.
        self.label = label_temp

    @property
    def cls(self) -> type:
        """
        Propriété non modificable correspondant à la classe de l'objet courant. Peut être _Variables, mais aussi une de
        ses classes dérivées.
        :return: cls, type. Classe de l'objet courant.
        """
        cls = self.__cls
        return cls

    @property
    def valeurs(self) -> np.ndarray:
        """
        Propriété non modifiable de l'attribut correspondant aux valeurs internes de l'objet courant.
        :return: valeurs, np.ndarray. Array NumPy correspondant aux valeurs internes.
        """
        valeurs = self._valeurs.copy()
        return valeurs

    @property
    def modification_taille_est_bloquee(self) -> bool:
        """
        Propriété non modifiable de l'attribut spécifiant si l'on peut modifier la taille de l'objet courant.
        :return: bloquer, bool. Booléen spécifiant si la taille de l'objet courant est fixe (`True`) ou non (`False`).
        """
        bloquer = self._bloquer_modifcation_taille
        return bloquer

    def __getitem__(self, item: Union[int, slice, Iterable]) -> Union[int, float, complex, np.ndarray]:
        """
        Méthode permettant d'accéder à une ou plusieurs valeurs de l'objet courant à l'aide de la syntaxe self[item],
        où item est la clé (paramètre de la fonction présente).
        :param item: int, slice, Iterable. Clé permettant d'accéder à l'élément (ou les éléments) qu'on veut avoir.
        :return: vals: int, float, complex ou np.ndarray. Valeur(s) obtenue(s) par l'accès avec la clé. Si la clé
        retourne un scalaire, ce scalaire est retourné seul. Si la clé retourne un itérable (sous array des valeurs
        courantes), on retourne ce sous-array sous la forme d'une instance _Variables (ou ses dérivées).
        """
        vals = self._valeurs[item]
        if not isinstance(vals, Number):
            vals = self.__cls(vals)
        return vals

    def __setitem__(self, cle: Union[int, slice, Iterable], valeur: Union[int, float, complex, Iterable]) -> None:
        """
        Méthode permettant de modifier une ou plusieurs valeurs de l'objet courant à l'aide de la syntaxe
        self[cle] = valeur, où cle et valeur sont des argument de la fonction présente.
        :param cle: int, slice, Iterable. Clé permettant d'accéder à l'élément (ou les éléments) qu'on veut modifier.
        :param valeur: int, float, complex, Iterable. Nouvelle(s) valeur(s) qu'on veut assigner. Si cet argument ainsi
        que la clé sont scalaires, l'élément à la position `cle` sera changé à cette valeur. Si la clé n'est pas
        scalaire (slice ou itérable), tous les éléments accédés par la clé seront changés à cette valeur. Finalement, si
        cet argument est itérable, la clé doit être soit une slice ou aussi itérable, ils doivent avoir la même taille.
        :return: None. Rien n'est retourné.
        """
        valeur = np.ravel(valeur).copy()
        valeur_type = valeur.dtype.type
        self._type_cast_priorite(valeur_type)
        self._valeurs[cle] = valeur

    def __len__(self) -> int:
        """
        Méthode permettant d'obtenir la taille de l'objet courant. Permet d'utiliser la fonction builtin `len`.
        :return: len_, int. Entier représentant la taille de l'objet courant.
        """
        len_ = self._len
        return len_

    def __iter__(self):
        """
        Méthode permettant (avec `__next__`) de rendre la classe courante (et l'objet courant) itératif en rendant
        possible l'utilisation de la fonction builtin `iter`. Permet de réinitialiser l'itération courante à 0 pour
        pouvoir itérer sur l'objet courant à nouveau une fois qu'il a déjà été parcouru.
        :return: self, l'objet courant.
        """
        self._iteration = 0
        return self

    def __next__(self) -> Union[int, float, complex]:
        """
        Méthode servant (avec `__iter__`) à rendre la classe courante (et l'objet courant) itératif en rendant possible
        l'utilisation de la fonction builtin `next`.
        :return: valeur: int, float, complex. Valeur suivante dans l'itération courante.
        """
        if self._iteration < self._len:
            valeur = self._valeurs[self._iteration]
            self._iteration += 1
            return valeur
        raise StopIteration("Itérateur épuisé!")

    def __copy__(self):
        """
        Méthode permettant de faire la copie en surface à l'aide du module `copie` et de sa fonction `copy`.
        :return: copie, nouvel objet copié sur celui courant (voir la méthode `copie`)
        """
        copie = self.copie()
        return copie

    def __deepcopy__(self, *args, **kwargs):
        """
        Méthode permettant de faire la copie en profondeur à l'aide du module `copie` et de sa fonction `deepcopy`.
        :param args: Arguments inutilisés, mais servant à la compatibilités
        :param kwargs: Arguments inutilisés, mais servant à la compatibilités
        :return: copie, nouvel objet copié sur celui courant (voir la méthode `copie`)
        """
        copie = self.copie()
        return copie

    def __eq__(self, other: object) -> Union[bool, Iterable[bool]]:
        """
        Méthode permettant de déterminer si un objet autre quelconque est égal à celui courant. L'égalité de plusieurs
        valeurs entrant en jeu, il est (plus souvent qu'autrement) ambiguë d'avoir une valeur booléenne unique. Dans ces
        cas communs, un array de booléens sera retourné. Cet array est la comparaison terme à terme entre l'objet
        courant et l'autre objet.
        :param other: object. Objet quelconque dont on veut évaluer la non égalité avec l'objet courant. Il existe
        quelques cas:

        - S'il s'agit d'un scalaire, on compare ce scalaire à chaque terme de l'objet courant. Cela donne un array
        dont l'entrée à l'index `i` indique si le scalaire est égal à l'entrée à l'index `i` de l'objet courant.

        - S'il s'agit d'un itérable, on compare premièrement si la taille est cohérente avec celle de l'objet courant.
        Si ce n'est pas le cas, `False` est retourné (booléen unique). S'ils ont la même taille, on effectue la
        comparaison terme à terme. Cela donne un array dont l'entrée à l'index `i` indique si l'entrée à l'index `i` de
        l'autre objet est égale à l'entrée à l'index `i` de l'objet courant.

        - S'il s'agit d'un autre type d'objet, ils ne sont pas supportés et `False` est retourné (booléen unique).

        :return: eq: bool ou Iterable[bool]. Array de booléen ou booléen unique, selon la valeur et le type de
        l'argument other.
        """
        # On ne peut pas simplement utiliser la comparaison == de NumPy, car si les tailles sont différentes, il y a
        # un DeprecationWarning et pourra éventuellement lever une exception.
        eq = False
        len_courant = self._len
        vals = other
        if isinstance(other, _Variables):
            vals = other._valeurs
        if isinstance(other, Sized):
            len_other = len(other)
            if len_other == len_courant:
                eq = vals == self._valeurs
        elif isinstance(other, Number):
            eq = self._valeurs == other
        return eq

    def __ne__(self, other: object) -> Union[bool, Iterable[bool]]:
        """
        Méthode permettant de déterminer si un objet autre quelconque est différent de celui courant. Voir `__eq__`
        :param other: object. Objet quelconque dont on veut évaluer la non égalité avec l'objet courant. Voir `__eq__`.
        :return: ne: bool ou Iterable[bool]. Inverse logique de `__eq__`.
        """
        ret_inv = self == other
        if isinstance(ret_inv, bool):
            ne = not ret_inv
        else:
            ne = ~ret_inv
        return ne

    def __neg__(self):
        """
        Méthode permettant d'effectuer la négation de chaque terme de l'objet courant, donc de faire -self.
        :return: neg, nouvel objet dont les valeurs sont multipliées par -1 (inverse d'addition).
        """
        neg = self.__cls(-self._valeurs, self.label)
        return neg

    def __pos__(self):
        """
        Méthode permettant d'effectuer +self (opérateur unaire). Techniquement, ne change absolument rien.
        :return: pos, nouvel objet identique à celui courant.
        """
        pos = self.__cls(+self._valeurs, self.label)
        return pos

    def __add__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant d'additionner un objet quelconque à celui courant, donc self + other. Cette addtion est
        équivalente à other + self (`__radd__`).
        :param other: int, float, complex, Iterable. Objet autre qu'on veut additionner à celui courant. Si c'est un
        scalaire, on additionne tous les termes de l'objet courant à ce scalaire. Si c'est un itérable, on additionne
        chaque terme de l'objet courant par le terme correspondant de l'objet autre, ils doivent donc avoir la même
        taille.
        :return: add, nouvel objet dont les valeurs sont celles courantes à lesquelles on a additionné un autre objet.
        """
        add = np.add(self, other)
        add.label = self.label
        return add

    def __radd__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer other + self, où self est l'objet courant et où other est un autre objet
        quelconque. Voir `__add__`.
        :param other: int, float, complex, Iterable. Objet à additionner à l'objet courant. Voir `__add__`.
        :return: radd, nouvel objet dont les valeurs sont celles courantes additionnées de l'aute objet quelconque.
        """
        radd = self + other
        return radd

    def __sub__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de soustraire un objet quelconque à l'objet courant, donc self - other.
        :param other: int, float, complex, Iterable. Objet qu'on soustrait à l'objet courant. Si c'est un scalaire, on
        soustrait chaque élément courant par ce scalaire. Si c'est un itérable, on soustrait chaque terme courant par
        l'élément correspondant de l'objet courant, ils doivent donc avoir la même taille.
        :return: sub, nouvel objet dont les valeurs sont celles courantes à lesquelles on a soustrait l'objet
        quelconque.
        """
        sub = np.subtract(self, other)
        sub.label = self.label
        return sub

    def __rsub__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de caluler la soustraction other - self, où self est l'objet courant et où other est un autre
        objet quelconque.
        :param other: int, float, complex, Iterable. Objet auquel on soustrait l'objet courant. Dans le cas où c'est un
        scalaire, on soustrait à ce dernier chaque élément de l'objet courant un après l'autre (séparément). Dans le cas
        où c'est un itérable, on effectue la soustraction terme à terme entre l'objet quelconque et l'objet courant, ils
        doivent donc avoir la même taille.
        :return: rsub, nouvel objet dont les éléments sont ceux (ou celui) de l'objet quelconque moins ceux de l'objet
        courant.
        """
        rsub = np.subtract(other, self)
        rsub.label = self.label
        return rsub

    def __mul__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de multipliser un objet quelconque par celui courant, donc self * other. La multiplication
        étant "terme à terme", elle est donc équivalente à other * self (`__rmul__`).
        :param other: int, float, complex ou Iterable. Objet quelconque qu'on multiplie à celui courant. S'il s'agit
        d'un scalaire, on multiplie chaque terme de l'objet courant par ce scalaire. S'il s'agit d'un itérable, on
        effectue la multiplication terme à terme, donc l'objet courant et celui multiplicateur doivent avoir la même
        taille.
        :return: rmul, nouvel objet composé de la multiplication de l'objet courant par un autre.
        """
        mul = np.multiply(self, other)
        mul.label = self.label
        return mul

    def __rmul__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de multipliser un objet quelconque par celui courant, donc other * self. Voir `__mul__`.
        :param other: int, float, complex ou Iterable. Objet quelconque qu'on multiplie à celui courant. Voir `__mil__`.
        :return: rmul, nouvel objet composé de la multiplication de l'objet courant par un autre. Voir `__mul__`.
        """
        rmul = self * other
        return rmul

    def __truediv__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de diviser l'objet courant par un autre objet quelconque, donc de faire self / other, où
        self est l'objet courant et other est un autre objet (le diviseur).
        :param other: int, float, complex ou Iterable. Objet servant de diviseur à celui courant. Dans le cas où c'est
        un scalaire, chaque élément de l'objet courant sera divisé par ce scalaire. Dans le cas où c'est un itérable,
        chaque élément de l'objet courant sera divisé par celui correspondant dans l'objet diviseur, ils doivent donc
        avoir la même taille.
        :return: div, nouvel objet dont les valeurs sont celles courantes divisées par other.
        """
        div = np.true_divide(self, other)
        div.label = self.label
        return div

    def __rtruediv__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant d'effectuer other / self, où self est l'objet courant et other est un autre objet quelconque.
        Voir `__truediv__`.
        :param other: int, float, complex ou Iterable. Terme qui sera divisé par l'objet courant. Dans le cas où c'est
        un scalaire, on le divisera par chaque élément de l'objet courant, un après l'autre. Dans le cas où c'est un
        itérable, on divisera chaque élément par celui correspondant dans l'objet courant, ils doivent donc être de même
        taille.
        :return: rdiv, nouvel objet dont les valeurs
        """
        rdiv = np.true_divide(other, self)
        rdiv.label = self.label
        return rdiv

    def __pow__(self, power: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer l'objet courant à une puissance spécifiée par power. Cela permet d'utiliser la
        syntaxe self ** power, où self est l'objet courant (la base) et power est un autre objet servant d'exposant.
        :param power: int, float, complex, Iterable. Objet servant d'exposant (ou puissance) qu'on applique sur l'objet
        courant. Dans le cas où c'est un scalaire, on l'applique sur tous les éléments courants. Dans le cas où c'est
        un itérable, on l'applique élément par élément sur l'objet courant, ils doivent donc être de même taille.
        :return: pow, nouvel objet composé des valeurs de l'objet courant dont on a élevé à la puissance power.
        """
        power = np.ravel(power)
        t_pow = power.dtype.type
        t_self = self._valeurs.dtype.type
        if issubclass(t_pow, Integral) and issubclass(t_self, Integral):
            # Peut-être overkill?
            pow_ = np.power(self, power)
        else:
            pow_ = np.float_power(self, power)
        pow_.label = self.label
        return pow_

    def __rpow__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer other ** self, où self est l'objet courant servant comme exposant et other est un
        autre objet servant comme base. Voir `__pow__`.
        :param other: int, float, complex, Iterable. Objet quelconque servant de base. Si c'est un scalaire, ce-dernier
        sera elevé à la puissance de chaque élément de l'objet courant. Si c'est un itérable, chaque élément sera élevé
        à la puissance de l'élément de l'objet courant correspondant (de manière terme à terme), ils doivent donc avoir
        la même taille.
        :return: rpow, un nouvel objet dont les valeurs sont other ** self
        """
        power = np.ravel(other)
        t_pow = power.dtype.type
        t_self = self._valeurs.dtype.type
        if issubclass(t_pow, Integral) and issubclass(t_self, Integral):
            # Peut-être overkill?
            rpow = np.power(power, self)
        else:
            rpow = np.float_power(power, self)
        rpow.label = self.label
        return rpow

    def __floordiv__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer l'expression self // other, où self est l'objet courant et other est le diviseur.
        Cette expression est la division entière (ou euclidienne), soit trouver le quotient (et reste, mais celui-ci
        n'est pas donné dans le cas présent) d'une division. Le résultat est un entier. Par exemple, 5 // 2 = 2
        (reste 1), car 2 * 2 + 1 = 5.
        :param other: int, float, complex ou Iterable. Objet diviseur. S'il s'agit d'un scalaire
        on fait la division entière avec ce scalaire pour chaque élément de l'objet courant. S'il s'agit d'un itérable,
        chaque élément de ce-dernier sera utilisé pour diviser le terme associé de l'objet courant (calcul terme à
        terme).
        :return: floordiv, un nouvel objet dont les valeurs sont celles courantes divisées (entièrement) par other.
        """
        floordiv = np.floor_divide(self, other)
        floordiv.label = self.label
        return floordiv

    def __rfloordiv__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer l'expression other // self, où self est l'objet courant et other est le diviseur.
        Voir `__floordiv__`.
        :param other: int, float, complex ou Iterable. Objet qu'on divisera par celui courant. S'il s'agit d'un scalaire
        on fait la division entière sur ce scalaire pour chaque élément de l'objet courant. S'il s'agit d'un itérable,
        on fait la division terme à terme en utilisant l'objet courant comme diviseur.
        :return: floordiv, un nouvel objet dont les valeurs sont other // self
        """
        floordiv = np.floor_divide(other, self)
        floordiv.label = self.label
        return floordiv

    def __mod__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer l'expression self % other, où self est l'objet courant et other est le modulo.
        Le modulo est le reste de la division entière. Par exemple, 5%2 donne 1, car 5//2 donne 2 et il reste 1. Dans le
        cas présent, comme l'objet courant contient plusieurs valeurs, on effectue le modulo terme à terme.
        :param other: int, float, complex, Iterable. Modulo à utiliser. Dans le cas où c'est un scalaire, on applique
        le modulo sur chaque terme. Dans le cas où c'est un itérable, on effectue le module terme à terme, donc les
        dimensions doivent être les mêmes.
        :return: mod, un nouvel objet dont les valeurs ont subies le modulo.
        """
        mod = np.remainder(self, other)
        mod.label = self.label
        return mod

    def __rmod__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer l'expression other % self, où self est l'objet courant et other est un autre
        objet. Voir `__mod__`
        :param other: int, float, complex, Iterable. Objet sur lequel appliquer le modulo. Dans le cas où c'est un
        scalaire, applique sur ce scalaire le modulo de chaque terme de l'objet courant. Dans le cas où c'est un
        itérable, on effectue le module terme à terme, donc les dimensions doivent être les mêmes.
        :return: mod, un nouvel objet dont les valeurs sont other % self.
        """
        mod = np.remainder(other, self)
        mod.label = self.label
        return mod

    def __abs__(self):
        """
        Méthode permettant de calculer la valeur absolue, définie comme -nombre si nombre est négatif, sinon donne
        nombre. Dans le cas présent, comme il s'agit de plusieurs valeurs contenu dans un array, on effectue l'opération sur
        chaque élément un après l'autre. Permet d'utiliser la fonction builtin `abs`.
        :return: absolu. Nouvel objet avec les valeurs abosulue.
        """
        absolu = np.abs(self)
        absolu.label = self.label
        return absolu

    def __ceil__(self):
        """
        Méthode permettant de calculer la fonction mathématique "ceil" (plafond) définie comme l'entier supérieur
        (ou égal) le plus proche d'un nombre quelconque. Par exemple, floor(1.01) donne 2, alors que floor(1) donne 1.
        Dans le cas présent, comme il s'agit de plusieurs valeurs contenu dans un array, on effectue l'opération sur
        chaque élément un après l'autre. Permet d'utiliser la fonction `ceil` du module `math`.
        :return: ceil. Nouvel objet avec les valeurs arrondies selon la logique de la fonction plafond.
        """
        ceil = np.ceil(self)
        ceil.label = self.label
        return ceil

    def __floor__(self):
        """
        Méthode permettant de calculer la fonction mathématique "floor" (plancher) définie comme l'entier inférieur
        (ou égal) le plus proche d'un nombre quelconque. Par exemple, floor(1.9) donne 1, alors que floor(2) donne 2.
        Dans le cas présent, comme il s'agit de plusieurs valeurs contenu dans un array, on effectue l'opération sur
        chaque élément un après l'autre. Permet d'utiliser la fonction `floor` du module `math`.
        :return: floor. Nouvel objet avec les valeurs arrondies selon la logique de la fonction plancher.
        """
        floor = np.floor(self)
        floor.label = self.label
        return floor

    def __round__(self, n: int = None):
        """
        Méthode permettant d'arrondir l'objet courant selon un nombre de décimales. Permet la compatibilité avec la
        méthode builtin `round`.
        :param n: int. Nombre de décimales. Par défaut, vaut None, donc 0 (on arrondie à l'unité).
        :return: arrondi. Nouvel objet dont les valeurs sont arrondies.
        """
        if n is None:
            n = 0
        arrondi = np.round(self, n)
        arrondi.label = self.label
        return arrondi

    def __gt__(self, other: Union[int, float, complex, Iterable]) -> Iterable[bool]:
        """
        Méthode permettant de comparer l'objet courant à un autre selon la logique "plus grand que", soit
        self > other.
        :param other:int, float, complex, Iterable. Autre objet à comparer. Si scalaire, on compare toutes les entrées
        courante à ce scalaire. Si itérable, doit être de même dimension que l'objet courant. Dans ce cas, on effectue
        la comparaison terme à terme.
        :return:gt, array de booléens. Comparaison de l'objet courant avec l'autre objet selon la logique
        "plus grand que".
        """
        if isinstance(other, _Variables):
            other = other._valeurs
        gt = np.greater(self._valeurs, other)
        return gt

    def __le__(self, other: Union[int, float, complex, Iterable]) -> Iterable[bool]:
        """
        Méthode permettant de comparer l'objet courant à un autre selon la logique "plus petit ou égal à", soit
        self <=other.
        :param other:int, float, complex, Iterable. Autre objet à comparer. Si scalaire, on compare toutes les entrées
        courante à ce scalaire. Si itérable, doit être de même dimension que l'objet courant. Dans ce cas, on effectue
        la comparaison terme à terme.
        :return:le, array de booléens. Comparaison de l'objet courant avec l'autre objet selon la logique
        "plus petit ou égal à".
        """
        le = ~(self > other)
        return le

    def __ge__(self, other: Union[int, float, complex, Iterable]) -> Iterable[bool]:
        """
        Méthode permettant de comparer l'objet courant à un autre selon la logique "plus grand ou égal à", soit
        self >=other.
        :param other:int, float, complex, Iterable. Autre objet à comparer. Si scalaire, on compare toutes les entrées
        courante à ce scalaire. Si itérable, doit être de même dimension que l'objet courant. Dans ce cas, on effectue
        la comparaison terme à terme.
        :return:ge, array de booléens. Comparaison de l'objet courant avec l'autre objet selon la logique
        "plus grand ou égal à".
        """
        if isinstance(other, _Variables):
            other = other._valeurs
        ge = np.greater_equal(self._valeurs, other)
        return ge

    def __lt__(self, other: Union[int, float, complex, Iterable]) -> Iterable[bool]:
        """
        Méthode permettant de comparer l'objet courant à un autre selon la logique "plus petit que", soit self < other.
        :param other:int, float, complex, Iterable. Autre objet à comparer. Si scalaire, on compare toutes les entrées
        courante à ce scalaire. Si itérable, doit être de même dimension que l'objet courant. Dans ce cas, on effectue
        la comparaison terme à terme.
        :return:lt, array de booléens. Comparaison de l'objet courant avec l'autre objet selon la logique
        "plus petit que".
        """
        lt = ~(self >= other)
        return lt

    def __iadd__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer "sur place" (i.e. en modifiant l'objet courant) l'expression self + other, où
        self est l'objet courant et other est l'opérande. Permet de calculer self += other.
        :param other: int, float, complex, Iterable. Opérande à utiliser. Voir `__add__`.
        :return: self, l'objet courant modifié.
        """
        self._valeurs = (self + other)._valeurs
        return self

    def __isub__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer "sur place" (i.e. en modifiant l'objet courant) l'expression self - other, où
        self est l'objet courant et other est l'opérande. Permet de calculer self -= other.
        :param other: int, float, complex, Iterable. Opérande à utiliser. Voir `__sub__`.
        :return: self, l'objet courant modifié.
        """
        self._valeurs = (self - other)._valeurs
        return self

    def __itruediv__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer "sur place" (i.e. en modifiant l'objet courant) l'expression self / other, où
        self est l'objet courant et other est le diviseur. Permet de calculer self /= other.
        :param other: int, float, complex, Iterable. Diviseur à utiliser. Voir `__truediv__`.
        :return: self, l'objet courant modifié.
        """
        self._valeurs = (self / other)._valeurs
        return self

    def __ifloordiv__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer "sur place" (i.e. en modifiant l'objet courant) l'expression self // other, où
        self est l'objet courant et other est le diviseur. Permet de calculer self //= other.
        :param other: int, float, complex, Iterable. Diviseur à utiliser. Voir `__floordiv__`.
        :return: self, l'objet courant modifié.
        """
        self._valeurs = (self // other)._valeurs
        return self

    def __imod__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer "sur place" (i.e. en modifiant l'objet courant) l'expression self % other, où
        self est l'objet courant et other est le modulo (?). Permet de calculer self %= other.
        :param other: int, float, complex, Iterable. Modulo à utiliser. Voir `__mod__`.
        :return: self, l'objet courant modifié.
        """
        self._valeurs = (self % other)._valeurs
        return self

    def __imul__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer "sur place" (i.e. en modifiant l'objet courant) l'expression self * other, où
        self est l'objet courant et other est le facteur. Permet de calculer self *= other.
        :param other: int, float, complex, Iterable. Facteur à utiliser. Voir `__mul__`.
        :return: self, l'objet courant modifié.
        """
        self._valeurs = (self * other)._valeurs
        return self

    def __ipow__(self, other: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de calculer "sur place" (i.e. en modifiant l'objet courant) l'expression self ** other, où
        self est l'objet courant et other est l'exposant. Permet de calculer self **= other.
        :param other: int, float, complex, Iterable. Exposant à utiliser. Voir `__pow__`.
        :return: self, l'objet courant modifié.
        """
        self._valeurs = (self ** other)._valeurs
        return self

    def __str__(self) -> str:
        """
        Méthode retournant une représentation basique en caractères de l'objet courant.
        :return: msg, str. Représentation en caractères (str) de l'objet courant.
        """
        msg = f"{self.label} {self._valeurs}"
        return msg

    def __contains__(self, item) -> bool:
        """
        Méthode permettant de déterminer si un item est dans l'objet courant. Permet d'utiliser item in self où self
        est l'objet courant.
        :param item: object. Objet quelconque dont on veut déterminer s'il est présent dans l'objet courant. Voir la
        méthode `inclus` avec le paramètre `retourne_booleen_unique` mis à True.
        :return: isin, bool. Retourne si item est dans l'objet courant. Voir la méthode `inclus` avec le paramètre
        `retourne_booleen_unique` mis à True.
        """
        isin = self.inclus(item, True)
        return isin

    def inclus(self, item: Union[int, float, complex, Iterable], retourne_booleen_unique: bool = False) -> Union[
        Iterable[bool], bool]:
        """
        Méthode permettant de déterminer si un item est inclus dans l'objet courant.

        :param item: int, float, complex ou Iterable. Item à déterminer s'il est inclus dans l'objet courant.
        :param retourne_booleen_unique: bool. Paramètre spécifiant si on veut retourner un booléen unique. Si True,
        on retourne True si tous les éléments de item (si item est itérable, sinon item lui-même) sont présents dans
        l'objet courant. False sinon. Si cet argument est False, on retourne un array NumPy de la même taille que item
        (donc si item est scalaire, cela est un array à 1 élément) et chaque entrée possède un booléen spécifiant si
        l'élément associé d'item est inclus dans l'objet courant.
        :return: inclus, Iterable[bool] ou bool. On retourne un array NumPy de booléens spécifiant quel élément d'item
        est inclus (ou pas) dans l'objet courant. Si retourne_booleen_unique est True, on retourne True si tous les
        éléments d'inclus sont True, False si au moins un élément est False.
        """
        if not isinstance(item, _Variables):
            item = np.ravel(item)
        inclus = np.zeros_like(item, dtype=bool)
        for i, element in enumerate(item):
            inclus[i] = element in self._valeurs
        if retourne_booleen_unique:
            return all(inclus)
        else:
            return inclus

    def ajouter_variables(self, valeurs: Union[int, float, complex, Iterable],
                          positions: Union[int, slice, Iterable] = -1) -> bool:
        """
        Méthode permettant d'ajouter un ou plusieurs éléments à l'objet courant, si la modification de la taille est
        permise.
        :param valeurs: int, float, complex ou Iterable. Valeur(s) à ajouter dans l'objet courant. Si itérable, l'ordre
        est très important, car les valeurs sont ajoutées dans l'ordre qu'elles sont données.
        :param positions: int, slice, Iterable. Position(s) où ajouter les nouvelles valeurs (ajouter après).
        Il existe un cas particulier: si positions est -1, ajoute à la fin de l'objet courant (append). Pour le reste,
        les valeurs sont ajoutées après positions si entier, sinon après chaque entrées si iterable. Si itérable, il
        doit avoir la même taille que les valeurs.
        :return:not _bloquer_modifcation_taille, bool. On retourne le booléen associé à la modification de la taille. Si
        celle-ci n'est pas permise, aucune modification n'est faite et on retourne False. Sinon, on modifie l'objet
        courant et on retourne True.
        """
        if isinstance(valeurs, _Variables):
            ajout = valeurs.valeurs
        else:
            ajout = np.ravel(valeurs).copy()
        if not self._bloquer_modifcation_taille:
            type_ajout = ajout.dtype.type
            self._type_cast_priorite(type_ajout)
            if isinstance(positions, Integral) and positions == -1:
                self._valeurs = np.append(self._valeurs, valeurs)
            else:
                self._valeurs = np.insert(self._valeurs, positions, valeurs)
        self._len = len(self._valeurs)
        return not self._bloquer_modifcation_taille

    def enlever_variables(self, positions: Union[int, slice, Iterable] = None,
                          valeurs: Union[int, float, complex, Iterable] = None,
                          enlever_toutes_occurences: bool = False) -> Union[int, slice, Iterable, None]:
        """
        Méthode permettant d'enlever des valeurs de l'objet courant, si la modification de la taille est permise.
        :param positions: int, slice, Iterable ou None (défaut). Si spécifié (donc différent de None), on retire les
        éléments associées aux positions (si entier, on ne retire qu'un seul élément, si iterable ou slice, on retire
        tous ceux associés). Ne peut être spécifié en même temps que valeurs et doit être spécifié si valeurs est None.
        :param valeurs: int, float, complex, Iterable ou None (défaut). Si spécifié (donc différent de None), on retire
        les éléments de même valeur dans l'objet courant. Ne peut être spécifié en même temps que positions et
        doit être spécifié si positions est None.
        :param enlever_toutes_occurences: bool. Important seulement lorsque valeurs est spécifié. Sert à indiquer si on
        veut enlever toutes les valeurs correspondantes ou seulement la première occurence. Si True, seulement la
        première est enlever. Si False, toutes les occurences sont enlevées
        :return: positions: int, slice, Iterable ou None. Retourne la ou les positions où des éléments ont été retirés.
        Si on ne permet pas la modification de la taille, on retourne None.
        """
        if not self._bloquer_modifcation_taille:
            if positions is not None and valeurs is not None:
                msg = "Veuillez spécifier la position des éléments à retirer ou les valeurs à retirer, pas les deux."
                raise ValueError(msg)
            if positions is None and valeurs is None:
                return None
            if positions is None:
                if isinstance(valeurs, _Variables):
                    valeurs = valeurs.valeurs
                if enlever_toutes_occurences:
                    positions = np.where(np.isin(self._valeurs, valeurs))[0]
                else:
                    positions = self._trouver_indices_premiere_occurence(self._valeurs, valeurs)
            elif isinstance(positions, Number):
                positions = np.ravel(positions)
            self._valeurs = np.delete(self._valeurs, positions)

        else:
            positions = None
        self._len = len(self._valeurs)
        return positions

    def copie(self):
        """
        Méthode permettant de faire une copie (profonde) de l'objet courant.
        :return: copie, une copie (profonde) de l'objet courant.
        """
        copie = self.__cls(self)
        return copie

    def egalite_totale(self, autre: object) -> bool:
        """
        Méthode permettant de déterminer si deux objets sont totalement égaux. Par totalement, on entend un booléen
        unique spécifiant si l'autre objet est exactement égal à celui courant. Dans le cas où l'autre est itérable,
        on s'assure que toutes les entrées sont les mêmes. Si scalaire, on s'assure que toutes les entrées courantes
        sont égales à ce scalaire.
        :param autre: object. Autre objet à comparer avec celui courant
        :return:egalite, bool. Si autre est itérable, on comparer élément par élément et si tous concordent, on
        retourne True. Si scalaire, on vérifie que tous les éléments sont égaux à ce scalaire et si vrai, on retourne
        True.
        """
        egalite = np.all(self == autre)
        return egalite

    def concatener_a_courant(self, *variables: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de concatener de nouveaux objets à celui courant. Version concaténant à l'objet courant de
        `concatener` (repose sur ce code).
        :param variables: Iterable(s). Variables à concatener à l'objet courant. Il peut y en a voir un nombre
        arbitraire.
        :return: concat, nouvel objet composé des variables concatener à l'objet courant.
        """
        # DIFFÉRENT DE ajouter_variables: crée un nouvel objet, ne modifie pas un objet spécifique
        concat = self.__cls.concatener(self, *variables)
        return concat

    def _type_cast_priorite(self, nouveau_type: type):
        """
        Méthode permettant de "caster" l'array interne de valeurs selon le type de l'élément (ou des éléments) qu'on
        veut ajouter. Permet de régler des problèmes de pertes d'information. Par exemple, si l'array initial est
        de type 'uint 8', donc formé d'entiers non signés sur 8 bits, et qu'on veut ajouter un élément de type
        'uint16', on doit "caster" tout l'array à 'uint16', car c'est le type plus général permettant de garder toute
        l'information.
        :param nouveau_type: type. Classe de l'élément (ou des éléments) qu'on veut ajouter (peut être par
        modification avec __setitem__ ou ajouter directement). Doit être une classe, cela signifie que si l'élément
        est un ndarray de NumPy, simplement accéder à l'attribut 'dtype' n'est pas assez, car cela retourne une
        représentation qui n'est pas un type. On doit plutôt accéder à 'dtype.type'.
        :return: cast_dtype, le type permettant d'avoir un array conservant toute l'information, autant celle "ajoutée"
        que celle déjà présente.
        """
        if not isinstance(nouveau_type, type):
            raise TypeError("L'argument 'nouveau_type' doit être une classe.")
        if not issubclass(nouveau_type, Number):
            raise TypeError(f"Le type de données '{nouveau_type}' n'est pas supporté.")
        type_initial = self._valeurs.dtype
        cast_dtype = np.promote_types(type_initial, nouveau_type)
        if cast_dtype != type_initial:
            self._valeurs = self._valeurs.astype(cast_dtype)
        return cast_dtype

    @classmethod
    def concatener(cls, *variables: Union[int, float, complex, Iterable]):
        """
        Méthode permettant de concatener  plusieurs variables. Cette méthode est différente de `ajouter_variables`, car
        on ne modifie pas d'objet courant (méthode de classe).
        :param variables: Iterable(s). Variables à concatener. Il peut y en a voir un nombre arbitraire.
        :return: concat, nouvel objet composé de la concaténation de nouvelles variables une à la suite de l'autre.
        """
        # DIFFÉRENT DE ajouter_variables: crée un nouvel objet, ne modifie pas un objet spécifique
        premier = variables[0]
        label = cls.__label__
        if isinstance(premier, _Variables):
            label = premier.label
        concat = cls(np.concatenate(variables, axis=None), label)  # On doit mettre axis=None, car sinon les
        # variables doivent avoir la même taille.
        return concat

    @classmethod
    def __array_wrap__(cls, array):
        """
        Méthode permettant de s'assurer que la classe _Variables (et ses dérivées) puisse être utilisée de manière
        cohérente avec NumPy.
        :param array: Array NumPy résultant d'une manipulation quelconque par la librairie NumPy.
        :return: objet_variables, _Variables (ou ses dérivées). On retourne une représentation de l'array d'entrée en
        objet _Variables (ou ses dérivées).
        """
        objet_variables = cls(array)
        return objet_variables

    @staticmethod
    def _trouver_indices_premiere_occurence(array: Iterable, valeurs: Union[int, float, complex, Iterable]) -> list:
        """
        Méthode utilitaire permettant de trouver l'indice de première occurence de valeurs quelconque dans un array
        quelconque.
        :param array: Iterable. Conteneur dans lequel on veut trouver les éléments.
        :param valeurs: int, float, complex, Iterable. Valeur(s) qu'on désire obtenir l'index de première occurence.
        :return: indices, list. Liste d'indices de première occurence.
        """
        # Pas trouvé autre moyen de le faire plus efficacement
        indices = []
        valeurs = set(np.ravel(valeurs))  # S'assure d'avoir un itérable et on enlève les doublons.
        for val in valeurs:
            index = np.where(array == val)[0]
            if len(index) > 0:
                index = index[0]
                indices.append(index)
        return indices


class VariablesIndependantes(_Variables):
    __label__ = "Variables indépendantes"

    # TODO: Dans __setitem__, s'assurer que pas de doublons après. Même chose pour ajouter.

    def __init__(self, x: Iterable, label: str = None,
                 bloquer_ajout_modification_taille: bool = True):
        counts = np.unique(x, return_counts=True)[1]
        if any(counts != 1):
            raise ValueError("Les variables indépendantes doivent être uniques.")
        super(VariablesIndependantes, self).__init__(x, bloquer_ajout_modification_taille, label)


class VariablesDependantes(_Variables):
    __label__ = "Variables dépendantes"

    def __init__(self, y: Iterable, label: str = None,
                 bloquer_ajout_modification_taille: bool = True):
        super(VariablesDependantes, self).__init__(y, bloquer_ajout_modification_taille, label)

    @classmethod
    def from_variables_independantes(cls, x: Iterable,
                                     fonction: Callable):
        variables = fonction(x)
        variables = cls(variables)
        return variables


class Fonction:

    def __init__(self, x: Iterable, y: Iterable):
        if not isinstance(y, VariablesDependantes):
            y = VariablesDependantes(y)
        if not isinstance(x, VariablesIndependantes):
            x = VariablesIndependantes(x)
        if len(x) != len(y):
            raise ValueError("Les variables dépendantes et indépendantes doivent avoir la même taille.")
        self._x = x
        self._y = y
        self._x._bloquer_modifcation_taille = True
        self._y._bloquer_modifcation_taille = True
        self._liaison = None

    @property
    def info(self):
        liaison_info = None
        if self._liaison is not None:
            liaison_info = self._liaison.info
        info = {"observable x": self.x, "observable y": self.y, "liaisons": liaison_info}
        return info

    @property
    def liaison(self) -> _Liaison:
        return self._liaison

    @property
    def x(self) -> VariablesIndependantes:
        return self._x.copie()

    @property
    def y(self) -> VariablesDependantes:
        return self._y.copie()

    def __len__(self):
        return len(self._x)

    def __call__(self, x: Union[int, float, complex, Iterable]):
        if self._liaison is None:
            raise ValueError("Veuillez spécifier une manière de 'lier' les valeurs.")
        valeurs = self._liaison(x)
        return VariablesDependantes(valeurs)

    def __getitem__(self, item: Union[int, slice]) -> Tuple[
        Union[int, float, complex, Iterable], Union[int, float, complex, Iterable]]:
        y = self._y[item]
        x = self._x[item]
        return x, y

    def __setitem__(self, key: Union[int, slice, Iterable],
                    values: Tuple[Union[int, float, complex, Iterable, None], Union[
                        int, float, complex, Iterable, None]]) -> None:
        nouvelles_x = values[0]
        nouvelle_y = values[1]
        if nouvelle_y is not None:
            self._y[key] = nouvelle_y
        if nouvelles_x is not None:
            self._x[key] = nouvelles_x

    def changer_variables(self, key: Union[int, slice, Iterable],
                          values: Tuple[Union[int, float, complex, Iterable], Union[
                              int, float, complex, Iterable]] = (None, None)) -> None:
        self[key] = values

    def ajouter_variables(self, valeurs: Tuple[Union[int, float, complex, Sequence], Union[
        int, float, complex, Sequence]], positions: Union[int, slice, Iterable] = -1):
        # TODO: Permettre de refaire _Liaison OU mettre warning
        self._x._bloquer_modifcation_taille = False
        self._y._bloquer_modifcation_taille = False
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
        self._x._bloquer_modifcation_taille = True
        self._y._bloquer_modifcation_taille = True
        return ret

    def enlever_variables(self, positions: Union[int, slice, Iterable] = None,
                          valeurs_x: Union[int, float, complex, Iterable] = None):
        self._x._bloquer_modifcation_taille = False
        self._y._bloquer_modifcation_taille = False
        pos = self._x.enlever_variables(positions, valeurs_x)
        if pos is not None:
            self._y.enlever_variables(pos)
        self._x._bloquer_modifcation_taille = True
        self._y._bloquer_modifcation_taille = True
        return pos

    def ajouter_liaison(self, type_liaison: Type[_Liaison], borne_inf: float = None, borne_sup: float = None,
                        label: str = None, discontinuites_permises: bool = False, epsilon_continuite: float = None,
                        executer: bool = True, *execution_args, **execution_kwargs):
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

    def ajouter_liaisons(self, type_liaisons: Sequence[Type[_Liaison]], bornes_inf: Sequence[float],
                         bornes_sup: Sequence[float], labels: Sequence[str] = None,
                         discontinuites_permises: bool = False, epsilon_continuite: float = None,
                         executer: bool = True, execution_kwargs: dict = None):
        if len(type_liaisons) != len(bornes_inf) or len(type_liaisons) != len(bornes_sup):
            raise ValueError("Il doit y avoir autant de liaisons que de bornes inférieures et supérieures")
        if labels is None:
            labels = [None] * len(type_liaisons)
        # Si on n'a pas fini de créer la LiaisonMixte finale, on ne se soucie pas des dicontinuités possibles.
        discontinuites_permises_temp = True
        for i, type_liaison in enumerate(type_liaisons):
            if i == len(type_liaisons) - 1:
                discontinuites_permises_temp = discontinuites_permises
            self.ajouter_liaison(type_liaison, bornes_inf[i], bornes_sup[i], labels[i], discontinuites_permises_temp,
                                 epsilon_continuite, False)
        if executer:
            self._liaison.executer(execution_kwargs)
