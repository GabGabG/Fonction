import os
import unittest
from warnings import warn, catch_warnings, simplefilter
import numpy as np
from numbers import Number
from typing import Iterable, Union, List


class CustomTestsUnitaires(unittest.TestCase):
    __test_dir = os.path.join(os.path.abspath(__file__), "..")
    __data_dir_output = os.path.join(__test_dir, "testDataOutput")
    __data_dir_input = os.path.join(__test_dir, "testDataInput")
    afficher_warnings = True

    def __init__(self, tests=()):
        super(CustomTestsUnitaires, self).__init__(tests)

    @classmethod
    def data_dir_output(cls):
        return cls.__data_dir_output[:]  # On retourne une "copie" pour ne pas directement modifier la valeur

    @classmethod
    def data_dir_input(cls):
        return cls.__data_dir_input[:]

    @classmethod
    def changer_data_dir_output(cls, nouveau_nom: str) -> None:
        if not cls.data_dir_output_existe():
            if cls.afficher_warnings:
                msg = "Le dossier de data existe déjà, son nom ne peut donc pas être changé maintenant."
                warn(msg, RuntimeWarning, 2)
        else:
            nouveau_nom = os.path.join(cls.__test_dir, nouveau_nom)
            cls.__data_dir_output = nouveau_nom

    @classmethod
    def setUpClass(cls, data_dir: str = None) -> None:
        cls.creer_data_dir_output(data_dir)

    @classmethod
    def data_dir_output_existe(cls) -> bool:
        return os.path.exists(cls.__data_dir_output)

    @classmethod
    def creer_data_dir_output(cls, data_dir: str = None):
        if data_dir is not None:
            cls.changer_data_dir_output(data_dir)
        if cls.data_dir_output_existe():
            if cls.afficher_warnings:
                msg = "Le dossier de ressources existe déjà. Il sera utilisé puis supprimé à la fin."
                warn(msg, ResourceWarning, 2)
        else:
            os.mkdir(cls.__data_dir_output)

    @classmethod
    def supprimer_data_dir_output(cls) -> None:
        dd = cls.__data_dir_output
        if cls.data_dir_output_existe():
            for element in os.listdir(dd):
                os.remove(os.path.join(dd, element))
            os.rmdir(dd)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.supprimer_data_dir_output()

    def assertNoRaise(self, func, *fargs, **fkwargs):
        try:
            func(*fargs, **fkwargs)
        except Exception as e:
            self.fail(f"Une exception a été attrapée:\n{e}")

    def assertNoWarn(self, func, *fargs, category=Warning, **fkwargs):
        with catch_warnings(record=True) as warnings:
            self.ignorer_warnings(func, *fargs, category=category, **fkwargs)
        if len(warnings) != 0:
            warnings_str = "\n".join(str(w) for w in warnings)
            self.fail(f"Des 'warnings' ont été reçus:\n{warnings_str}")

    def assertFileExists(self, file: str):
        if not os.path.exists(file):
            self.fail(f"Le fichier {file} n'existe pas.")

    def assertFileDoesNotExist(self, file: str):
        if os.path.exists(file):
            self.fail(f"Le fichier {file} existe.")

    def assertFileContentEqual(self, file: str, content: str, strip: bool = True):
        self.assertFileExists(file)
        with open(file, "r") as f:
            file_content = f.read()
        if strip:
            file_content = file_content.strip()
            content = content.strip()
        self.assertEqual(file_content, content)

    def assertArrayEqual(self, a1: Iterable, a2: Iterable):
        self.assertIsNone(np.testing.assert_array_equal(a1, a2))

    def assertArrayAllClose(self, a1: Iterable, a2: Iterable, places: int = None, rtol: float = None,
                            atol: float = None):
        if places is not None:
            self.assertIsNone(np.testing.assert_almost_equal(a1, a2, places))
        else:
            rtol = rtol if rtol is not None else 1e-7
            atol = atol if atol is not None else 0
            self.assertIsNone(np.testing.assert_allclose(a1, a2, rtol, atol))

    def assertIsNan(self, a: Union[Number, Iterable]):
        self.assertIsNone(np.testing.assert_array_equal(a, np.nan))

    def assertNoNans(self, a: Union[Number, Iterable]):
        self.assertIsNone(np.testing.assert_array_equal(np.isnan(a), False))

    def assertIterableEqual(self, i1: Iterable, i2: Iterable):
        l1 = list(i1)
        l2 = list(i2)
        self.assertListEqual(l1, l2)

    def assertAreNotNone(self, i: Iterable, exclude_index: Union[int, List[int]] = None):
        if isinstance(exclude_index, int):
            exclude_index = [exclude_index]
        if exclude_index is None:
            exclude_index = []
        for index, element in enumerate(i):
            if index not in exclude_index:
                self.assertIsNotNone(element)

    def assertAreNone(self, i: Iterable, exclude_index: Union[int, List[int]] = None):
        if isinstance(exclude_index, int):
            exclude_index = [exclude_index]
        if exclude_index is None:
            exclude_index = []
        for index, element in enumerate(i):
            if index not in exclude_index:
                self.assertIsNone(element)

    def ignorer_warnings(self, func, *fargs, category=Warning, **fkwargs):
        with catch_warnings():
            simplefilter("ignore", category)
            f = func(*fargs, *fkwargs)
        return f


def main(*args, **kwargs):
    return unittest.main(*args, **kwargs)
