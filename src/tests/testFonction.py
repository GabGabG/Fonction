from src.tests.customTestsUnitaires import CustomTestsUnitaires, main
from src.variables import VariablesDependantes, VariablesIndependantes
from src.fonction import Fonction
from src.regression_interpolation import LiaisonMixte, RegressionPolynomiale, \
    RegressionGenerale, InterpolationLineaire, InterpolationQuadratique, InterpolationCubique
import numpy as np
import os


class TestsFonction(CustomTestsUnitaires):

    def test_init_pas_objets_variables(self):
        x = np.arange(10)
        y = x ** 2
        f = Fonction(x, y)
        self.assertIsInstance(f._x, VariablesIndependantes)
        self.assertIsInstance(f._y, VariablesDependantes)

    def test_init_attributs(self):
        x = VariablesIndependantes(np.arange(20))
        y = VariablesDependantes.from_variables_independantes(x, lambda x_: 2 * np.sin(x_))
        f = Fonction(x, y)
        self.assertArrayEqual(f._x, x)
        self.assertArrayEqual(f._y, y)
        self.assertIsNone(f._liaison)

    def test_init_variables_pas_meme_taille(self):
        x = np.arange(10)
        y = x[:-1] ** 2.3
        with self.assertRaises(ValueError):
            Fonction(x, y)

    def test_property_liaison(self):
        x = np.arange(10)
        y = x ** 0.5
        f = Fonction(x, y)
        self.assertIsNone(f.liaison)

    def test_property_x(self):
        x = np.arange(10)
        y = x ** 1 / 9
        f = Fonction(x, y)
        self.assertArrayEqual(f.x, x)

    def test_property_y(self):
        x = np.arange(10)
        y = x ** 1 / 9
        f = Fonction(x, y)
        self.assertArrayEqual(f.y, y)

    def test_len(self):
        x = np.arange(10)
        y = x ** 1 / 9
        f = Fonction(x, y)
        self.assertEqual(len(f), len(x))
        self.assertEqual(len(f), len(y))

    def test_getitem_scalaire(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        f = Fonction(x, y)
        getitem = f[-1]
        self.assertTupleEqual(getitem, (9, 19))

    def test_getitem_slice(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        f = Fonction(x, y)
        getitem = f[:]
        self.assertIsInstance(getitem[0], VariablesIndependantes)
        self.assertIsInstance(getitem[1], VariablesDependantes)
        self.assertArrayEqual(getitem, (x, y))

    def test_setitem_cle_out_of_bounds(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        f = Fonction(x, y)
        with self.assertRaises(IndexError):
            f[len(x)] = 10, 10

    def test_setitem_cle_valide_valeurs_None(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        f = Fonction(x, y)
        f[0] = None, None
        self.assertArrayEqual(f.x, x)
        self.assertArrayEqual(f.y, y)

    def test_setitem_cle_slice_valeurs_pas_meme_size(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        f = Fonction(x, y)
        with self.assertRaises(ValueError):
            f[:] = ([10, 10], [10, 10])

    def test_setitem_change_valeurs(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        f = Fonction(x, y)
        f[-1] = (20, -1)
        new_x = x.copy()
        new_x[-1] = 20
        new_y = y.copy()
        new_y[-1] = -1
        self.assertArrayEqual(f.x, new_x)
        self.assertArrayEqual(f.y, new_y)

    def test_changer_variables(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        f = Fonction(x, y)
        f.changer_variables(-1, (20, -1))
        new_x = x.copy()
        new_x[-1] = 20
        new_y = y.copy()
        new_y[-1] = -1
        self.assertArrayEqual(f.x, new_x)
        self.assertArrayEqual(f.y, new_y)

    def test_ajouter_variables_pas_meme_taille(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        f = Fonction(x, y)
        with self.assertRaises(ValueError):
            f.ajouter_variables((-1, [10, 20]), 0)

    def test_ajouter_variables_debut_ajout_scalaires(self):
        x = np.arange(10)
        y = x ** 2
        f = Fonction(x, y)
        ret = f.ajouter_variables((-1, 1), 0)
        self.assertTrue(ret)
        self.assertEqual(len(f), len(x) + 1)
        self.assertArrayEqual(f.x, np.concatenate([-1, x], None))
        self.assertArrayEqual(f.y, np.concatenate([1, y], None))

    def test_ajouter_variables_fin_ajout_array(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        ajout_x = np.arange(10, 20)
        ajout_y = np.cos(ajout_x)
        ret = f.ajouter_variables((ajout_x, ajout_y), -1)
        new_x = np.arange(20)
        self.assertTrue(ret)
        self.assertEqual(len(f), len(x) * 2)
        self.assertArrayEqual(f.x, new_x)
        self.assertArrayEqual(f.y, np.cos(new_x))

    def test_enlever_valeurs_positions_et_valeurs_none(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        pos = f.enlever_variables()
        self.assertIsNone(pos)

    def test_enlever_valeurs_positions_pas_none(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        pos = f.enlever_variables(-1)
        self.assertEqual(pos, -1)
        self.assertEqual(len(f), len(x) - 1)
        self.assertArrayEqual(f.x, x[:-1])
        self.assertArrayEqual(f.y, y[:-1])

    def test_enlever_valeurs_positions_array(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        pos = f.enlever_variables([0, 9])
        self.assertArrayEqual(pos, [0, 9])
        self.assertEqual(len(f), len(x) - 2)
        self.assertArrayEqual(f.x, x[1:-1])
        self.assertArrayEqual(f.y, y[1:-1])

    def test_enlever_valeurs_valeur_scalaire(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        pos = f.enlever_variables(None, 0)
        self.assertArrayEqual(pos, [0])
        self.assertEqual(len(f), len(x) - 1)
        self.assertArrayEqual(f.x, x[1:])
        self.assertArrayEqual(f.y, y[1:])

    def test_enlever_valeurs_valeur_array(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        pos = f.enlever_variables(None, [0, 1, 2])
        self.assertArrayEqual(pos, [0, 1, 2])
        self.assertEqual(len(f), len(x) - 3)
        self.assertArrayEqual(f.x, x[3:])
        self.assertArrayEqual(f.y, y[3:])

    def test_enlever_valeur_pas_presente(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        pos = f.enlever_variables(None, -10)
        self.assertArrayEqual(pos, [])
        self.assertEqual(len(f), len(x))
        self.assertArrayEqual(f.x, x)
        self.assertArrayEqual(f.y, y)

    def test_ajouter_liaison_liaison_mixte(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        msg = "Spécifier une liaison mixte est ambiguë. Veuillez spécifier chaque liaison interne une à la " \
              "fois ou utiliser `ajouter_liaisons` avec toutes ses liaisons internes."
        with self.assertRaises(TypeError) as e:
            f.ajouter_liaison(LiaisonMixte)
        self.assertEqual(str(e.exception), msg)

    def test_ajouter_liaison_bornes_none_pas_executer(self):
        x = np.arange(10)
        y = np.cos(x)
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionGenerale, executer=False)
        self.assertIsInstance(f.liaison, RegressionGenerale)
        self.assertFalse(f.liaison.pret)
        self.assertArrayEqual(f.liaison.x_obs, x)
        self.assertArrayEqual(f.liaison.y_obs, y)
        self.assertEqual(f.liaison.label, "Regression")

    def test_ajouter_liaison_bornes_pas_none_executer(self):
        x = np.arange(11)
        y = np.cos(x)
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionGenerale, 0.45, 9.89, "Régression cosinus", False, None, True,
                          fonction=lambda x, a: np.cos(x) * a)
        self.assertIsInstance(f.liaison, RegressionGenerale)
        self.assertTrue(f.liaison.pret)
        self.assertArrayEqual(f.liaison.x_obs, x[1:-1])
        self.assertArrayEqual(f.liaison.y_obs, y[1:-1])
        self.assertEqual(f.liaison.label, "Régression cosinus")

    def test_ajouter_liaison_a_deja_presente(self):
        x = np.linspace(0, 10, 1000)
        x_med = 5
        x_1 = x[x < x_med]
        x_2 = x[x >= x_med]
        y_1 = np.cos(x_1)
        y_2 = (x_2 + y_1[-1]) ** 2 / 100
        y = np.concatenate([y_1, y_2], None)
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionGenerale, 0, 5, "Régression cosinus", False, None, True,
                          fonction=lambda x, a: np.cos(x) * a)
        self.ignorer_warnings(f.ajouter_liaison, RegressionPolynomiale, 5, 10, "Régression quadratique", False, 0.02,
                              True, 2)
        self.assertIsInstance(f.liaison, LiaisonMixte)
        self.assertTrue(f.liaison.pret)

    def test_ajouter_liaisons(self):
        RegressionPolynomiale.__warning_covariance_matrice__ = True
        x = np.linspace(0, 10, 1000)
        x_med = 5
        x_1 = x[x < x_med]
        x_2 = x[x >= x_med]
        y_1 = np.cos(x_1)
        y_2 = (x_2 + y_1[-1]) ** 2 / 100
        y = np.concatenate([y_1, y_2], None)
        f = Fonction(x, y)
        f.ajouter_liaisons([RegressionGenerale, RegressionPolynomiale], [0, 5], [5, 10], ["r1", "r2"], False, 0.02,
                           True, {0: {"fonction": lambda x, a: np.cos(x) * a}, 1: {"degre": 2}})
        self.assertIsInstance(f.liaison, LiaisonMixte)
        self.assertTrue(f.liaison.pret)

    def test_ajouter_liaisons_a_deja_presente_pas_executer(self):
        RegressionPolynomiale.__warning_covariance_matrice__ = True
        x_1 = np.linspace(-10, 0, 1000, True)[:-1]
        x = np.linspace(0, 10, 1000)
        x_med = 5
        x_2 = x[x < x_med]
        x_3 = x[x >= x_med]
        y_1 = x_2 ** 3 * 1 / 89
        y_2 = np.cos(x_1)
        y_3 = (x_3 + y_2[-1]) ** 2 / 100
        y = np.concatenate([y_1, y_2, y_3], None)
        x = np.concatenate([x_1, x], None)
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionGenerale, 0, 5, "Reg cos", False, 0.02, True, fonction=lambda x, a: np.cos(x) * a)
        f.ajouter_liaisons([RegressionPolynomiale, RegressionPolynomiale], [-10, 5], [0, 10], ["Reg cub", "Reg quad"],
                           False, 0.02, False)
        self.assertIsInstance(f.liaison, LiaisonMixte)
        self.assertFalse(f.liaison.pret)

    def test_ajouter_liaisons_desordre_temporairement_pas_continu(self):
        RegressionPolynomiale.__warning_covariance_matrice__ = True
        x_1 = np.linspace(-10, 0, 1000, True)[:-1]
        x = np.linspace(0, 10, 1000)
        x_med = 5
        x_2 = x[x < x_med]
        x_3 = x[x >= x_med]
        y_1 = x_2 ** 3 * 1 / 89
        y_2 = np.cos(x_1)
        y_3 = (x_3 + y_2[-1]) ** 2 / 100
        y = np.concatenate([y_1, y_2, y_3], None)
        x = np.concatenate([x_1, x], None)
        f = Fonction(x, y)
        f.ajouter_liaisons([RegressionPolynomiale, RegressionPolynomiale, RegressionGenerale], [-10, 5, 0], [0, 10, 5],
                           None, False, 0.02, True, {0: (3,), 1: (2,), 2: {"fonction": lambda x, a: np.cos(x) * a}})
        self.assertIsInstance(f.liaison, LiaisonMixte)
        self.assertTrue(f.liaison.pret)

    def test_ajouter_liaisons_pas_meme_nombre_de_bornes_inf_que_nombre_liaisons(self):
        x = np.linspace(0, 10, 100)
        y = x ** 1.3
        f = Fonction(x, y)
        with self.assertRaises(ValueError) as e:
            f.ajouter_liaisons([InterpolationLineaire, InterpolationQuadratique, InterpolationCubique], [0, 2],
                               [2, 5, 10], discontinuites_permises=True)
        msg = "Il doit y avoir autant de liaisons que de bornes inférieures et supérieures"
        self.assertEqual(str(e.exception), msg)

    def test_ajouter_liaisons_pas_meme_nombre_de_bornes_sup_que_nombre_liaisons(self):
        x = np.linspace(0, 10, 100)
        y = x ** 1.3
        f = Fonction(x, y)
        with self.assertRaises(ValueError) as e:
            f.ajouter_liaisons([InterpolationLineaire, InterpolationQuadratique, InterpolationCubique], [0, 2, 5],
                               [5, 10], discontinuites_permises=True)
        msg = "Il doit y avoir autant de liaisons que de bornes inférieures et supérieures"
        self.assertEqual(str(e.exception), msg)

    def test_call_pas_liaison(self):
        x = np.arange(10)
        y = x ** 2
        f = Fonction(x, y)
        msg = "Veuillez spécifier une manière de 'lier' les valeurs."
        with self.assertRaises(ValueError) as e:
            f(x)
        self.assertEqual(str(e.exception), msg)

    def test_call_liaison_pas_prete(self):
        x = np.arange(10)
        y = x ** 2
        f = Fonction(x, y)
        not_msg = "Veuillez spécifier une manière de 'lier' les valeurs."
        f.ajouter_liaison(RegressionPolynomiale, None, None, "Reg", False, None, False)
        with self.assertRaises(ValueError) as e:
            f(x)
        self.assertNotEqual(str(e), not_msg)

    def test_call_liaison_prete(self):
        RegressionPolynomiale.__warning_covariance_matrice__ = True
        x = np.arange(10)
        y = x ** 2
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionPolynomiale, None, None, "Reg", False, None, True, 2)
        y_eval = f(x)
        self.assertArrayAllClose(y_eval, y, 13)

    def test_call_multiple_laisons_pretes(self):
        RegressionPolynomiale.__warning_covariance_matrice__ = True
        x_s = np.linspace(0, 100, 1000)
        l_maxs = [25, 67, np.inf]
        l_mins = [-np.inf, 25, 67]
        x_1 = x_s[x_s <= 25]
        x_2 = x_s[(x_s >= 25) & (x_s <= 67)]
        x_3 = x_s[x_s >= 67]
        y_1 = x_1 ** 2
        y_2 = x_2 ** 2 * -1 + 2 * y_1[-1]
        y_3 = y_2[-1] + (x_3 - x_3[0]) * 10
        y_s = np.concatenate([y_1, y_2, y_3], None)
        f = Fonction(x_s, y_s)
        liaisons = [RegressionPolynomiale for _ in [1, 2, 3]]
        f.ajouter_liaisons(liaisons, l_mins, l_maxs, None, False, 0.2, True, {0: (2,), 1: (2,), 2: (1,)})
        y_eval = f(x_s)
        self.assertArrayAllClose(y_eval, y_s, atol=1e-11)

    def test_info_pas_liaison(self):
        x = np.arange(10)
        y = x ** 2
        f = Fonction(x, y)
        info = f.info
        self.assertIterableEqual(info.keys(), ["observable x", "observable y", "liaisons"])
        self.assertArrayEqual(info["observable x"], x)
        self.assertArrayEqual(info["observable y"], y)
        self.assertIsNone(info["liaisons"])

    def test_info_avec_liaisons(self):
        x = np.arange(10)
        y = (x ** 1 / 2) * 7
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionGenerale, 0, 5, label="Reg pas prête", executer=False)
        f.ajouter_liaison(RegressionGenerale, 5, 10, label="Reg pas prête 2", executer=False)
        info = f.info
        self.assertIterableEqual(info["liaisons"].keys(), ["Reg pas prête", "Reg pas prête 2"])

    def test_setitem_liaison_maintenant_invalide(self):
        x = np.arange(10)
        y = x ** 2 - x + 10
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionPolynomiale, None, None, "Regression polynomiale", degre=2)
        warning_msg = "Liaison maintenant invalide."
        with self.assertWarns(UserWarning) as w:
            f[0] = (-1, -1)
        self.assertNotEqual(str(w), warning_msg)
        self.assertIsNone(f.liaison)

    def test_ajouter_variable_liaison_maintenant_invalide(self):
        x = np.arange(10)
        y = x ** 2 - x + 10
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionPolynomiale, None, None, "Regression polynomiale", degre=2)
        warning_msg = "Liaison maintenant invalide."
        with self.assertWarns(UserWarning) as w:
            f.ajouter_variables((-1, 0), 0)
        self.assertNotEqual(str(w), warning_msg)
        self.assertIsNone(f.liaison)

    def test_enlever_variable_liaison_maintenant_invalide(self):
        x = np.arange(10)
        y = x ** 2 - x + 10
        f = Fonction(x, y)
        f.ajouter_liaison(RegressionPolynomiale, None, None, "Regression polynomiale", degre=2)
        warning_msg = "Liaison maintenant invalide."
        with self.assertWarns(UserWarning) as w:
            f.enlever_variables(slice(10))
        self.assertNotEqual(str(w), warning_msg)
        self.assertIsNone(f.liaison)

    def test_from_csv_file_pas_de_noms_de_colonnes(self):
        file = os.path.join(os.path.dirname(__file__), r"testData\\test_from_csv.csv")
        f = Fonction.from_csv_file(file)
        self.assertIterableEqual(f.x, [0, 1, 2, 3, 4, 5])
        self.assertIterableEqual(f.y, [1, 1, 2.2, 4.5, 1, 1])

    def test_from_csv_file_noms_de_colonnes(self):
        file = os.path.join(os.path.dirname(__file__), r"testData\\test_from_csv.csv")
        f = Fonction.from_csv_file(file, colonne_var_indep="toto1", colonne_var_dep="toto2")
        self.assertIterableEqual(f.x, [0, 1, 2, 3, 4, 5])
        self.assertIterableEqual(f.y, [1, 1, 2.2, 4.5, 1, 1])

    def test_from_csv_file_long_header(self):
        file = os.path.join(os.path.dirname(__file__), r"testData\\test_from_csv_long_header.csv")
        f = Fonction.from_csv_file(file, en_tete_noms_colonnes=5)
        self.assertIterableEqual(f.x, [0, 1, 2])
        self.assertIterableEqual(f.y, [0, 1, 2])

    def test_from_csv_file_long_header_cols_diff(self):
        file = os.path.join(os.path.dirname(__file__), r"testData\\test_from_csv_long_header.csv")
        f = Fonction.from_csv_file(file, ",", 5, "c1", "c3")
        self.assertIterableEqual(f.x, [0, 1, 2])
        self.assertIterableEqual(f.y, [1, 2, 3])


if __name__ == '__main__':
    main()
