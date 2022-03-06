# import envTests
from CodeUtilitaireSupplementaire.tests.customTestsUnitaires import CustomTestsUnitaires, main
from CodeUtilitaireSupplementaire import regression_interpolation as r_i
import numpy as np
from scipy.interpolate import interp1d


class TestsLiaison(CustomTestsUnitaires):
    def setUp(self) -> None:
        self.x = np.arange(10)
        self.y = np.random.randint(0, 100, 10)
        self.label = "label"
        self.l = r_i._Liaison(self.x, self.y, self.label)

    def test_liaison_init(self):
        l = self.l
        self.assertArrayEqual(l._x_obs, self.x)
        self.assertArrayEqual(l._y_obs, self.y)
        self.assertEqual(l.label, self.label)
        self.assertIsNone(l._fonction)

    def test_liaison_init_x_y_pas_meme_longueur(self):
        size_x = 10
        x = np.arange(size_x)
        y = np.random.randint(0, 100, size_x + 1)
        with self.assertRaises(ValueError):
            r_i._Liaison(x, y, "label")

    def test_liaison_property_fonction(self):
        l = self.l
        self.assertIsNone(l.fonction)

    def test_liaison_property_x(self):
        l = self.l
        self.assertArrayEqual(l.x_obs, self.x)

    def test_liaison_property_y(self):
        l = self.l
        self.assertArrayEqual(l.y_obs, self.y)

    def test_liaison_prete(self):
        l = self.l
        self.assertFalse(l.pret)

    def test_liaison_evaluer_aux_points_erreur(self):
        l = self.l
        with self.assertRaises(ValueError):
            l.evaluer_aux_points(self.x)

    def test_call_erreur(self):
        with self.assertRaises(ValueError):
            self.l(self.x)

    def test_executer_pas_implementee(self):
        with self.assertRaises(NotImplementedError):
            self.l.executer()

    def test_validation_valeurs_a_evaluer_toutes_valides(self):
        eval_x = self.l._validation_valeurs_a_evaluer(self.x)
        self.assertArrayEqual(eval_x, [True] * len(eval_x))

    def test_validation_valeurs_a_evaluer_warning(self):
        eval_x_initial = np.arange(-1, 10)
        msg_suppose = "Extrapolation non permise, " \
                      "donc les points en-dehors de l'intervalle `x_obs` ne sont pas pris en compte."
        with self.assertWarns(UserWarning) as w:
            self.l._validation_valeurs_a_evaluer(eval_x_initial)
            message_recu = str(w.warnings[0].message)
            self.assertEqual(message_recu, msg_suppose)

    def test_validation_valeurs_a_evaluer_pas_warning_mais_rejet(self):
        eval_x_initial = np.arange(-1, 10)
        self.assertNoWarn(self.l._validation_valeurs_a_evaluer, eval_x_initial, False)

    def test_validation_valeurs_a_evaluer_hors_range(self):
        eval_x_initial = np.arange(-1, 11)
        positions_valides = self.l._validation_valeurs_a_evaluer(eval_x_initial, False)
        self.assertArrayEqual(positions_valides, [False] + [True] * 10 + [False])

    def test_validation_valeurs_a_evaluer_warning_extrapolation_possible(self):
        eval_x_initial = np.arange(-1, 11)
        self.l._extrapolation_permise = True
        msg_suppose = "Extrapolation permise, donc les points en-dehors de l'intervall `x_obs` seront extrapolés. " \
                      "Veuillez considérer que les valeurs peuvent être loins de la vérité."
        with self.assertWarns(UserWarning) as w:
            self.l._validation_valeurs_a_evaluer(eval_x_initial)
            message_recu = str(w.warnings[0].message)
            self.assertEqual(message_recu, msg_suppose)

    def test_validation_valeurs_a_evaluer_pas_warning_extrapolation_possible(self):
        eval_x_initial = np.arange(10)
        self.l._extrapolation_permise = True
        self.assertNoWarn(self.l._validation_valeurs_a_evaluer, eval_x_initial)

    def test_validation_valeurs_a_evaluer_hors_range_mais_extrapolation_possible(self):
        eval_x_initial = np.arange(-1, 11)
        self.l._extrapolation_permise = True
        eval_x_apres = self.l._validation_valeurs_a_evaluer(eval_x_initial, False)
        self.assertArrayEqual(eval_x_apres, np.full_like(eval_x_apres, True, bool))

    def test_concatener_label_None(self):
        nb_liaisons = 5
        x = np.linspace(0, 100, 1000 * nb_liaisons)
        y = np.log(x + 1) * np.sin(x / np.pi)
        xy = np.vstack([x, y])
        all_xy = np.array_split(xy, nb_liaisons, 1)
        liaisons = []
        for i in range(nb_liaisons):
            liaisons.append(r_i._Liaison(all_xy[i][0], all_xy[i][1], label=f"Liaison {i + 1}"))
        concat = r_i._Liaison.concatener(True, None, None, *liaisons)
        self.assertIsInstance(concat, r_i.LiaisonMixte)
        self.assertEqual(len(concat), nb_liaisons)

    def test_concatener_label_pas_none(self):
        nb_liaisons = 5
        x = np.linspace(0, 100, 1000 * nb_liaisons)
        y = np.log(x + 1) * np.sin(x / np.pi)
        xy = np.vstack([x, y])
        all_xy = np.array_split(xy, nb_liaisons, 1)
        liaisons = []
        for i in range(nb_liaisons):
            liaisons.append(r_i._Liaison(all_xy[i][0], all_xy[i][1], label=f"Liaison {i + 1}"))
        concat = r_i._Liaison.concatener(True, None, "Test", *liaisons)
        self.assertIsInstance(concat, r_i.LiaisonMixte)
        self.assertEqual(len(concat), nb_liaisons)
        self.assertEqual(concat.label, "Test")

    def test_concatener_1_liaison(self):
        concat = r_i._Liaison.concatener(False, None, None, self.l)
        self.assertIsInstance(concat, r_i.LiaisonMixte)
        self.assertEqual(len(concat), 1)

    def test_concatener_a_courant(self):
        nb_liaisons = 5
        x = np.linspace(9.01, 100, 1000 * nb_liaisons)
        y = np.log(x + 1) * np.sin(x / np.pi)
        xy = np.vstack([x, y])
        all_xy = np.array_split(xy, nb_liaisons, 1)
        liaisons = []
        for i in range(nb_liaisons):
            liaisons.append(r_i._Liaison(all_xy[i][0], all_xy[i][1], label=f"Liaison {i + 1}"))
        concat = self.l.concatener_a_courant(True, None, None, *liaisons)
        self.assertIsInstance(concat, r_i.LiaisonMixte)
        self.assertEqual(len(concat), nb_liaisons + 1)

    def test_concatener_a_courant_1_liaison(self):
        x = np.arange(9, 20)
        y = np.random.randint(0, 100, len(x))
        liaison = r_i._Liaison(x, y, "adnfj")
        concat = self.l.concatener_a_courant(False, None, None, liaison)
        self.assertIsInstance(concat, r_i.LiaisonMixte)
        self.assertEqual(len(concat), 2)

    def test_add(self):
        x = np.arange(9, 20)
        y = np.random.randint(0, 100, len(x))
        liaison = r_i._Liaison(x, y, "adnfj")
        add = self.l + liaison
        self.assertIsInstance(add, r_i.LiaisonMixte)
        self.assertEqual(len(add), 2)

    def test_concatener_liaison_mixte(self):
        nb_liaisons = 5
        x = np.linspace(9.01, 100, 1000 * nb_liaisons)
        y = np.log(x + 1) * np.sin(x / np.pi)
        xy = np.vstack([x, y])
        all_xy = np.array_split(xy, nb_liaisons, 1)
        liaisons = []
        for i in range(nb_liaisons):
            liaisons.append(r_i._Liaison(all_xy[i][0], all_xy[i][1], label=f"Liaison {i + 1}"))
        lm = r_i.LiaisonMixte(liaisons, True, None)
        r_i._Liaison.__add_permet_discontinuites__ = True  # On s'assure que les discontinuités sont permises
        concat = self.l + lm
        self.assertIsInstance(concat, r_i.LiaisonMixte)
        self.assertEqual(len(concat), nb_liaisons + 1)

    def test_evaluer_aux_points_extrapolation_non_permise(self):
        x = np.arange(10)
        f = lambda x_: x_ ** 2
        y = f(x)
        liaison = r_i._Liaison(x, y, "Test")
        liaison._fonction = f
        eval_x = np.arange(-1, 11)
        eval_y_true = [np.nan] + list(x ** 2) + [np.nan]
        self.assertArrayEqual(liaison.evaluer_aux_points(eval_x, False), eval_y_true)

    def test_evaluer_aux_points_extrapolation_permise(self):
        x = np.arange(10)
        f = lambda x_: x_ ** 2
        y = f(x)
        liaison = r_i._Liaison(x, y, "Test")
        liaison._fonction = f
        liaison._extrapolation_permise = True
        eval_x = np.arange(-1, 11)
        eval_y_true = eval_x ** 2
        self.assertArrayEqual(liaison.evaluer_aux_points(eval_x, False), eval_y_true)

    def test_property_info(self):
        x = np.arange(10)
        f = lambda x_: x_ ** 2
        y = f(x)
        liaison = r_i._Liaison(x, y, "Test")
        self.assertDictEqual(liaison.info, {"fonction": None})


class TestsInterpolationBase(CustomTestsUnitaires):
    def setUp(self) -> None:
        self.x = np.arange(10)
        self.y = np.random.randint(0, 100, 10)
        self.label = "label"
        self.i = r_i._InterpolationBase(self.x, self.y, self.label)

    def test_init_extrapolation_pas_permise_defaut(self):
        i = self.i
        self.assertFalse(i._extrapolation_permise)

    def test_interpolation_base_sous_classe_liaison(self):
        i = self.i
        self.assertIsInstance(i, r_i._Liaison)

    def test_interpolation_pas_inplementee(self):
        with self.assertRaises(NotImplementedError):
            self.i.interpolation()

    def test_evaluer_aux_points_pas_pret_interpolation_pas_implementee(self):
        eval_x = np.linspace(2, 9)
        with self.assertRaises(ValueError):
            self.i.evaluer_aux_points(eval_x)

    def test_executer_erreur(self):
        with self.assertRaises(NotImplementedError):
            self.i.executer()

    def test_info(self):
        self.assertDictEqual(self.i.info, {"fonction interpolation": None})


class TestsInterpolationLineaire(CustomTestsUnitaires):

    def setUp(self) -> None:
        super(TestsInterpolationLineaire, self).setUp()

    def test_init_label_defaut(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        self.assertEqual(interpolation.label, "Interpolation linéaire")

    def test_interpolation_retourne_fonction_interp1d(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        interpolation_fonction = interpolation.interpolation()
        self.assertIsInstance(interpolation_fonction, interp1d)

    def test_interpolation_permet_pas_extrapolation(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        interpolation.interpolation()
        self.assertFalse(interpolation._extrapolation_permise)

    def test_interpolation_permet_extrapolation(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        interpolation.interpolation(True)
        self.assertTrue(interpolation._extrapolation_permise)

    def test_interpolation_prete(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        interpolation.interpolation()
        self.assertTrue(interpolation.pret)

    def test_interpolation_fonction_interp1d(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        interpolation.interpolation()
        self.assertIsInstance(interpolation.fonction, interp1d)

    # Equation spline linéaire:
    # y = ((x_i - x) / (x_i - x_(i-1))) * y_(i-1) + ((x - x_(i-1)) / (x_i - x_(i-1))) * y_i

    def test_interpolation_evaluer_points_fonction_lineaire(self):
        x = np.arange(1, 5)
        y = x.copy()  # Fonction linéaire
        eval_x = np.array([1.5, 2.5, 3.5])
        vrais_y = eval_x.copy()
        interp = r_i.InterpolationLineaire(x, y)
        interp.interpolation()
        eval_y = interp.evaluer_aux_points(eval_x)
        self.assertArrayEqual(eval_y, vrais_y)

    def test_interpolation_evaluer_points_fonction_quadratique(self):
        x = np.arange(1, 5)
        y = x ** 2  # Fonction quadratique
        eval_x = np.array([1.5, 2.5, 3.5])
        vrais_y = np.array([2.5, 6.5, 12.5])
        interp = r_i.InterpolationLineaire(x, y)
        interp.interpolation()
        eval_y = interp.evaluer_aux_points(eval_x)
        self.assertArrayEqual(eval_y, vrais_y)

    def test_call_fonctionne(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        interpolation.interpolation()
        eval_x = np.array([0.9, 1.24, 3.45])
        eval_y = interpolation.evaluer_aux_points(eval_x)
        eval_y_call = interpolation(eval_x)
        self.assertArrayEqual(eval_y, eval_y_call)

    def test_info_pas_pret(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        self.assertDictEqual(interpolation.info, {"fonction interpolation": None})

    def test_info_pret(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationLineaire(x, y)
        interpolation.interpolation()
        info = interpolation.info
        self.assertListEqual(list(info.keys()), ["fonction interpolation"])
        self.assertIsInstance(info["fonction interpolation"], interp1d)


class TestsInterpolationQuadratique(CustomTestsUnitaires):

    def test_init_label_defaut(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationQuadratique(x, y)
        self.assertEqual(interpolation.label, "Interpolation quadratique")

    def test_interpolation_retourne_fonction_interp1d(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationQuadratique(x, y)
        interpolation_fonction = interpolation.interpolation()
        self.assertIsInstance(interpolation_fonction, interp1d)

    def test_interpolation_permet_pas_extrapolation(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationQuadratique(x, y)
        interpolation.interpolation()
        self.assertFalse(interpolation._extrapolation_permise)

    def test_interpolation_permet_extrapolation(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationQuadratique(x, y)
        interpolation.interpolation(True)
        self.assertTrue(interpolation._extrapolation_permise)

    def test_interpolation_prete(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationQuadratique(x, y)
        interpolation.interpolation()
        self.assertTrue(interpolation.pret)

    def test_interpolation_fonction_interp1d(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationQuadratique(x, y)
        interpolation.interpolation()
        self.assertIsInstance(interpolation.fonction, interp1d)

    # Equation spline quadratique:
    # Système d'équations "compliquées"

    def test_interpolation_evaluer_points_fonction(self):
        # Ouf... ça serait compliqué à faire
        pass

    def test_call_fonctionne(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationQuadratique(x, y)
        interpolation.interpolation()
        eval_x = np.array([0.9, 1.24, 3.45])
        eval_y = interpolation.evaluer_aux_points(eval_x)
        eval_y_call = interpolation(eval_x)
        self.assertArrayEqual(eval_y, eval_y_call)


class TestsInterpolationCubique(CustomTestsUnitaires):

    def test_init_label_defaut(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationCubique(x, y)
        self.assertEqual(interpolation.label, "Interpolation cubique")

    def test_interpolation_retourne_fonction_interp1d(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationCubique(x, y)
        interpolation_fonction = interpolation.interpolation()
        self.assertIsInstance(interpolation_fonction, interp1d)

    def test_interpolation_permet_pas_extrapolation(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationCubique(x, y)
        interpolation.interpolation()
        self.assertFalse(interpolation._extrapolation_permise)

    def test_interpolation_permet_extrapolation(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationCubique(x, y)
        interpolation.interpolation(True)
        self.assertTrue(interpolation._extrapolation_permise)

    def test_interpolation_prete(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationCubique(x, y)
        interpolation.interpolation()
        self.assertTrue(interpolation.pret)

    def test_interpolation_fonction_interp1d(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationCubique(x, y)
        interpolation.interpolation()
        self.assertIsInstance(interpolation.fonction, interp1d)

    # Equation spline cubique:
    # Système d'équations "compliquées"

    def test_interpolation_evaluer_points_fonction(self):
        # Ouf... ça serait compliqué à faire
        pass

    def test_call_fonctionne(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        interpolation = r_i.InterpolationCubique(x, y)
        interpolation.interpolation()
        eval_x = np.array([0.9, 1.24, 3.45])
        eval_y = interpolation.evaluer_aux_points(eval_x)
        eval_y_call = interpolation(eval_x)
        self.assertArrayEqual(eval_y, eval_y_call)


class TestsRegressionBase(CustomTestsUnitaires):

    def test_init_donne_instance_liaison(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        obj = r_i._RegressionBase(x, y, "label")
        self.assertIsInstance(obj, r_i._Liaison)

    def test_init_reg_info_dict_vide(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        obj = r_i._RegressionBase(x, y, "label")
        self.assertDictEqual(obj._reg_info, {"paramètres optimisés": None, "sigma paramètres": None, "SSe": None,
                                             "fonction": None})

    def test_regression_pas_implementee(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        obj = r_i._RegressionBase(x, y, "label")
        with self.assertRaises(NotImplementedError):
            obj.regression()

    def test_generer_sigma_coefficients(self):
        matrice = np.arange(1, 26).reshape(5, 5) ** 2
        sigma_coefficients = np.array([1, 7, 13, 19, 25])
        self.assertArrayEqual(r_i._RegressionBase.generer_sigma_parametres(matrice), sigma_coefficients)

    def test_executer_erreur(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        obj = r_i._RegressionBase(x, y, "label")
        with self.assertRaises(NotImplementedError):
            obj.executer()

    def test_info(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        obj = r_i._RegressionBase(x, y, "label")
        self.assertDictEqual(obj.info, {"paramètres optimisés": None, "sigma paramètres": None, "SSe": None,
                                        "fonction": None})


class TestsRegressionPolynomiale(CustomTestsUnitaires):
    def setUp(self) -> None:
        r_i.RegressionPolynomiale.__warning_covariance_matrice__ = False

    def test_init_label(self):
        x = np.arange(10)
        y = np.random.randint(0, 100, 10)
        obj = r_i.RegressionPolynomiale(x, y)
        self.assertEqual(obj.label, "Regression polynomiale")

    def test_generer_matrice_vandermonde_puissance_croissante(self):
        x = np.arange(4)
        degre = 3
        vandermonde_supposee = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 4, 8], [1, 3, 9, 27]])
        vandermonde_calculee = r_i.RegressionPolynomiale.generer_matrice_Vandermonde(x, degre, True)
        self.assertArrayEqual(vandermonde_supposee, vandermonde_calculee)

    def test_generer_matrice_vandermonde_puissance_decroissante(self):
        x = np.arange(4)
        degre = 3
        vandermonde_supposee = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 4, 8], [1, 3, 9, 27]])
        vandermonde_supposee = np.flip(vandermonde_supposee, 1)  # On "flip" horizontalement l'array
        vandermonde_calculee = r_i.RegressionPolynomiale.generer_matrice_Vandermonde(x, degre)
        self.assertArrayEqual(vandermonde_supposee, vandermonde_calculee)

    def test_generer_matrice_covariance_pas_warning(self):
        x = np.arange(10)
        degre = 3
        vandermonde = r_i.RegressionPolynomiale.generer_matrice_Vandermonde(x, degre)
        nb_obs_x = len(x)
        SSe = 1e-7
        args = (vandermonde, SSe, nb_obs_x, degre, False)
        self.assertNoWarn(r_i.RegressionPolynomiale.generer_matrice_covariance, *args)

    def test_generer_matrice_covariance_warning(self):
        x = np.arange(10)
        degre = 3
        vandermonde = r_i.RegressionPolynomiale.generer_matrice_Vandermonde(x, degre)
        nb_obs_x = len(x)
        SSe = 1e-7
        args = (vandermonde, SSe, nb_obs_x, degre, True)
        msg_suppose = "La matrice de covariance n'est peut-être pas exacte. " \
                      "Elle est toutefois proportionnelle à un facteur multiplicatif " \
                      "près et peut servir de bonne d'approximation."
        with self.assertWarns(UserWarning) as w:
            r_i.RegressionPolynomiale.generer_matrice_covariance(*args)
            message_recu = str(w.warnings[0].message)
            self.assertEqual(message_recu, msg_suppose)

    def test_generer_matrice_covariance_warning_deja_sorti(self):
        x = np.arange(10)
        degre = 3
        vandermonde = r_i.RegressionPolynomiale.generer_matrice_Vandermonde(x, degre)
        nb_obs_x = len(x)
        SSe = 1e-7
        args = (vandermonde, SSe, nb_obs_x, degre, True)
        self.ignorer_warnings(r_i.RegressionPolynomiale.generer_matrice_covariance, *args)
        self.assertNoWarn(r_i.RegressionPolynomiale.generer_matrice_covariance, *args)

    def test_generer_matrice_covariance_output(self):
        x = np.arange(10)
        degre = 3
        vandermonde = r_i.RegressionPolynomiale.generer_matrice_Vandermonde(x, degre)
        nb_obs_x = len(x)
        SSe = 1e-7
        args = (vandermonde, SSe, nb_obs_x, degre, False)
        sigma = 1 / 60_000_000
        mat = np.array(
            [[5 / 15444, -5 / 1144, 461 / 30888, -7 / 858], [-5 / 1144, 19 / 312, -125 / 572, 19 / 143],
             [461 / 30888, -125 / 572, 26365 / 30888, -545 / 858], [-7 / 858, 19 / 143, -545 / 858, 589 / 715]])
        # Mat calculée avec WolframAlpha
        covariance = sigma * mat
        covariance_calculee = r_i.RegressionPolynomiale.generer_matrice_covariance(*args)
        self.assertArrayAllClose(covariance, covariance_calculee, 20)  # Égales à (au moins) 20 décimales près!!

    def test_generer_matrice_covariance_erreur(self):
        degre = 10
        x = np.arange(11)
        vandermonde = r_i.RegressionPolynomiale.generer_matrice_Vandermonde(x, degre)
        nb_obs_x = len(x)
        SSe = 1e-7
        args = (vandermonde, SSe, nb_obs_x, degre, False)
        with self.assertRaises(ValueError):
            r_i.RegressionPolynomiale.generer_matrice_covariance(*args)

    def test_regression_SSe_pas_trouve_warning(self):
        x = np.array([1, 2, 3])
        y = x.copy()
        degre = len(x)
        reg = r_i.RegressionPolynomiale(x, y)
        msg_suppose = "Impossible de calculer la somme des erreurs au carré (SSe), " \
                      "donc impossible d'estimer la matrice de covariance"
        with self.assertWarns(RuntimeWarning) as w:
            reg.regression(degre)
            message_recu = str(w.warnings[0].message)
            self.assertEqual(message_recu, msg_suppose)

    def test_regression_SSe_correct_pas_warnings(self):
        x = np.array([1, 2, 3, 4, 5, 6])
        y = x.copy()
        degre = 1
        reg = r_i.RegressionPolynomiale(x, y)

        self.assertNoWarn(reg.regression, degre, category=UserWarning)

    def test_regression_lineaire_output(self):
        # Fonction parfaitement linéaire
        x = np.arange(4)
        y = x.copy()
        degre = 1
        reg = r_i.RegressionPolynomiale(x, y)
        coefs, sigmas, SSe = self.ignorer_warnings(reg.regression, degre)
        pente = coefs[0]
        oo = coefs[1]
        self.assertAlmostEqual(pente, 1, 15)
        self.assertAlmostEqual(oo, 0, 15)
        self.assertArrayEqual(sigmas, np.zeros_like(sigmas))
        self.assertEqual(SSe, 0)

    def test_regression_pas_SSe_output(self):
        x = [0]
        y = [0]
        degre = 1
        reg = r_i.RegressionPolynomiale(x, y)
        coefs, sigmas, SSe = self.ignorer_warnings(reg.regression, degre)
        self.assertArrayEqual(coefs, np.zeros_like(coefs))
        self.assertArrayEqual(sigmas, np.nan)
        self.assertIsNone(SSe)

    def test_regression_quadratique_output(self):
        x = np.arange(10)
        y = x ** 2 + x * 2.5 - 0.5
        degre = 2
        reg = r_i.RegressionPolynomiale(x, y)
        coefs, sigmas, SSe = self.ignorer_warnings(reg.regression, degre)
        coef_x_carre = coefs[0]
        coef_x = coefs[1]
        oo = coefs[2]
        self.assertAlmostEqual(coef_x_carre, 1, 10)
        self.assertAlmostEqual(coef_x, 2.5, 10)
        self.assertAlmostEqual(oo, -0.5, 10)
        self.assertArrayAllClose(sigmas, np.zeros_like(sigmas), 10)
        self.assertAlmostEqual(SSe, 0, 20)

    def test_evaluer_aux_points_regression_pas_faite(self):
        x = np.arange(10)
        y = x.copy()
        reg = r_i.RegressionPolynomiale(x, y)
        with self.assertRaises(ValueError):
            reg.evaluer_aux_points(x)

    def test_evaluer_aux_points_regression_faite(self):
        x = np.arange(10)
        y = x.copy()
        reg = r_i.RegressionPolynomiale(x, y)
        self.ignorer_warnings(reg.regression, 1)
        self.assertNoRaise(reg.evaluer_aux_points, x)

    def test_evaluer_aux_points_output(self):
        x = np.arange(10)
        f = lambda x_: x_ ** 3 - x_ ** 2
        y = f(x)
        reg = r_i.RegressionPolynomiale(x, y)
        self.ignorer_warnings(reg.regression, 3, True)
        eval_x = np.array([-1, 0.5, 10, 5.56])
        eval_y = reg.evaluer_aux_points(eval_x, False)
        supposes_y = f(eval_x)
        self.assertArrayAllClose(eval_y, supposes_y, rtol=1e-10)

    def test_info_pas_pret(self):
        x = np.arange(10)
        f = lambda x_: x_ ** 3 - x_ ** 2
        y = f(x)
        reg = r_i.RegressionPolynomiale(x, y)
        self.assertDictEqual(reg.info, {"paramètres optimisés": None, "sigma paramètres": None, "SSe": None,
                                        "fonction": None, "degré": None})

    def test_info_regression_degre_3(self):
        x = np.arange(10)
        f = lambda x_: x_ ** 3 - x_ ** 2
        y = f(x)
        reg = r_i.RegressionPolynomiale(x, y)
        ret_reg = self.ignorer_warnings(reg.regression, 3, True)
        info = reg.info
        self.assertArrayEqual(info["paramètres optimisés"], ret_reg[0])
        self.assertArrayEqual(info["sigma paramètres"], ret_reg[1])
        self.assertEqual(info["SSe"], ret_reg[-1])
        self.assertIsNotNone(info["fonction"])
        self.assertEqual(info["degré"], 3)


class TestsRegressionGenerale(CustomTestsUnitaires):

    def test_init_label(self):
        x = np.arange(10)
        y = x.copy()
        reg = r_i.RegressionGenerale(x, y)
        self.assertEqual(reg.label, "Regression")

    def test_fonction_gaussienne_valeur_0(self):
        x = 0
        a, mu, sigma, b = 1, 0, 1, 2
        y = a + b
        self.assertEqual(r_i.RegressionGenerale.fonction_gaussienne(x, a, mu, sigma, b), y)

    def test_fonction_gaussienne_valeur_HWHM(self):
        sigma = 2
        a, mu, b = 1 / (sigma * np.sqrt(2 * np.pi)), 0, 0
        x = np.sqrt(2 * np.log(2)) * sigma
        y = a / 2
        self.assertEqual(r_i.RegressionGenerale.fonction_gaussienne(x, a, mu, sigma, b), y)

    def test_fonction_exponentielle_valeur_0(self):
        x = 0
        a, b, c, d = 10, 1, 0, 2
        y = a + d
        self.assertEqual(r_i.RegressionGenerale.fonction_exponentielle(x, a, b, c, d), y)

    def test_fonction_exponentielle_negative_0_vers_infini(self):
        x = 10_000_000
        a, b, c, d = 10, -1, 4, 2
        y = 0 + d
        self.assertEqual(r_i.RegressionGenerale.fonction_exponentielle(x, a, b, c, d), y)

    def test_fonction_sinus_valeur_0(self):
        x = 0
        a, b, c, d = 1, 10, 0, 2
        y = 0 + d
        self.assertEqual(r_i.RegressionGenerale.fonction_sinus(x, a, b, c, d), y)

    def test_fonction_sinus_devient_cosinus_valeur_0(self):
        x = 0
        a, b, c, d = 1.25, 17, np.pi / 2, -2.36
        y = a + d
        self.assertEqual(r_i.RegressionGenerale.fonction_sinus(x, a, b, c, d), y)

    def test_fonction_log_valeur_0(self):
        x = 0
        a, b, c, d = 10, 1, 0, 4
        y = -np.inf
        y_calcule = self.ignorer_warnings(r_i.RegressionGenerale.fonction_ln, x, a, b, c, d)
        self.assertEqual(y_calcule, y)

    def test_fonction_log_valeurs_negatives(self):
        x = np.arange(-10, 0)
        a, b, c, d = 1, 1, 0, 1
        y_calcules = self.ignorer_warnings(r_i.RegressionGenerale.fonction_ln, x, a, b, c, d)
        self.assertArrayEqual(y_calcules, np.nan)

    def test_fonction_log_valeur_e(self):
        x = np.e
        a, b, c, d = 10, 1, 0, 1
        y = a + d
        self.assertEqual(r_i.RegressionGenerale.fonction_ln(x, a, b, c, d), y)

    def test_regression_sans_estimation_initiale_output(self):
        x = np.linspace(0, np.pi * 2, 100)
        y = np.sin(x)
        reg = r_i.RegressionGenerale(x, y)
        f = lambda x, a, b: a * np.sin(x) + b
        params, sigmas, SSe = reg.regression(f)
        a, b = params
        self.assertEqual(a, 1)
        self.assertAlmostEqual(b, 0, delta=1e-9)
        self.assertArrayAllClose(sigmas, np.zeros_like(sigmas), 10)
        self.assertIsNone(SSe)

    def test_regression_estimation_initiale_output(self):
        x = np.linspace(-100, 100, 100_000)
        f = r_i.RegressionGenerale.fonction_gaussienne
        y = f(x, 1, 0, 1, 10)
        estimation_b = np.min(y)
        noise = 1e-7
        estimation = (np.max(y) - estimation_b + noise, np.mean(x) + noise, 1 + noise,
                      estimation_b + noise)
        reg = r_i.RegressionGenerale(x, y)
        params, sigmas, SSe = reg.regression(f, estimation)
        a, mu, sigma, b = params
        self.assertAlmostEqual(a, 1, 6)
        self.assertAlmostEqual(mu, 0, 6)
        self.assertAlmostEqual(sigma, 1, 6)
        self.assertAlmostEqual(b, 10, 6)
        self.assertArrayAllClose(sigmas, np.zeros_like(sigmas), 6)
        self.assertIsNone(SSe)

    def test_regression_change_avec_limites_output(self):
        # Fonction "carrée"
        def f(x, borne_droite, borne_gauche, minimum, maximum):
            if borne_droite < borne_gauche:
                t = borne_droite
                borne_droite = borne_gauche
                borne_gauche = t
            res = np.full_like(x, minimum, dtype=float)
            res[(borne_gauche <= x) & (x <= borne_droite)] = maximum
            return res

        x = np.arange(-10, 10)
        b_droite = 5
        b_gauche = -5
        minimum = -10
        maximum = 10
        p = (b_droite, b_gauche, minimum, maximum)
        y = f(x, *p)
        reg = r_i.RegressionGenerale(x, y)
        max_y = max(y)
        min_y = min(y)
        bounds = ([4, -6, min_y - 0.1, max_y - 0.1],
                  [6, -4, min_y + 0.1, max_y + 0.1])
        params, sigmas, SSe = reg.regression(f, None, bounds)
        b_d_estimation, b_g_estimation, min_estimation, max_estimation = params
        self.assertEqual(b_d_estimation, b_droite)
        self.assertEqual(b_g_estimation, b_gauche)
        self.assertEqual(min_estimation, minimum)
        self.assertEqual(max_estimation, maximum)
        self.assertArrayAllClose(sigmas, np.zeros_like(sigmas), 10)
        self.assertIsNone(SSe)

    def test_info_fonction_sinus(self):
        x = np.linspace(0, 10, 1000)
        f = lambda x_, a: a * np.sin(x_)
        y = f(x, 2)
        reg = r_i.RegressionGenerale(x, y)
        ret_reg = reg.regression(f)
        info = reg.info
        self.assertArrayEqual(info["paramètres optimisés"], ret_reg[0])
        self.assertArrayEqual(info["sigma paramètres"], ret_reg[1])
        self.assertIsNone(info["SSe"])
        self.assertEqual(info["fonction"], f)


class TestsLiaisonGenerale(CustomTestsUnitaires):

    def setUp(self) -> None:
        self.skipTest("Classe pas terminée!")

    def test_init_x_obs_et_y_obs_none(self):
        f = lambda x: x
        liaison = r_i.LiaisonGenerale(f)
        self.assertArrayEqual(liaison._x_obs, [])
        self.assertArrayEqual(liaison._y_obs, [])
        self.assertEqual(liaison.label, "Liaison générale")
        self.assertTrue(liaison.pret)

    def test_init_fonction_pas_callable(self):
        f = "x ** 2"
        with self.assertRaises(TypeError):
            r_i.LiaisonGenerale(f)

    def test_executer(self):
        f = lambda x: x
        liaison = r_i.LiaisonGenerale(f)
        self.assertIsNone(liaison.executer())


class TestsLiaisonMixte(CustomTestsUnitaires):

    @classmethod
    def setUpClass(cls, data_dir: str = None) -> None:
        super(TestsLiaisonMixte, cls).setUpClass(data_dir)
        r_i.RegressionPolynomiale.__warning_covariance_matrice__ = True  # Ne pas afficher le warning

    def test_analyse_continuite_epsilon_None(self):
        bornes = [(0, 10), (10, 20), (20, 50), (50, 100)]
        continuite = r_i.LiaisonMixte._analyse_continuite(bornes)
        self.assertTrue(continuite)

    def test_analyse_continuite_epsilon_none_pas_continu(self):
        bornes = [(0, 10), (10.025, 20), (20, 50), (50, 100)]
        continuite = r_i.LiaisonMixte._analyse_continuite(bornes)
        self.assertFalse(continuite)

    def test_analyse_continuite_epsilon_1e_moins_9(self):
        bornes = [(0, 10 + 0.99e-9), (10, 20 - 1e-10), (20, 30)]
        continuite = r_i.LiaisonMixte._analyse_continuite(bornes, 1e-9)
        self.assertTrue(continuite)

    def test_analyse_continuite_epsilon_1e_moins_9_pas_continu(self):
        bornes = [(0, 10 + 1.01e-9), (10, 20 - 1e-10), (20, 30)]
        continuite = r_i.LiaisonMixte._analyse_continuite(bornes, 1e-9)
        self.assertFalse(continuite)

    def test_trouver_bornes_nested_lists(self):
        listes = [np.arange(10 + 1), np.arange(10, 20 + 1), np.arange(20, 30)]
        bornes = r_i.LiaisonMixte._trouver_bornes_nested_lists(listes)
        self.assertArrayEqual(bornes, np.array([[0, 10], [10, 20], [20, 29]]).T)

    def test_trouver_bornes_nested_lists_retour_continuite_est_continue_epsilon_none(self):
        listes = [np.arange(10 + 1), np.arange(10, 20 + 1), np.arange(20, 30)]
        bornes, c = r_i.LiaisonMixte._trouver_bornes_nested_lists(listes, None, True)
        self.assertArrayEqual(bornes, np.array([[0, 10], [10, 20], [20, 29]]).T)
        self.assertTrue(c)

    def test_trouver_bornes_nested_lists_retour_continuite_est_pas_continue_epsilon_none(self):
        listes = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
        bornes, c = r_i.LiaisonMixte._trouver_bornes_nested_lists(listes, None, True)
        self.assertArrayEqual(bornes, np.array([[0, 9], [10, 19], [20, 29]]).T)
        self.assertFalse(c)

    def test_trouver_bornes_nested_lists_retour_continuite_est_continue_epsilon_1(self):
        listes = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
        bornes, c = r_i.LiaisonMixte._trouver_bornes_nested_lists(listes, 1, True)
        self.assertArrayEqual(bornes, np.array([[0, 9], [10, 19], [20, 29]]).T)
        self.assertTrue(c)

    def test_trouver_bornes_nested_lists_retour_continuite_est_pas_continue_epsilon_0p5(self):
        listes = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
        bornes, c = r_i.LiaisonMixte._trouver_bornes_nested_lists(listes, 0.5, True)
        self.assertArrayEqual(bornes, np.array([[0, 9], [10, 19], [20, 29]]).T)
        self.assertFalse(c)

    def test_creer_fonction_une_seule_liaison(self):
        bornes = np.array([[0], [9]])
        x = np.arange(*bornes)
        y = 3 * x + 2
        interpolation = r_i.InterpolationLineaire(x, y)
        interpolation.interpolation()
        liaisons = [interpolation]
        f = r_i.LiaisonMixte._creer_fonction(liaisons, bornes)
        self.assertTrue(callable(f))

    def test_creer_fonction_multiples_liaisons(self):
        bornes_1 = [0, 10]
        bornes_2 = [10, 20]
        bornes_3 = [20, 50]
        x_1 = np.linspace(*bornes_1, 1000)
        y_1 = x_1 + 2
        x_2 = np.linspace(*bornes_2, 1000)
        y_2 = x_2 ** 2 + 2
        x_3 = np.linspace(*bornes_3, 1000)
        y_3 = x_3 ** 3 + 2
        bornes = np.vstack([bornes_1, bornes_2, bornes_3]).T
        interpolation_1 = r_i.InterpolationLineaire(x_1, y_1)
        interpolation_2 = r_i.InterpolationQuadratique(x_2, y_2)
        interpolation_3 = r_i.InterpolationCubique(x_3, y_3)
        interpolation_1.interpolation()
        interpolation_2.interpolation()
        interpolation_3.interpolation()
        liaisons = [interpolation_1, interpolation_2, interpolation_3]
        f = r_i.LiaisonMixte._creer_fonction(liaisons, bornes)
        self.assertTrue(callable(f))

    def test_init_objet_pas_liaison(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(10, 20)
        y_2 = x_2.copy()
        l1 = r_i.RegressionPolynomiale(x_1, y_1)
        l2 = r_i.RegressionPolynomiale(x_2, y_2)
        l1.regression(2)
        l2.regression(1)
        liaisons = [l1, l2, lambda x: x * 9 + 10]
        with self.assertRaises(TypeError):
            r_i.LiaisonMixte(liaisons)

    def test_init_tout_ok(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(10, 20)
        y_2 = x_2.copy()
        l1 = r_i.RegressionPolynomiale(x_1, y_1)
        l2 = r_i.RegressionPolynomiale(x_2, y_2)
        l1.regression(2)
        l2.regression(1)
        liaisons = [l1, l2]
        self.assertNoRaise(r_i.LiaisonMixte, liaisons, True)

    def test_init_pas_ok_discontinuite_presente(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(10, 20)
        y_2 = x_2.copy()
        l1 = r_i.RegressionPolynomiale(x_1, y_1)
        l2 = r_i.RegressionPolynomiale(x_2, y_2)
        l1.regression(2)
        l2.regression(1)
        liaisons = [l1, l2]
        with self.assertRaises(ValueError) as e:
            r_i.LiaisonMixte(liaisons)
            msg = "Les liaisons ne sont pas continues selon une différence absolue maximale de 0. " \
                  "Veuillez vous assurer qu'elles sont continues ou que les discontinuités sont permises."
            self.assertEqual(str(e), msg)

    def test_init_attributs(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = x_2.copy()
        l1 = r_i.RegressionPolynomiale(x_1, y_1)
        l2 = r_i.RegressionPolynomiale(x_2, y_2)
        l1.regression(2)
        l2.regression(1)
        liaisons = [l1, l2]
        all_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        all_y = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81] + [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        x_s = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        y_s = [[0, 1, 4, 9, 16, 25, 36, 49, 64, 81], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        lm = r_i.LiaisonMixte(liaisons)
        self.assertListEqual(lm._liaisons, liaisons)
        self.assertArrayEqual(lm._bornes, np.array([[0, 9], [9, 19]]).T)
        self.assertArrayEqual(lm._x_obs, all_x)
        self.assertArrayEqual(lm._y_obs, all_y)
        self.assertTrue(lm._pret)
        self.assertListEqual([list(x) for x in lm._x_obs_all], x_s)
        self.assertListEqual([list(y) for y in lm._y_obs_all], y_s)
        self.assertTrue(callable(lm._fonction))

    def test_init_liaison_unique(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        liaison = r_i.InterpolationQuadratique(x_1, y_1)
        liaison.interpolation()
        lm = r_i.LiaisonMixte(liaison)
        self.assertListEqual(lm._liaisons, [liaison])
        self.assertArrayEqual(lm._bornes, np.array([[0], [9]]))
        self.assertArrayEqual(lm._x_obs, x_1)
        self.assertArrayEqual(lm._y_obs, y_1)
        self.assertTrue(lm._pret)
        self.assertListEqual([list(x) for x in lm._x_obs_all], [list(x_1)])
        self.assertListEqual([list(y) for y in lm._y_obs_all], [list(y_1)])

    def test_init_liaison_unique_pas_prete(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        liaison = r_i.InterpolationQuadratique(x_1, y_1)
        lm = r_i.LiaisonMixte(liaison)
        self.assertListEqual(lm._liaisons, [liaison])
        self.assertArrayEqual(lm._bornes, np.array([[0], [9]]))
        self.assertArrayEqual(lm._x_obs, x_1)
        self.assertArrayEqual(lm._y_obs, y_1)
        self.assertFalse(lm._pret)
        self.assertListEqual([list(x) for x in lm._x_obs_all], [list(x_1)])
        self.assertListEqual([list(y) for y in lm._y_obs_all], [list(y_1)])

    def test_init_2_liaisons_derniere_pas_prete(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = x_2.copy()
        l1 = r_i.RegressionPolynomiale(x_1, y_1)
        l2 = r_i.RegressionPolynomiale(x_2, y_2)
        l1.regression(2)
        liaisons = [l1, l2]
        all_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        all_y = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81] + [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        x_s = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        y_s = [[0, 1, 4, 9, 16, 25, 36, 49, 64, 81], [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        lm = r_i.LiaisonMixte(liaisons)
        self.assertListEqual(lm._liaisons, liaisons)
        self.assertArrayEqual(lm._bornes, np.array([[0, 9], [9, 19]]).T)
        self.assertArrayEqual(lm._x_obs, all_x)
        self.assertArrayEqual(lm._y_obs, all_y)
        self.assertFalse(lm._pret)
        self.assertListEqual([list(x) for x in lm._x_obs_all], x_s)
        self.assertListEqual([list(y) for y in lm._y_obs_all], y_s)
        self.assertTrue(callable(lm._fonction))

    def test_property_liaisons_1_seule(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        liaison = r_i.InterpolationQuadratique(x_1, y_1)
        liaison.interpolation()
        lm = r_i.LiaisonMixte(liaison)
        self.assertListEqual(lm.liaisons, [liaison])

    def test_property_liasons_multiples(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        for l in liaisons: l.executer()
        lm = r_i.LiaisonMixte(liaisons)
        self.assertSequenceEqual(lm.liaisons, liaisons)

    def test_property_bornes_1_liaison(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        lm = r_i.LiaisonMixte(liaison_1)
        self.assertArrayEqual(lm.bornes, np.array([[0], [9]]))

    def test_property_bornes_liaisons_multiples(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        lm = r_i.LiaisonMixte(liaisons)
        self.assertArrayEqual(lm.bornes, [[0, 9, 19], [9, 19, 29]])

    def test_property_pret(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        for l in liaisons: l.executer()
        lm = r_i.LiaisonMixte(liaisons)
        self.assertTrue(lm.pret)

    def test_property_pret_pas_prets(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        for l in liaisons[::2]: l.executer()
        lm = r_i.LiaisonMixte(liaisons)
        self.assertFalse(lm.pret)

    def test_len_liaison_unique(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        liaison = r_i.InterpolationQuadratique(x_1, y_1)
        liaison.interpolation()
        lm = r_i.LiaisonMixte(liaison)
        self.assertEqual(len(lm), 1)

    def test_len_liaisons_multiples(self):
        nb_liaisons = 5
        x_s = np.linspace(0, 100, 1000 * nb_liaisons)
        all_x = np.array_split(x_s, nb_liaisons)
        y_s = np.sin(x_s ** 0.5) ** 2 + np.cos(x_s)
        all_y = np.array_split(y_s, nb_liaisons)
        liaisons = []
        for i in range(nb_liaisons):
            liaisons.append(r_i.InterpolationCubique(all_x[i], all_y[i]))
        lm = r_i.LiaisonMixte(liaisons, True, 0.5)
        self.assertEqual(len(lm), nb_liaisons)

    def test_concater_a_courant(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2)
        lm = r_i.LiaisonMixte(liaisons)
        concat = lm.concatener_a_courant(False, None, None, liaison_3)
        self.assertIsInstance(concat, r_i.LiaisonMixte)
        self.assertEqual(len(concat), 3)

    def test_getitem_index_scalaire_positif_out_of_range(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        lm = r_i.LiaisonMixte(liaisons)
        with self.assertRaises(IndexError):
            lm[len(lm)]

    def test_getitem_index_scalaire_negatif_out_of_range(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        lm = r_i.LiaisonMixte(liaisons)
        with self.assertRaises(IndexError):
            lm[-len(lm) - 1]

    def test_getitem_index_scalaire_positif_valide(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        lm = r_i.LiaisonMixte(liaisons)
        liaison_getitem = liaison_2
        bornes_getitem = np.array([[9], [19]])
        tuple_obs_getitem = (x_2, y_2)
        getitem = lm[1]
        self.assertEqual(getitem[0], liaison_getitem)
        self.assertArrayEqual(getitem[1], bornes_getitem)
        self.assertArrayEqual(getitem[2], tuple_obs_getitem)

    def test_getitem_index_scalaire_negatif_valide(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        lm = r_i.LiaisonMixte(liaisons)
        liaison_getitem = liaison_1
        bornes_getitem = np.array([[0], [9]])
        tuple_obs_getitem = (x_1, y_1)
        getitem = lm[-3]
        self.assertEqual(getitem[0], liaison_getitem)
        self.assertArrayEqual(getitem[1], bornes_getitem)
        self.assertArrayEqual(getitem[2], tuple_obs_getitem)

    def test_getitem_index_slice_valide(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        lm = r_i.LiaisonMixte(liaisons)
        liaison_getitem = [liaison_1, liaison_3]
        bornes_getitem = np.array([[0, 19], [9, 29]])
        getitem = lm[::2]
        self.assertSequenceEqual(getitem[0], liaison_getitem)
        self.assertArrayEqual(getitem[1], bornes_getitem)
        self.assertArrayEqual(getitem[2][0][0], x_1)
        self.assertArrayEqual(getitem[2][0][1], x_3)
        self.assertArrayEqual(getitem[2][1][0], y_1)
        self.assertArrayEqual(getitem[2][1][1], y_3)

    def test_getitem_slice_out_of_range(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        lm = r_i.LiaisonMixte(liaisons)
        getitem = lm[10:]
        self.assertSequenceEqual(getitem[0], [])
        self.assertArrayEqual(getitem[1], [[], []])
        self.assertArrayEqual(getitem[2], ([], []))

    def test_executer_mixte_args_kwargs(self):
        x_1 = np.arange(10)
        y_1 = x_1 ** 2
        x_2 = np.arange(9, 20)
        y_2 = np.full_like(x_2, y_1[-1])
        x_3 = np.arange(19, 30)
        y_3 = np.exp(-x_3) + y_2[-1]
        liaison_1 = r_i.InterpolationQuadratique(x_1, y_1)
        liaison_2 = r_i.InterpolationLineaire(x_2, y_2)
        liaison_3 = r_i.InterpolationCubique(x_3, y_3)
        liaisons = (liaison_1, liaison_2, liaison_3)
        lm = r_i.LiaisonMixte(liaisons)
        args = {0: {"permettre_extrapolation": True}, 2: (True,)}
        lm.executer(args)
        self.assertTrue(lm.pret)
        self.assertTrue(lm.liaisons[0].pret)
        self.assertTrue(lm.liaisons[2].pret)
        self.assertTrue(lm.liaisons[1].pret)
        self.assertTrue(lm.liaisons[0]._extrapolation_permise)
        self.assertTrue(lm.liaisons[2]._extrapolation_permise)
        self.assertFalse(lm.liaisons[1]._extrapolation_permise)

    def test_fonction_1_liaison(self):
        x = np.arange(10)
        y = np.sin(x)
        reg = r_i.RegressionGenerale(x, y)
        reg.regression(lambda x, a: a * np.sin(x))
        y_estimes = reg(x)
        lm = r_i.LiaisonMixte(reg)
        y_fct = lm._fonction(x)
        self.assertArrayEqual(y_fct, y_estimes)

    def test_fonction_2_liaisons(self):
        min_x = 0
        max_x = 20
        x_s = np.linspace(min_x, max_x, 1000)
        x_sin = x_s[x_s <= 2 * np.pi]
        x_quad = x_s[x_s > 2 * np.pi]
        x_1 = np.linspace(min_x, np.pi * 2, 100)
        y_1 = np.sin(x_1)
        x_2 = np.linspace(np.pi * 2, max_x, 100)
        y_2 = ((x_1[-1] - x_2) ** 2) / 100
        reg_1 = r_i.RegressionGenerale(x_1, y_1)
        reg_1.regression(lambda x, a: a * np.sin(x))
        reg_2 = r_i.RegressionPolynomiale(x_2, y_2)
        reg_2.regression(2)
        liaisons = [reg_1, reg_2]
        lm = r_i.LiaisonMixte(liaisons)
        y_estimes_1 = reg_1(x_sin)
        y_estimes_2 = reg_2(x_quad)
        y_estimes = np.append(y_estimes_1, y_estimes_2)
        y_fonction = lm._fonction(x_s)
        self.assertArrayEqual(y_fonction, y_estimes)

    def test_fonction_2_liaisons_extrapolations_non_permises(self):
        min_x = 0
        max_x = 20
        x_s = np.arange(min_x - 1, max_x + 2)
        x_sin = x_s[x_s <= 2 * np.pi].copy()
        x_quad = x_s[x_s > 2 * np.pi].copy()
        x_1 = np.linspace(min_x, np.pi * 2, 100)
        y_1 = np.sin(x_1)
        x_2 = np.linspace(np.pi * 2, max_x, 100)
        y_2 = ((x_1[-1] - x_2) ** 2) / 100
        reg_1 = r_i.RegressionGenerale(x_1, y_1)
        reg_1.regression(lambda x, a: a * np.sin(x))
        reg_2 = r_i.RegressionPolynomiale(x_2, y_2)
        reg_2.regression(2)
        liaisons = [reg_1, reg_2]
        lm = r_i.LiaisonMixte(liaisons)
        y_estimes_1 = reg_1(x_sin, False)
        y_estimes_2 = reg_2(x_quad, False)
        y_estimes = np.append(y_estimes_1, y_estimes_2)
        y_fonction = self.ignorer_warnings(lm._fonction, x_s)
        self.assertArrayEqual(y_fonction, y_estimes)
        self.assertIsNan(y_fonction[0])
        self.assertIsNan(y_fonction[-1])

    def test_fonction_2_liaisons_extrapolations_permises(self):
        min_x = 0
        max_x = 20
        x_s = np.linspace(min_x - 1, max_x + 1, 1000)
        x_sin = x_s[x_s <= 2 * np.pi]
        x_quad = x_s[x_s > 2 * np.pi]
        x_1 = np.linspace(min_x, np.pi * 2, 100)
        y_1 = np.sin(x_1)
        x_2 = np.linspace(np.pi * 2, max_x, 100)
        y_2 = ((x_1[-1] - x_2) ** 2) / 100
        reg_1 = r_i.RegressionGenerale(x_1, y_1)
        reg_1.regression(lambda x, a: a * np.sin(x), permettre_extrapolation=True)
        reg_2 = r_i.RegressionPolynomiale(x_2, y_2)
        reg_2.regression(2, True)
        liaisons = [reg_1, reg_2]
        lm = r_i.LiaisonMixte(liaisons)
        y_estimes_1 = reg_1(x_sin, False)
        y_estimes_2 = reg_2(x_quad, False)
        y_estimes = np.append(y_estimes_1, y_estimes_2)
        y_fonction = self.ignorer_warnings(lm._fonction, x_s)
        self.assertArrayEqual(y_fonction, y_estimes)
        self.assertNoNans(y_fonction)

    def test_fonction_2_liaisons_extrapolation_permise_min(self):
        min_x = 0
        max_x = 20
        x_s = np.linspace(min_x - 1, max_x + 1, 1000)
        x_sin = x_s[x_s <= 2 * np.pi]
        x_quad = x_s[x_s > 2 * np.pi]
        x_1 = np.linspace(min_x, np.pi * 2, 100)
        y_1 = np.sin(x_1)
        x_2 = np.linspace(np.pi * 2, max_x, 100)
        y_2 = ((x_1[-1] - x_2) ** 2) / 100
        reg_1 = r_i.RegressionGenerale(x_1, y_1)
        reg_1.regression(lambda x, a: a * np.sin(x), permettre_extrapolation=False)
        reg_2 = r_i.RegressionPolynomiale(x_2, y_2)
        reg_2.regression(2, True)
        liaisons = [reg_1, reg_2]
        lm = r_i.LiaisonMixte(liaisons)
        y_estimes_1 = reg_1(x_sin, False)
        y_estimes_2 = reg_2(x_quad, False)
        y_estimes = np.append(y_estimes_1, y_estimes_2)
        y_fonction = self.ignorer_warnings(lm._fonction, x_s)
        self.assertArrayEqual(y_fonction, y_estimes)
        self.assertIsNan(y_fonction[0])

    def test_fonction_2_liaisons_extrapolation_permise_max(self):
        min_x = 0
        max_x = 20
        x_s = np.linspace(min_x - 1, max_x + 1, 1000)
        x_sin = x_s[x_s <= 2 * np.pi]
        x_quad = x_s[x_s > 2 * np.pi]
        x_1 = np.linspace(min_x, np.pi * 2, 100)
        y_1 = np.sin(x_1)
        x_2 = np.linspace(np.pi * 2, max_x, 100)
        y_2 = ((x_1[-1] - x_2) ** 2) / 100
        reg_1 = r_i.RegressionGenerale(x_1, y_1)
        reg_1.regression(lambda x, a: a * np.sin(x), permettre_extrapolation=True)
        reg_2 = r_i.RegressionPolynomiale(x_2, y_2)
        reg_2.regression(2, False)
        liaisons = [reg_1, reg_2]
        lm = r_i.LiaisonMixte(liaisons)
        y_estimes_1 = reg_1(x_sin, False)
        y_estimes_2 = reg_2(x_quad, False)
        y_estimes = np.append(y_estimes_1, y_estimes_2)
        y_fonction = self.ignorer_warnings(lm._fonction, x_s)
        self.assertArrayEqual(y_fonction, y_estimes)
        self.assertIsNan(y_fonction[-1])

    def test_info_liaison_prete_et_liaison_pas_prete(self):
        min_x = 0
        max_x = 20
        x_1 = np.linspace(min_x, np.pi * 2, 100)
        y_1 = np.sin(x_1)
        x_2 = np.linspace(np.pi * 2, max_x, 100)
        y_2 = ((x_1[-1] - x_2) ** 2) / 100
        reg_1 = r_i.RegressionGenerale(x_1, y_1, "Régression sin")
        reg_1.regression(lambda x, a: a * np.sin(x), permettre_extrapolation=True)
        reg_2 = r_i.RegressionPolynomiale(x_2, y_2, "Régression polynôme")
        liaisons = [reg_1, reg_2]
        lm = r_i.LiaisonMixte(liaisons)
        info = lm.info
        self.assertIterableEqual(info.keys(), ["Régression sin", "Régression polynôme"])
        self.assertIterableEqual(info["Régression sin"].keys(),
                                 ["paramètres optimisés", "sigma paramètres", "SSe", "fonction"])
        self.assertAreNotNone(info["Régression sin"].values(), 2)
        self.assertIterableEqual(info["Régression polynôme"].keys(),
                                 ["paramètres optimisés", "sigma paramètres", "SSe", "fonction", "degré"])
        self.assertAreNone(info["Régression polynôme"].values())

    def test_info_liaisons_pretes(self):
        min_x = 0
        max_x = 20
        x_1 = np.linspace(min_x, np.pi * 2, 100)
        y_1 = np.sin(x_1)
        x_2 = np.linspace(np.pi * 2, max_x, 100)
        y_2 = ((x_1[-1] - x_2) ** 2) / 100
        reg_1 = r_i.RegressionGenerale(x_1, y_1, "Régression sin")
        reg_1.regression(lambda x, a: a * np.sin(x), permettre_extrapolation=True)
        reg_2 = r_i.RegressionPolynomiale(x_2, y_2, "Régression polynôme")
        liaisons = [reg_1, reg_2]
        lm = r_i.LiaisonMixte(liaisons)
        reg_2.executer(2, False)
        info = lm.info
        self.assertIterableEqual(info.keys(), ["Régression sin", "Régression polynôme"])
        self.assertIterableEqual(info["Régression sin"].keys(),
                                 ["paramètres optimisés", "sigma paramètres", "SSe", "fonction"])
        self.assertAreNotNone(info["Régression sin"].values(), 2)
        self.assertIterableEqual(info["Régression polynôme"].keys(),
                                 ["paramètres optimisés", "sigma paramètres", "SSe", "fonction", "degré"])
        self.assertAreNotNone(info["Régression polynôme"].values())


if __name__ == '__main__':
    main()
