# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Fonction.tests.customTestsUnitaires import CustomTestsUnitaires, main
from Fonction import data_generation
import numpy as np
import os
import warnings


class TestsMethodesStatiques(CustomTestsUnitaires):

    def test_factorielle_entier_negatif(self):
        with self.assertRaises(ValueError):
            data_generation.factorielle(-1)

    def test_factorielle_flottant(self):
        with self.assertRaises(ValueError):
            data_generation.factorielle(1 + 1e-9)

    def test_factorielle_valide_flottant_decimale_0(self):
        self.assertNoRaise(data_generation.factorielle, 9.0)

    def test_factorielle_critere_arret(self):
        self.assertEqual(data_generation.factorielle(1), 1)

    def test_factorielle_0(self):
        self.assertEqual(data_generation.factorielle(0), 1)

    def test_factorielle_5(self):
        self.assertEqual(data_generation.factorielle(5), 120)

    def test_factorielle_10(self):
        self.assertEqual(data_generation.factorielle(10), 3628800)

    def test_coefficient_binomial_k_superieur_n(self):
        n = 1
        k = n + 1
        self.assertEqual(data_generation.coefficient_binomial(n, k), 0)

    def test_coefficient_binomial_k_0(self):
        n = np.random.randint(0, 100, 1)
        k = 0
        self.assertEqual(data_generation.coefficient_binomial(n, k), 1)

    def test_coefficient_binomial(self):
        n = 10
        k = 2
        self.assertEqual(data_generation.coefficient_binomial(n, k), 45)


class TestsNuageDePoints(CustomTestsUnitaires):

    def setUp(self):
        self.x = np.random.randint(0, 100, (10,))
        self.y = np.random.randint(0, 100, (10,))
        self.contenu = "\n".join(f"{a},{b}" for a, b in zip(self.x, self.y))

    def test_init_vecteurs_taille_diff(self):
        x = self.x
        y = np.hstack([self.x, self.x])
        with self.assertRaises(ValueError):
            data_generation.NuageDePointsBase(x, y)

    def test_init_vecteurs_meme_taille(self):
        x = self.x
        y = self.y
        self.assertNoRaise(data_generation.NuageDePointsBase, x, y)

    def test_enregistrer_pas_erreur(self):
        x = self.x
        y = self.y
        n = data_generation.NuageDePointsBase(x, y)
        self.assertNoRaise(n.enregistrer, os.path.join(self.data_dir_output(), "test_avec_colonnes.csv"))
        self.assertNoRaise(n.enregistrer, os.path.join(self.data_dir_output(), "test_pas_colonnes.csv"), False)

    def test_contenu_enregistrer_correct_pas_colonnes(self):
        x = self.x
        y = self.y
        n = data_generation.NuageDePointsBase(x, y)
        f = os.path.join(self.data_dir_output(), "test_contenu_pas_colonnes.csv")
        n.enregistrer(f, False)
        c = self.contenu
        self.assertFileContentEqual(f, c)

    def test_contenu_enregistrer_correct_avec_colonnes(self):
        x = self.x
        y = self.y
        n = data_generation.NuageDePointsBase(x, y)
        f = os.path.join(self.data_dir_output(), "test_contenu_pas_colonnes.csv")
        n.enregistrer(f)
        c = "x,y\n" + self.contenu
        self.assertFileContentEqual(f, c)


class TestsDistribution(CustomTestsUnitaires):

    def test_x_et_y_None(self):
        d = data_generation.Distribution()
        self.assertIsNone(d.x)
        self.assertIsNone(d.y)

    def test_fonction_leve_exception(self):
        d = data_generation.Distribution()
        with self.assertRaises(NotImplementedError):
            d.fonction(None)

    def test_distribution_probabilite_leve_exception(self):
        d = data_generation.Distribution()
        with self.assertRaises(NotImplementedError):
            d.distribution_probabilite(0, 10, 100)

    def test_enregistrer_leve_exception(self):
        d = data_generation.Distribution()
        with self.assertRaises(ValueError):
            d.enregistrer("test_pas_enregistre.csv")

    def test_distribution_discrete_fonction_valeur_pas_entiere(self):
        d = data_generation.DistributionDiscrete()
        with self.assertRaises(TypeError):
            d.fonction(1.2)

    def test_distribution_discrete_fonction_valeurs_pas_entieres(self):
        d = data_generation.DistributionDiscrete()
        vals = np.array([1, 2, 3, 4.5, 5, 5.6], dtype=float)
        with self.assertRaises(TypeError):
            d.fonction(vals)

    def test_distribution_discrete_fonction_valeurs_entiere(self):
        d = data_generation.DistributionDiscrete()
        vals = np.array([1.0, 2, 3, 100], dtype=float)
        self.assertNoRaise(d.fonction, vals)

    def test_distribution_discrete_distribution_probabilite_valeurs(self):
        x_min = 0
        x_max = 10
        pas = 1
        d = data_generation.DistributionDiscrete()
        x, y = d.distribution_probabilite(x_min, x_max, pas)
        self.assertArrayEqual(x, range(x_min, x_max + 1, pas))
        self.assertIsNone(y)

    def test_distribution_continue_fonction_valeurs_decimales_ok(self):
        x_min = 0
        x_max = 10
        nb_x = 100
        d = data_generation.DistributionContinue()
        self.assertNoRaise(d.fonction, np.linspace(x_min, x_max, nb_x))

    def test_distribution_continue_fonction_valuers_entieres_ok(self):
        x_min = 0
        x_max = 10
        nb_x = 11
        d = data_generation.DistributionContinue()
        self.assertNoRaise(d.fonction, np.linspace(x_min, x_max, nb_x))

    def test_distribution_continue_distribution_probabilite_valeurs(self):
        x_min = 0
        x_max = 10
        nb_x = 50
        d = data_generation.DistributionContinue()
        x, y = d.distribution_probabilite(x_min, x_max, nb_x)
        self.assertArrayEqual(x, np.linspace(x_min, x_max, nb_x))
        self.assertIsNone(y)


class TestsDistributionBinomiale(CustomTestsUnitaires):

    def test_probabilite_negative(self):
        p = -1e-9
        with self.assertRaises(ValueError):
            data_generation.DistributionBinomiale(p, 10)

    def test_probabilite_plus_de_1(self):
        p = 1 + 1e-9
        with self.assertRaises(ValueError):
            data_generation.DistributionBinomiale(p, 10)

    def test_probabilite_0(self):
        p = 0
        self.assertNoRaise(data_generation.DistributionBinomiale, p, 1)

    def test_probabilite_1(self):
        p = 1
        self.assertNoRaise(data_generation.DistributionBinomiale, p, 1)

    def test_binomial_distribution_discrete(self):
        p = 1
        n = 1
        d = data_generation.DistributionBinomiale(p, n)
        self.assertIsInstance(d, data_generation.DistributionDiscrete)

    def test_nombre_essais_negatif(self):
        n = -1
        with self.assertRaises(ValueError):
            data_generation.DistributionBinomiale(1, n)

    def test_nombre_essais_0(self):
        n = 0
        self.assertNoRaise(data_generation.DistributionBinomiale, 1, n)

    def test_probabilite_nombre_essais_ok(self):
        p = np.random.uniform()
        n = np.random.randint(0, 500, 1, int)
        self.assertNoRaise(data_generation.DistributionBinomiale, p, n)

    def test_property_getters(self):
        p = np.random.uniform()
        n = np.random.randint(0, 500, 1, int)
        d = data_generation.DistributionBinomiale(p, n)
        q = 1 - p
        self.assertEqual(d.p, p)
        self.assertEqual(d.n, n)
        self.assertEqual(d.q, q)

    def test_property_setter_p_invalide_superieur_1(self):
        p = 1
        n = np.random.randint(0, 500, 1, int)
        d = data_generation.DistributionBinomiale(p, n)
        with self.assertRaises(ValueError):
            d.p = p + 1e-9

    def test_property_setter_p_invalide_inferieur_0(self):
        p = 0
        n = np.random.randint(0, 500, 1, int)
        d = data_generation.DistributionBinomiale(p, n)
        with self.assertRaises(ValueError):
            d.p = p - 1e-9

    def test_property_setter_p_valide_egal_0(self):
        p = 0
        n = np.random.randint(0, 500, 1, int)
        d = data_generation.DistributionBinomiale(p, n)
        try:
            d.p = p
        except Exception as e:
            self.fail(f"Une exception a été attrapée:\n{e}")

    def test_property_setter_p_valide_egal_1(self):
        p = 1
        n = np.random.randint(0, 500, 1, int)
        d = data_generation.DistributionBinomiale(p, n)
        try:
            d.p = p
        except Exception as e:
            self.fail(f"Une exception a été attrapée:\n{e}")

    def test_property_setter_p_valide(self):
        p = 0
        n = np.random.randint(0, 500, 1, int)
        d = data_generation.DistributionBinomiale(p, n)
        n_p = 0.56
        d.p = n_p
        self.assertEqual(d.p, n_p)

    def test_property_setter_n_negatif(self):
        p = np.random.uniform()
        n = 0
        d = data_generation.DistributionBinomiale(p, n)
        with self.assertRaises(ValueError):
            d.n = n - 1

    def test_property_setter_n_egal_0(self):
        p = np.random.uniform()
        n = 0
        d = data_generation.DistributionBinomiale(p, n)
        try:
            d.n = n
        except Exception as e:
            self.fail(f"Une exception a été attrapée:\n{e}")

    def test_property_setter_n_valide(self):
        p = np.random.uniform()
        n = 0
        d = data_generation.DistributionBinomiale(p, n)
        n_n = 12
        d.n = n_n
        self.assertEqual(d.n, n_n)

    def test_fonction_x_superieur_n(self):
        p = np.random.uniform()
        n = np.random.randint(0, 500, 1, int)
        x = n + 1
        d = data_generation.DistributionBinomiale(p, n)
        self.assertEqual(d.fonction(x), 0)

    def test_fonction_x_inferieur_0(self):
        p = np.random.uniform()
        n = np.random.randint(0, 500, 1, int)
        x = -1
        d = data_generation.DistributionBinomiale(p, n)
        self.assertEqual(d.fonction(x), 0)

    def test_fonction_x_scalaire(self):
        p = 0.25
        n = 10
        x = 4
        d = data_generation.DistributionBinomiale(p, n)
        self.assertAlmostEqual(d.fonction(x), 0.1459980011, places=10)

    def test_fonction_x_vecteur(self):
        p = 0.75
        n = 5
        x = np.arange(7)
        d = data_generation.DistributionBinomiale(p, n)
        valeurs = np.array([0.0009765625, 0.0146484375, 0.087890625, 0.263671875, 0.3955078125, 0.2373046875, 0])
        self.assertArrayAllClose(d.fonction(x), valeurs, 10)

    def test_distribution_probabilite(self):
        x_min = 0
        x_max = 3
        pas = 1
        p = 0.9
        n = 3
        d = data_generation.DistributionBinomiale(p, n)
        x_s, y_s = d.distribution_probabilite(x_min, x_max, pas)
        valeurs = np.array([0.001, 0.027, 0.243, 0.729])
        self.assertArrayEqual(x_s, range(x_min, x_max + 1, pas))
        self.assertArrayAllClose(y_s, valeurs, 10)

    def test_distribution_probabilite_avec_x_inferieurs_0(self):
        x_min = -2
        x_max = 3
        pas = 1
        p = 0.9
        n = 3
        d = data_generation.DistributionBinomiale(p, n)
        x_s, y_s = d.distribution_probabilite(x_min, x_max, pas)
        valeurs = np.array([0, 0, 0.001, 0.027, 0.243, 0.729])
        self.assertArrayEqual(x_s, range(x_min, x_max + 1, pas))
        self.assertArrayAllClose(y_s, valeurs, 10)


class TestsDistributionPoisson(CustomTestsUnitaires):

    def test_init_moyenne_negative(self):
        mu = -1e-9
        with self.assertRaises(ValueError):
            data_generation.DistributionPoisson(mu)

    def test_init_moyenne_nulle(self):
        mu = 0
        self.assertNoRaise(data_generation.DistributionPoisson, mu)

    def test_poisson_distribution_discrete(self):
        mu = 0
        d = data_generation.DistributionPoisson(mu)
        self.assertIsInstance(d, data_generation.DistributionDiscrete)

    def test_property_getter_moyenne(self):
        mu = np.random.randint(0, 100) / np.random.randint(1, 1000)
        d = data_generation.DistributionPoisson(mu)
        self.assertEqual(d.moyenne, mu)

    def test_property_setter_moyenne_invalide(self):
        mu = 0
        d = data_generation.DistributionPoisson(mu)
        with self.assertRaises(ValueError):
            d.moyenne = mu - 1e-9

    def test_property_setter_moyenne_valide(self):
        mu = 0
        d = data_generation.DistributionPoisson(mu)
        n_mu = mu + 1e-9
        try:
            d.mu = n_mu
        except Exception as e:
            self.fail(f"Une exception a été attrapée:\n{e}")

    def test_property_setter_moyenne_changee(self):
        mu = 0
        d = data_generation.DistributionPoisson(mu)
        n_mu = mu + 1e-9
        d.mu = n_mu
        self.assertEqual(d.mu, n_mu)

    def test_fonction_x_negatif(self):
        x = -1
        mu = np.random.randint(0, 100) / np.random.randint(1, 1000)
        d = data_generation.DistributionPoisson(mu)
        self.assertEqual(d.fonction(x), 0)

    def test_fonction_x_scalaire(self):
        x = 2
        mu = 1
        d = data_generation.DistributionPoisson(mu)
        self.assertAlmostEqual(d.fonction(x), 0.1839397206, 10)

    def test_fonction_x_vecteur(self):
        x = np.arange(0, 5, 1)
        mu = 4
        d = data_generation.DistributionPoisson(mu)
        valeurs = np.array([0.0183156389, 0.07326255556, 0.1465251111, 0.1953668148, 0.1953668148])
        self.assertArrayAllClose(d.fonction(x), valeurs, 10)

    def test_distribution_probabilite(self):
        x_min = 5
        x_max = 8
        pas = 1
        mu = 10.56
        d = data_generation.DistributionPoisson(mu)
        x_s, y_s = d.distribution_probabilite(x_min, x_max, pas)
        valeurs = np.array([0.0283784464, 0.04994606567, 0.07534720764, 0.0994583141])
        self.assertArrayEqual(x_s, range(x_min, x_max + 1, pas))
        self.assertArrayAllClose(y_s, valeurs, 10)

    def test_distribution_probabilite_avec_x_inferieurs_0(self):
        x_min = -2
        x_max = 3
        pas = 1
        p = 0.9
        n = 3
        d = data_generation.DistributionBinomiale(p, n)
        x_s, y_s = d.distribution_probabilite(x_min, x_max, pas)
        valeurs = np.array([0, 0, 0.001, 0.027, 0.243, 0.729])
        self.assertArrayEqual(x_s, range(x_min, x_max + 1, pas))
        self.assertArrayAllClose(y_s, valeurs, 10)


class TestsDistributionUniforme(CustomTestsUnitaires):

    def test_init_warning(self):
        alpha = 10
        beta = alpha - 1
        with self.assertWarns(UserWarning):
            data_generation.DistributionUniforme(alpha, beta)

    def test_init_beta_smaller_alpha_switch(self):
        alpha = 10
        beta = alpha - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = data_generation.DistributionUniforme(alpha, beta)
        self.assertEqual(d.alpha, beta)
        self.assertEqual(d.beta, alpha)

    def test_init_beta_egal_alpha(self):
        alpha = 10
        beta = alpha
        with self.assertRaises(ValueError):
            data_generation.DistributionUniforme(alpha, beta)

    def test_init_pas_erreur(self):
        alpha = 10
        beta = alpha * 2
        self.assertNoRaise(data_generation.DistributionUniforme, alpha, beta)

    def test_init_pas_warnings(self):
        alpha = 10
        beta = alpha + 1
        self.assertNoWarn(data_generation.DistributionUniforme, alpha, beta)

    def test_init_bonnes_valeurs(self):
        alpha = 10
        beta = alpha + 1
        d = data_generation.DistributionUniforme(alpha, beta)
        self.assertEqual(d.alpha, alpha)
        self.assertEqual(d.beta, beta)

    def test_fonction_x_avant_alpha(self):
        alpha = 10
        beta = alpha * 2
        x = alpha - 1e-9
        d = data_generation.DistributionUniforme(alpha, beta)
        self.assertEqual(d.fonction(x), 0)

    def test_fonction_x_apres_beta(self):
        alpha = 10
        beta = alpha * 2
        x = beta + 1e-9
        d = data_generation.DistributionUniforme(alpha, beta)
        self.assertEqual(d.fonction(x), 0)

    def test_fonction_x_scalaire(self):
        alpha = 10
        beta = 20
        x = np.random.choice([alpha, beta], p=[0.5, 0.5])
        d = data_generation.DistributionUniforme(alpha, beta)
        self.assertEqual(d.fonction(x), 1 / 10)

    def test_fonction_x_vecteur(self):
        alpha = 10
        beta = 20
        x = np.linspace(12, 16, 100)
        valeurs = np.full_like(x, 1 / 10)
        d = data_generation.DistributionUniforme(alpha, beta)
        self.assertArrayEqual(d.fonction(x), valeurs)

    def test_distribution_probabilite(self):
        alpha = -10
        beta = 10
        x_min = 5
        x_max = 6
        nb_x = 100
        x = np.linspace(x_min, x_max, nb_x)
        d = data_generation.DistributionUniforme(alpha, beta)
        valeurs = np.full_like(x, 1 / 20)
        x_s, y_s = d.distribution_probabilite(x_min, x_max, nb_x)
        self.assertArrayEqual(x_s, x)
        self.assertArrayEqual(y_s, valeurs)

    def test_distribution_probabilite_x_hors_range(self):
        alpha = -10
        beta = 10
        x_min = -12
        x_max = 11
        nb_x = 24
        x = np.arange(x_min, x_max + 1)
        d = data_generation.DistributionUniforme(alpha, beta)
        valeurs = np.full_like(x, 1 / 20, dtype=float)
        valeurs[:2] = 0
        valeurs[-1] = 0
        x_s, y_s = d.distribution_probabilite(x_min, x_max, nb_x)
        self.assertArrayEqual(x_s, x)
        self.assertArrayEqual(y_s, valeurs)


class TestsDistributionExponentielle(CustomTestsUnitaires):

    def test_init_taux_negatif(self):
        with self.assertRaises(ValueError):
            data_generation.DistributionExponentielle(-1e-9)

    def test_init_taux_nul(self):
        self.assertNoRaise(data_generation.DistributionExponentielle, 0)

    def test_init_taux_positif(self):
        taux = np.random.randint(1, 1000) / np.random.randint(0, 10_000)
        self.assertNoRaise(data_generation.DistributionExponentielle, taux)

    def test_property_getter_taux(self):
        taux = np.random.randint(0, 1000) / np.random.randint(0, 10_000)
        d = data_generation.DistributionExponentielle(taux)
        self.assertEqual(d.taux, taux)

    def test_property_setter_taux_invalide(self):
        taux = np.random.randint(0, 1000) / np.random.randint(0, 10_000)
        d = data_generation.DistributionExponentielle(taux)
        with self.assertRaises(ValueError):
            d.taux = -1e-9

    def test_property_setter_taux_valide(self):
        taux = np.random.randint(1, 1000) / np.random.randint(0, 10_000)
        d = data_generation.DistributionExponentielle(taux)
        try:
            d.taux = 0
        except Exception as e:
            self.fail(f"Une exception a été attrapée:\n{e}")

    def test_property_setter_taux_change(self):
        taux = np.random.randint(0, 1000) / np.random.randint(0, 10_000)
        d = data_generation.DistributionExponentielle(taux)
        n_taux = taux + np.random.randint(1, 1000) / np.random.randint(0, 10_000)
        d.taux = n_taux
        self.assertEqual(d.taux, n_taux)

    def test_fonction_x_inferieur_0(self):
        x = -1e-9
        taux = np.random.randint(0, 1000) / np.random.randint(0, 10_000)
        d = data_generation.DistributionExponentielle(taux)
        self.assertEqual(d.fonction(x), 0)

    def test_fonction_x_egal_0(self):
        x = 0.5
        taux = 4
        d = data_generation.DistributionExponentielle(taux)
        self.assertAlmostEqual(d.fonction(x), 0.54134113294645, 14)


if __name__ == '__main__':
    main()
