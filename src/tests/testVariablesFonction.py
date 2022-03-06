from CodeUtilitaireSupplementaire.tests.customTestsUnitaires import CustomTestsUnitaires, main
from CodeUtilitaireSupplementaire.fonction import _Variables, VariablesDependantes, VariablesIndependantes, Fonction
from CodeUtilitaireSupplementaire.regression_interpolation import LiaisonMixte, RegressionPolynomiale, \
    RegressionGenerale
import numpy as np


class TestsVariablesBase(CustomTestsUnitaires):

    def test_init_type_non_supporte(self):
        var = ["a", "b", "c"]
        with self.assertRaises(TypeError) as e:
            _Variables(var)
            self.assertEqual(str(e), "Le type de données '<class 'numpy.str_'>' n'est pas supporté.")

    def test_init_type_supporte(self):
        var = np.arange(10, dtype=np.float64)
        self.assertNoRaise(_Variables, var)

    def test_init_attributs(self):
        l = 1000
        var = np.random.randint(0, 100, l)
        variables = _Variables(var)
        self.assertArrayEqual(variables._valeurs, var)
        self.assertEqual(variables._iteration, 0)
        self.assertEqual(variables._len, l)
        self.assertTrue(variables._bloquer_modifcation_taille)
        self.assertEqual(variables.label, "Variables (base)")

    def test_init_constructeur_copie(self):
        l = 1000
        var = np.random.randint(0, 100, l)
        variables = _Variables(var, False, "Copie")
        variables2 = _Variables(variables)
        self.assertArrayEqual(variables._valeurs, variables2._valeurs)
        self.assertEqual(variables2._iteration, 0)
        self.assertEqual(variables2._len, l)
        self.assertFalse(variables2._bloquer_modifcation_taille)
        self.assertEqual(variables2.label, variables.label)

    def test_array_wrap(self):
        array = np.arange(10)
        obj = _Variables.__array_wrap__(array)
        self.assertIsInstance(obj, _Variables)
        self.assertArrayEqual(obj._valeurs, array)

    def test_property_cls(self):
        var = np.random.randint(0, 100, 100)
        variables = _Variables(var)
        self.assertEqual(variables.cls, _Variables)

    def test_property_valeurs(self):
        var = np.random.randint(0, 100, 100)
        variables = _Variables(var)
        self.assertArrayEqual(variables.valeurs, var)

    def test_property_valeurs_modification_change_rien(self):
        var = np.random.randint(0, 100, 100)
        variables = _Variables(var)
        valeurs = variables.valeurs
        valeurs += 2
        self.assertArrayEqual(var, variables.valeurs)
        self.assertFalse(np.array_equal(valeurs, var))

    def test_property_modification_taille_est_bloquee(self):
        var = np.random.randint(0, 100, 100)
        variables = _Variables(var)
        self.assertTrue(variables.modification_taille_est_bloquee)

    def test_property_modification_taille_pas_bloquee(self):
        var = np.random.randint(0, 100, 100)
        variables = _Variables(var, False)
        self.assertFalse(variables.modification_taille_est_bloquee)

    def test_getitem_int_dernier_m1(self):
        borne_sup = 10
        var = np.arange(borne_sup)
        dernier = var[borne_sup - 1]
        variables = _Variables(var)
        item = variables[-1]
        self.assertEqual(item, dernier)

    def test_getitem_int_dernier(self):
        borne_sup = 10
        var = np.arange(borne_sup)
        dernier_index = len(var) - 1
        dernier = var[dernier_index]
        variables = _Variables(var)
        item = variables[dernier_index]
        self.assertEqual(item, dernier)

    def test_getitem_int_hors_range_positif(self):
        var = np.arange(10)
        index = len(var)
        variables = _Variables(var)
        with self.assertRaises(IndexError):
            variables[index]

    def test_getitem_int_hors_range_negatif(self):
        var = np.arange(10)
        index = -len(var) - 1
        variables = _Variables(var)
        with self.assertRaises(IndexError):
            variables[index]

    def test_getitem_int_dans_range(self):
        nb_vars = 10
        index = nb_vars // 2
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertEqual(variables[index], var[index])

    def test_getitem_slice_hors_range_pas_erreur(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var)
        slice_ = slice(nb_vars + 1, nb_vars + 20)
        self.assertNoRaise(variables.__getitem__, slice_)

    def test_getitem_slice_hors_range_vide(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var)
        slice_ = slice(nb_vars + 1, nb_vars + 20)
        items = variables[slice_]
        self.assertArrayEqual(items, [])

    def test_getitem_all(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertArrayEqual(variables[:], var)

    def test_getitem_retourne_array_modification_change_pas_original(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var)
        items = variables[:]
        items += 10
        self.assertArrayEqual(variables.valeurs, var)
        self.assertFalse(np.array_equal(items, var))

    def test_type_cast_priorite_pas_une_classe(self):
        var = np.arange(10)
        nouveau_type = var.dtype  # N'est pas correct! On doit faire var.dtype.type.
        variables = _Variables(var)
        with self.assertRaises(TypeError) as e:
            variables._type_cast_priorite(nouveau_type)
            self.assertEqual(str(e), "L'argument 'nouveau_type' doit être une classe.")

    def test_type_cast_priorite_type_non_supporte(self):
        var = np.arange(10)
        variables = _Variables(var)
        with self.assertRaises(TypeError) as e:
            variables._type_cast_priorite(str)
            self.assertEqual(str(e), "Le type de données 'str' n'est pas supporté.")

    def test_type_cast_priorite_meme_types(self):
        var = np.arange(10, dtype=np.uint8)
        autre_type = var.dtype.type
        dtype_original = var.dtype.type
        variables = _Variables(var)
        variables._type_cast_priorite(autre_type)
        self.assertEqual(variables.valeurs.dtype, dtype_original)

    def test_type_cast_priorite_type_inclus(self):
        var = np.arange(10j)
        autre_type = float
        dtype_original = var.dtype
        variables = _Variables(var)
        variables._type_cast_priorite(autre_type)
        self.assertEqual(variables.valeurs.dtype, dtype_original)

    def test_type_cast_priorite_doit_changer(self):
        var = np.arange(10, dtype=np.uint8)
        autre_type = np.int8
        variables = _Variables(var)
        nouveau_dtype = np.int16  # Comme à l'origine uint8 va de 0 à 255 et on veut ajouter un élément de type int8
        # (-128 à 127), on ne peut garder uint, car il ne permet pas les négatifs. De plus, on ne peut mettre en int8,
        # car initialement on peut aller jusqu'à 255, ce qui n'est pas permis par int8. On doit donc aller au type
        # non signé supérieur qui inclus les deux ranges possibles, soit int16 (-32768 à 32767).
        variables._type_cast_priorite(autre_type)
        self.assertEqual(variables.valeurs.dtype, nouveau_dtype)

    def test_setitem_cle_out_of_range(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        with self.assertRaises(IndexError):
            variables[nb_vars] = 0

    def test_setitem_slice_out_of_range_pas_erreur(self):
        # Comportement de ndarray. Ne lève pas d'exception.
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertNoRaise(variables.__setitem__, slice(nb_vars + 1, nb_vars + 10), 0)

    def test_setitem_slice_partie_out_of_range_nouvelles_valeurs(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables[nb_vars - 2:nb_vars + 1] -= 1
        nouvelles_valeurs = var.copy()
        nouvelles_valeurs[-2:] -= 1
        self.assertArrayEqual(variables.valeurs, nouvelles_valeurs)

    def test_setitem_change_type(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        original_dtype = variables.valeurs.dtype
        variables[-1] /= 4.52
        nouveau_dtype = variables.valeurs.dtype
        self.assertNotEqual(original_dtype, nouveau_dtype)
        self.assertEqual(nouveau_dtype, float)

    def test_len(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertEqual(len(variables), nb_vars)

    def test_iter_retour(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        i = iter(variables)
        # __iter__ retourne self, donc i est IDENTIQUE à variables.
        self.assertIs(i, variables)

    def test_iter_remet_iterations_a_0(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables._iteration = 100
        iter(variables)
        self.assertEqual(variables._iteration, 0)

    def test_next_retour(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        val = next(variables)
        self.assertEqual(val, variables[0])

    def test_next_augmente_iterations(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        next(variables)
        self.assertEqual(variables._iteration, 1)

    def test_next_erreur_lorsque_epuise(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables._iteration = nb_vars
        with self.assertRaises(StopIteration):
            next(variables)

    def test_for_loop_possible(self):
        def for_loop(iterateur):
            for _ in iterateur:
                continue

        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertNoRaise(for_loop, variables)

    def test_for_loop_valeurs(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        elements = [i for i in variables]
        self.assertArrayEqual(variables.valeurs, elements)

    def test_copie_memes_valeurs(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables2 = variables.copie()
        self.assertArrayEqual(variables.valeurs, variables2.valeurs)

    def test_copie_changer_valeurs_pas_impact(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables2 = variables.copie()
        variables[:] -= 10
        self.assertFalse(np.array_equal(variables.valeurs, variables2.valeurs))

    def test_copy(self):
        import copy
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables2 = copy.copy(variables)
        self.assertArrayEqual(variables.valeurs, variables2.valeurs)

    def test_deepcopy(self):
        import copy
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables2 = copy.deepcopy(variables)
        self.assertArrayEqual(variables.valeurs, variables2.valeurs)

    def test_egalite_memes_valeurs(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables2 = variables.copie()
        self.assertTrue(all(variables == variables2))

    def test_egalite_pas_memes_valeurs(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables2 = _Variables(var + 1e-8)
        self.assertTrue(all(~(variables == variables2)))
        # On transforme les False en True et on s'assure qu'ils sont tous True (donc False!)

    def test_egalite_une_valeur_differente(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables2 = variables.copie()
        variables2[5] -= 1e-8
        comparaison = variables == variables2
        count = dict(zip(*np.unique(comparaison, return_counts=True)))
        self.assertDictEqual(count, {False: 1, True: nb_vars - 1})

    def test_egalite_nb_elements_differents(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        variables2 = _Variables(var[:-1])
        self.assertFalse(variables == variables2)

    def test_egalite_avec_array_numpy(self):
        nb_vars = 18
        var = np.arange(-nb_vars, 0)
        variables = _Variables(var)
        self.assertTrue(all(variables == var))

    def test_egalite_avec_array_numpy_differentes_valeurs(self):
        nb_vars = 18
        var = np.arange(-nb_vars, 0)
        variables = _Variables(var)
        self.assertTrue(all(~(variables == var + 1e-8)))

    def test_egalite_avec_array_numpy_une_seule_val_differente(self):
        nb_vars = 253
        var = np.arange(nb_vars)
        variables = _Variables(var)
        var[102] -= 1e-8
        comparaison = variables == var
        count = dict(zip(*np.unique(comparaison, return_counts=True)))
        self.assertDictEqual(count, {False: 1, True: nb_vars - 1})

    def test_egalite_avec_array_numpy_taille_differente(self):
        nb_vars = 17
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertFalse(variables == var[:-2])

    def test_egalite_avec_chiffre_memes_valeurs(self):
        nb_vars = 99
        var = np.ones(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(variables == 1))

    def test_egalite_avec_chiffre_pas_tous_meme_valeur(self):
        nb_vars = 99
        var = np.ones(nb_vars)
        variables = _Variables(var)
        variables[10:20] = 0
        comparaison = variables == var
        count = dict(zip(*np.unique(comparaison, return_counts=True)))
        self.assertDictEqual(count, {False: 10, True: nb_vars - 10})

    def test_egalite_avec_chiffre_pas_egal(self):
        nb_vars = 99
        var = np.ones(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(~(variables == 0)))

    def test_pas_egal_memes_valeurs(self):
        nb_vars = 99
        var = np.ones(nb_vars)
        variables = _Variables(var)
        variables2 = _Variables(var)
        self.assertTrue(all(~(variables != variables2)))

    def test_pas_egal_pas_memes_valeurs(self):
        nb_vars = 99
        var = np.ones(nb_vars)
        variables = _Variables(var)
        variables2 = _Variables(var + 1e-9)
        self.assertTrue(all(variables != variables2))

    def test_pas_egal_tailles_differentes(self):
        nb_vars = 99
        var = np.ones(nb_vars)
        variables = _Variables(var)
        variables2 = _Variables(var[:-1])
        self.assertTrue(variables != variables2)

    def test_pas_egal_array_numpy(self):
        nb_vars = 99
        var = np.ones(nb_vars)
        variables = _Variables(var)
        self.assertTrue(variables != var[2:-1])

    def test_pas_egal_chiffre(self):
        nb_vars = 99
        var = np.ones(nb_vars)
        var[10:20] = 10
        variables = _Variables(var)
        comparaison = variables != 10
        count = dict(zip(*np.unique(comparaison, return_counts=True)))
        self.assertDictEqual(count, {False: 10, True: nb_vars - 10})

    def test_neg(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(-variables == _Variables(-var)))

    def test_pos(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(+variables == variables))

    def test_add_pas_variables(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var, label="Test")
        self.assertTrue(all(variables + 10 == _Variables(var + 10)))

    def test_add_variables(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(variables + variables == _Variables(var + var)))

    def test_radd(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        add = variables + 10
        radd = 10 + variables
        self.assertTrue(all(add == radd))

    def test_sub(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(variables - 10 == _Variables(var - 10)))

    def test_rsub(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(10 - variables == _Variables(10 - var)))

    def test_mul_pas_variables(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(variables * 10 == _Variables(var * 10)))

    def test_mul_variables(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(variables * variables == _Variables(var * var)))

    def test_rmul(self):
        nb_vars = 99
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(10 * variables == _Variables(10 * var)))

    def test_div_variables(self):
        nb_vars = 99
        var = np.arange(nb_vars) + 1
        variables = _Variables(var)
        self.assertTrue(all(variables / variables == 1))

    def test_div_pas_variables(self):
        nb_vars = 99
        var = np.arange(nb_vars) + 1
        variables = _Variables(var)
        self.assertTrue(all(variables / 10.2 == _Variables(var / 10.2)))

    def test_rdiv(self):
        nb_vars = 99
        var = np.arange(nb_vars) + 1
        variables = _Variables(var)
        self.assertTrue(all(0.25 / variables == _Variables(0.25 / var)))

    def test_pow_variables(self):
        nb_vars = 9
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(variables ** variables == _Variables(var ** var)))

    def test_pow_pas_variables(self):
        nb_vars = 9
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(variables ** 10 == _Variables(var ** 10)))

    def test_rpow_all_entiers(self):
        nb_vars = 9
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(10 ** variables == _Variables(10 ** var)))

    def test_rpow_pas_tous_entiers(self):
        nb_vars = 9.5
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(10 ** variables == _Variables(10 ** var)))

    def test_floordiv_variables(self):
        nb_vars = 9
        var = np.arange(nb_vars) + 1
        variables = _Variables(var)
        self.assertTrue(all(variables // variables == 1))

    def test_floordiv_pas_variables(self):
        nb_vars = 9
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertTrue(all(variables // 10 == _Variables(var // 10)))

    def test_rfloordiv(self):
        nb_vars = 20
        var = np.arange(nb_vars) + 1
        variables = _Variables(var)
        self.assertTrue(all(10 // variables == _Variables(10 // var)))

    def test_mod_variables(self):
        nb_vars = 20
        var = np.arange(nb_vars) + 1
        variables = _Variables(var)
        self.assertTrue(all(variables % variables == 0))

    def test_mod_pas_variables(self):
        nb_vars = 20
        var = np.arange(nb_vars) + 1
        variables = _Variables(var)
        self.assertTrue(all(variables % 3 == _Variables(var % 3)))

    def test_rmod(self):
        nb_vars = 819
        var = np.arange(nb_vars) + 1
        variables = _Variables(var)
        self.assertTrue(all(785 % variables == _Variables(785 % var)))

    def test_abs(self):
        var = np.linspace(-100, 100, 100_000)
        variables = _Variables(var)
        abs_variables = abs(variables)
        self.assertTrue(all(abs_variables == _Variables(abs(var))))

    def test_ceil(self):
        from math import ceil
        var = np.array([0.5, 0.8, 0.9, 0.999999999, 1, 1.2])
        variables = _Variables(var)
        ceils = ceil(variables)
        self.assertArrayEqual(ceils, [1, 1, 1, 1, 1, 2])

    def test_floor(self):
        from math import floor
        var = np.array([0.5, 0.8, 0.9, 0.999999999, 1, 1.2])
        variables = _Variables(var)
        floors = floor(variables)
        self.assertArrayEqual(floors, [0, 0, 0, 0, 1, 1])

    def test_round_no_digits(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        round_ = round(variables)
        self.assertArrayEqual(round_, [1, 1, 1, 1, 1, 2])

    def test_round_with_1_digit(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        round_ = round(variables, 1)
        self.assertArrayEqual(round_, [0.5, 0.9, 0.9, 1.0, 1.0, 1.6])

    def test_gt_pas_meme_taille(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        with self.assertRaises(ValueError):
            variables > [1, 2]

    def test_gt_scalaire(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        comp = variables > 2
        self.assertArrayEqual(comp, [False] * len(comp))

    def test_gt_variables(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        comp = variables > variables
        self.assertArrayEqual(comp, [False] * len(comp))

    def test_le_erreur_pas_meme_taille(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        with self.assertRaises(ValueError):
            variables <= [1, 2]

    def test_le_scalaire(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        comp = variables <= 1
        self.assertArrayEqual(comp, [True, True, True, True, True, False])

    def test_le_variables(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        comp = variables <= variables
        self.assertArrayEqual(comp, [True] * len(comp))

    def test_ge_erreur_pas_meme_taille(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        with self.assertRaises(ValueError):
            variables >= [1, 2]

    def test_ge_scalaire(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        comp = variables >= 0.89
        self.assertArrayEqual(comp, [False, False, True, True, True, True])

    def test_ge_variables(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        comp = variables >= variables
        self.assertArrayEqual(comp, [True] * len(comp))

    def test_lt_pas_meme_taille(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        with self.assertRaises(ValueError):
            variables < [1, 2]

    def test_lt_scalaire(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        comp = variables < 0.9
        self.assertArrayEqual(comp, [True, True, False, False, False, False])

    def test_lt_variables(self):
        var = np.array([0.51, 0.88, 0.91, 0.999999999, 1, 1.63])
        variables = _Variables(var)
        comp = variables < variables
        self.assertArrayEqual(comp, [False, False, False, False, False, False])

    def test_iadd(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables += 1.9
        self.assertTrue(all(variables == 2.9))

    def test_iadd_variables(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables += variables
        self.assertTrue(all(variables == 2))

    def test_isub(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables -= 1
        self.assertTrue(all(variables == 0))

    def test_isub_variables(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables -= variables
        self.assertTrue(all(variables == 0))

    def test_idiv(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables /= 2
        self.assertTrue(all(variables == 0.5))

    def test_idiv_variables(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables /= variables
        self.assertTrue(all(variables == 1))

    def test_ifloordiv(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables //= 2
        self.assertTrue(all(variables == 0))

    def test_ifloordiv_variables(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables //= _Variables(np.full_like(var, 0.3))
        self.assertTrue(all(variables == 3))

    def test_imod(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables %= 2
        self.assertTrue(all(variables == 1))

    def test_imod_variables(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables %= variables
        self.assertTrue(all(variables == 0))

    def test_imul(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables *= 10
        self.assertTrue(all(variables == 10))

    def test_imul_variables(self):
        var = np.ones(10)
        variables = _Variables(var)
        variables *= _Variables(np.full_like(var, np.e))
        self.assertTrue(all(variables == np.e))

    def test_ipow(self):
        var = np.ones(10) + 1
        variables = _Variables(var)
        variables **= 2
        self.assertTrue(all(variables == 4))

    def test_ipow_variables(self):
        var = np.ones(10) + 2
        variables = _Variables(var)
        variables **= variables
        self.assertTrue(all(variables == 27))

    def test_str(self):
        var = np.ones(10) + 2
        variables = _Variables(var)
        msg = f"Variables (base) {var}"
        self.assertEqual(str(variables), msg)

    def test_inclus_scalaire_retourne_booleen_seul(self):
        var = np.ones(10)
        variables = _Variables(var)
        self.assertTrue(variables.inclus(1, True))

    def test_inclus_scalaire_retourne_array(self):
        var = np.ones(10)
        variables = _Variables(var)
        self.assertArrayEqual(variables.inclus(1), [True])

    def test_inclus_scalaire_pas_dedans(self):
        var = np.ones(10)
        variables = _Variables(var)
        self.assertArrayEqual(variables.inclus(1.01), [False])

    def test_inclus_variables_retourne_booleen_seul_tous_dedans(self):
        var = np.ones(10)
        variables = _Variables(var)
        self.assertTrue(variables.inclus(variables, True))

    def test_inclus_variables_retourne_array_tous_dedans(self):
        var = np.ones(10)
        variables = _Variables(var)
        self.assertArrayEqual(variables.inclus(variables), [True] * 10)

    def test_inclus_variables_retourne_booleen_unique_pas_tous_dedans(self):
        var = np.arange(10)
        variables = _Variables(var)
        variables2 = _Variables(var + 2)
        self.assertFalse(variables.inclus(variables2, True))

    def test_inclus_variables_retourne_array_pas_tous_dedans(self):
        var = np.arange(10)
        variables = _Variables(var)
        variables2 = _Variables(var + 2)
        suppose = [True, True, True, True, True, True, True, True, False, False]
        self.assertArrayEqual(variables.inclus(variables2), suppose)

    def test_contains_scalaire_pas_dedans(self):
        var = np.ones(10)
        variables = _Variables(var)
        self.assertFalse(10 in variables)

    def test_contains_scalaire_dedans(self):
        var = np.ones(10)
        variables = _Variables(var)
        self.assertTrue(1 in variables)

    def test_contains_variables_toutes_dedans(self):
        var = np.arange(10)
        variables = _Variables(var)
        variables2 = _Variables(np.arange(5))
        isin = variables2 in variables
        np.array(var)
        self.assertTrue(isin)

    def test_contains_variables_pas_dedans(self):
        var = np.arange(10)
        variables = _Variables(var)
        variables2 = _Variables(np.arange(10, 20))
        isin = variables2 in variables
        self.assertFalse(isin)

    def test_ajouter_variables_modification_taille_pas_permise(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var)
        self.assertFalse(variables.ajouter_variables(var))
        self.assertEqual(variables._len, nb_vars)

    def test_ajouter_variables_a_la_fin(self):
        var = np.arange(10)
        var2 = np.arange(10, 20)
        variables = _Variables(var, False)
        variables2 = _Variables(var2)
        ajout = variables.ajouter_variables(variables2)
        self.assertTrue(ajout)
        self.assertArrayEqual(variables, np.concatenate((var, var2)))

    def test_ajouter_variables_au_debut(self):
        var = np.arange(10)
        var2 = np.arange(10, 20)
        variables = _Variables(var, False)
        variables2 = _Variables(var2)
        variables.ajouter_variables(variables2, 0)
        self.assertArrayEqual(variables, np.concatenate((var2, var)))

    def test_ajouter_variables_type_different_cast(self):
        var = np.arange(10)
        variables = _Variables(var, False)
        ajout = 5j + 6
        position = 2
        dtype_initial = var.dtype
        variables.ajouter_variables(ajout, position)
        dtype_final = variables._valeurs.dtype
        self.assertNotEqual(dtype_final, dtype_initial)

    def test_ajouter_positions_arbitraires(self):
        var = np.arange(10)
        var2 = [1.5, 2.5, 3.5]
        indices = [2, 3, 4]
        variables = _Variables(var, False)
        variables2 = _Variables(var2)
        variables.ajouter_variables(variables2, indices)
        self.assertArrayEqual(variables, [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9])

    def test_ajouter_variables_change_taille(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        var2 = [1.5, 2.5, 3.5]
        indices = [2, 3, 4]
        variables = _Variables(var, False)
        variables2 = _Variables(var2)
        variables.ajouter_variables(variables2, indices)
        self.assertEqual(variables._len, nb_vars + 3)

    def test_ajouter_position_hors_range(self):
        var = np.arange(10)
        variables = _Variables(var, False)
        with self.assertRaises(IndexError):
            variables.ajouter_variables(10, len(var) + 1)

    def test_trouver_indice_premiere_occurence_objet_pas_present(self):
        var = np.arange(10)
        variables = _Variables(var)
        where = variables._trouver_indices_premiere_occurence(variables, 10)
        self.assertListEqual(where, [])

    def test_trouver_indice_premiere_occurence_plusieurs_memes_valeurs(self):
        var = [1, 2, 3, 1, 1, 2, 3]
        variables = _Variables(var)
        where = variables._trouver_indices_premiere_occurence(variables, 1)
        self.assertListEqual(where, [0])

    def test_trouver_indice_premiere_occurence_plusieurs_valeurs(self):
        var = [1, 2, 3, 1, 1, 2, 3]
        variables = _Variables(var)
        where = variables._trouver_indices_premiere_occurence(variables, (1, 2, 3))
        self.assertListEqual(where, [0, 1, 2])

    def test_enlever_variables_modification_taille_pas_permise(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var)
        positions = variables.enlever_variables(0)
        self.assertIsNone(positions)
        self.assertEqual(variables._len, nb_vars)

    def test_enlever_variables_positions_et_valeurs_specifiees(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var, False)
        with self.assertRaises(ValueError) as e:
            variables.enlever_variables(0, -1)
            msg = "Veuillez spécifier la positions des éléments à retirer ou les valeurs à retirer, pas les deux."
            self.assertEqual(str(e), msg)

    def test_enlever_variables_positions_et_valeurs_none(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var, False)
        positions = variables.enlever_variables()
        self.assertIsNone(positions)

    def test_enlever_variables_positions_none_valeurs_instance_variables(self):
        nb_vars = 10
        var = np.arange(nb_vars)
        variables = _Variables(var, False)
        var2 = var[:5]
        variables2 = _Variables(var2)
        positions = variables.enlever_variables(None, variables2)
        self.assertArrayEqual(variables._valeurs, var[5:])
        self.assertEqual(len(variables), 5)
        self.assertArrayEqual(positions, [0, 1, 2, 3, 4])

    def test_enlever_variables_positions_none_valeurs_scalaire_retirer_toutes_occurences(self):
        var = [1, 2, 3, 1, 2, 3]
        variables = _Variables(var, False)
        a_enlever = 1
        positions = variables.enlever_variables(None, a_enlever, True)
        self.assertArrayEqual(variables._valeurs, [2, 3, 2, 3])
        self.assertEqual(len(variables), 4)
        self.assertArrayEqual(positions, [0, 3])

    def test_enlever_variables_positions_none_valeurs_array_retirer_premiere_occurence(self):
        var = [1, 2, 3, 1, 2, 3]
        variables = _Variables(var, False)
        a_enlever = [1, 2, 3]
        positions = variables.enlever_variables(None, a_enlever, False)
        self.assertArrayEqual(variables._valeurs, [1, 2, 3])
        self.assertEqual(len(variables), 3)
        self.assertArrayEqual(positions, [0, 1, 2])

    def test_enlever_variables_positions_none_enlever_tout(self):
        var = np.arange(10)
        variables = _Variables(var, False)
        positions = variables.enlever_variables(None, variables)
        self.assertArrayEqual(variables._valeurs, [])
        self.assertEqual(len(variables), 0)
        self.assertArrayEqual(positions, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_enlever_variables_positions_pas_none_scalaire(self):
        var = np.arange(10)
        variables = _Variables(var, False)
        positions = variables.enlever_variables(-1)
        self.assertArrayEqual(variables._valeurs, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(variables), 9)
        self.assertArrayEqual(positions, [-1])

    def test_enlever_variables_positions_pas_none_array(self):
        var = np.arange(10)
        variables = _Variables(var, False)
        positions = variables.enlever_variables((-1, 0))
        self.assertArrayEqual(variables._valeurs, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(variables), 8)
        self.assertArrayEqual(positions, [-1, 0])

    def test_enlever_variables_positions_toutes(self):
        var = np.arange(10)
        indices = range(len(var))
        variables = _Variables(var, False)
        positions = variables.enlever_variables(indices)
        self.assertArrayEqual(variables._valeurs, [])
        self.assertEqual(len(variables), 0)
        self.assertArrayEqual(positions, indices)

    def test_enlever_variables_position_array_avec_hors_range(self):
        var = np.arange(10)
        indices = range(4, len(var) + 4)
        variables = _Variables(var, False)
        with self.assertRaises(IndexError):
            variables.enlever_variables(indices)

    def test_enlever_variables_position_scalaire_out_of_bounds(self):
        var = np.arange(10)
        indices = -len(var) - 1
        variables = _Variables(var, False)
        with self.assertRaises(IndexError):
            variables.enlever_variables(indices)

    def test_egalite_totale_vraie(self):
        var = np.arange(10)
        variables = _Variables(var)
        self.assertTrue(variables.egalite_totale(var))

    def test_egalite_totale_faux(self):
        var = np.arange(10)
        var2 = var.astype(float)
        var2[0] -= 1e-9
        variables = _Variables(var)
        self.assertFalse(variables.egalite_totale(var2))

    def test_concatener_methode_de_classe(self):
        var = np.arange(5)
        variables = _Variables(var)
        concat = _Variables.concatener(variables, var + 10, variables)
        self.assertArrayEqual(concat, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4])

    def test_concatener_methode_de_classe_avec_scalaire(self):
        var = np.arange(5)
        scalaire = 10
        concat = _Variables.concatener(var, scalaire)
        self.assertArrayEqual(concat, [0, 1, 2, 3, 4, 10])

    def test_concatener_methode_de_classe_un_seul_scalaire(self):
        scalaire = 10
        concat = _Variables.concatener(scalaire)
        self.assertArrayEqual(concat, [10])

    def test_concatener_avec_autres_objets(self):
        a = range(10)
        b = list(a)
        c = np.array(b)
        d = tuple(c)
        e = -10.25
        variables = _Variables(c)
        concat = variables.concatener_a_courant(a, b, c, d, e)
        self.assertArrayEqual(concat, list(range(10)) * 5 + [-10.25])


class TestsVariablesIndependantes(CustomTestsUnitaires):

    def test_init_avec_doublons(self):
        var = np.arange(1000)
        var[-1] = var[0]
        with self.assertRaises(ValueError) as e:
            VariablesIndependantes(var)
            self.assertEqual(str(e), "Les variables indépendantes doivent être uniques.")

    def test_bonne_instance(self):
        var = np.arange(1000)
        variables = VariablesIndependantes(var)
        self.assertIsInstance(variables, _Variables)
        self.assertIsInstance(variables, VariablesIndependantes)

    def test_str(self):
        var = np.arange(10)
        variables = VariablesIndependantes(var)
        self.assertEqual(str(variables), f"Variables indépendantes {var}")


class TestsVariablesDependantes(CustomTestsUnitaires):

    def test_bonne_instance_init(self):
        var = np.arange(1000)
        variables = VariablesDependantes(var)
        self.assertIsInstance(variables, _Variables)
        self.assertIsInstance(variables, VariablesDependantes)

    def test_str(self):
        var = np.arange(10)
        variables = VariablesDependantes(var)
        self.assertEqual(str(variables), f"Variables dépendantes {var}")

    def test_from_variables_independantes_liste(self):
        f = lambda x: np.add(x, 2)
        var = [0, 1, 2, 3, 4]
        variables = VariablesDependantes.from_variables_independantes(var, f)
        self.assertIsInstance(variables, VariablesDependantes)
        self.assertArrayEqual(variables, [2, 3, 4, 5, 6])

    def test_from_variables_independantes_variables_independantes(self):
        f = lambda x: np.multiply(x, 4) + 2
        var = VariablesIndependantes([0, 1, 2, 3, 4])
        variables = VariablesDependantes.from_variables_independantes(var, f)
        self.assertIsInstance(variables, VariablesDependantes)
        self.assertArrayEqual(variables, [2, 6, 10, 14, 18])


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
            self.assertEqual(str(e), msg)

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

    def test_call_pas_liaison(self):
        x = np.arange(10)
        y = x ** 2
        f = Fonction(x, y)
        msg = "Veuillez spécifier une manière de 'lier' les valeurs."
        with self.assertRaises(ValueError) as e:
            f(x)
            self.assertEqual(str(e), msg)

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
        self.assertArrayAllClose(y_eval, y, 14)

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


if __name__ == '__main__':
    main()
