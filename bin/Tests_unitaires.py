# =============================================================================
# =============================================================================
# ============ MEC 8211 - DEVOIR 1 - Verification de code - H24 ===============
# Redigé par:
# Mohammed Mahdi Sahbi Ben Daya
# Acile Sfeir
# Alexandre Deschenes
# =============================================================================
# =============================================================================

# -*- coding: utf-8 -*-
"""
Ce code permet de faire des tests unitaires sur l'equation de diffusion de sel dans le beton.

"""

import unittest
import math
from Fonctions import *  # Importation des fonctions pour les tests
'''Creation d'une classe qui servira pour tous les cas  La classe pourra etre modifier au besoin'''

# def test_strictement_nulle(self):
#     prm=param()
#     result_analytique = C_analytique2(prm,N)
#     result = PbB_S(prm,N)
#     self.assertEqual(result, result_analytique)  # Vérifie si le résultat est égal au résulat analytique

# def test_diffusion_nulle(self):
#     prm=param()
#     result_analytique = C_analytique2(prm,N)
#     result = PbB_S(prm,N)
#     self.assertEqual(result, result_analytique)  # Vérifie si le résultat est égal au résulat analytique


class param():
    # S         -- Terme source
    # D         -- Diametre de la colonne
    # R         -- Rayon de la colonne
    # Ce        -- Concentration en sel de l'eau
    # D_eff     -- Coefficient de diffusion du sel dans le beton
    # dr        -- Pas en espace
    # dt        -- Pas en temps
    # n         -- Nombre de noeuds
    # err_t_tdt -- Condition d'arret

    S = 0
    D = 1
    R = D/2
    Ce = 0
    D_eff = 10
    dr = 0.005
    dt = 0.5*dr**2/(D_eff*100)
    n = int(R/dr)
    err_t_tdt = 10e-5


class TestFunction(unittest.TestCase):

    def test_nombre_element(self):
        prm = param()
        N=100
        result_analytique = C_analytique2(prm,N)
        result = PbB_S(prm,N)
        # Vérifie si le résultat a le même nombre d'élément que le résulat analytique
        self.assertEqual(len(result[0]), len(result_analytique[0]))

    def test_concentration_nulle(self):
        prm = param()
        N=100
        result_analytique = C_analytique2(prm,N)
        result = PbB_S(prm,N)
       # Vérifie si le résultat est à une distance de 1e-7 du résulat analytique
        for pair in zip(result[1], result_analytique[1]):
          self.assertTrue(math.isclose(pair[0], pair[1], abs_tol=1e-7))


if __name__ == '__main__':
    unittest.main()
