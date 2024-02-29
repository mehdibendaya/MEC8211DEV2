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
''' Ce code permet de resoudre l'equation de diffusion de sel dans le beton par la MDF.
Le projet se divise 4 sous-programmes appelés du fichier Fonctions.py:
    - PartieE() : Schema d'ordre 1 resolvant le systeme transitoire 
    - PartieE_S() : Schema d'ordre 1 resolvant le systeme stationnaire
    - PartieF() : Schema d'ordre 2 resolvant le systeme transitoire 
    - PartieF_S() : Schema d'ordre 2 resolvant le systeme stationnaire
    '''
from os import environ
from time import time
from math import log
import numpy as np
import matplotlib.pyplot as plt
from Fonctions import *
from Tests_unitaires import *
import unittest
import csv


N_THREADS = '12'
environ['OMP_NUM_THREADS'] = N_THREADS
environ['OPENBLAS_NUM_THREADS'] = N_THREADS
environ['MKL_NUM_THREADS'] = N_THREADS
environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
environ['NUMEXPR_NUM_THREADS'] = N_THREADS


'''Creation d'une classe qui servira pour tous les cas  La classe pourra etre modifier au besoin'''
class param():
    S=8e-9      #Terme source [mol/m3/s]
    D=1         #Diametre de la colonne [m]
    R=D/2       #Rayon de la colonne [m]
    Ce=12       #Concentration en sel de l'eau [mol/m3]
    D_eff=1e-10 #Coefficient de diffusion du sel dans le beton [m2/s]
    #Pour regime transitoire    
    dr=0.005  #Pas en espace
    dt=0.5*dr**2/(D_eff*10) # Pas en temps
    n=int(R/dr)+1
    err_t_tdt=10e-7 #Condition d'arret 
    k=4e-9   
    tf=10e6
    

'''# ==========================================================================
# =============================================================================
# ============================  Schema d'ordre 2  =============================
# =============================================================================
# =============================================================================

# ============================================================================= 
# ============================= Vérification code à code ======================
# ==========================================================================''' 
    
def Cac():    
     
    prm=param()
    r=np.linspace(0,prm.R,prm.n)
    prm.tf=tf=10e6
    C_num1,tps1=PbF(prm)
    # prm.tf=10e8
    # C_num2,tps2=PbF(prm)
    # prm.tf=10e9
    # C_num3,tps3=PbF(prm)
    plt.plot(r,C_num1)
    plt.grid()
    
    with open('10e7_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])
    with open('10e8_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])
    with open('10e9_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])


print("Veuillez attendre la vérification du code est en cours.")    
unittest.main(module=__name__)  
Cac()
