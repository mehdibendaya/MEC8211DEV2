# =============================================================================
# =============================================================================
# ============ MEC 8211 - DEVOIR 2 - Verification de code - H24 ===============
# Redigé par:
# Mohammed Mahdi Sahbi Ben Daya
# Acile Sfeir
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
    #err_t_tdt=10e-7 #Condition d'arret 
    k=4e-9   
    tf=10e6
    


# parametres modifiés pour la MMS
class param2():
    S=8e-9    #Terme source [mol/m3/s]
    D=1         #Diametre de la colonne [m]
    R=D/2       #Rayon de la colonne [m]
    # Ce=12       #Concentration en sel de l'eau [mol/m3]
    Ce=0       #Condition de Dirichlet modifiée pour la MMS
    D_eff=1e-10 #Coefficient de diffusion du sel dans le beton [m2/s]
    #Pour regime transitoire
    dr=0.01  #Pas en espace
    dt=0.5*dr**2/(D_eff*10) # Pas en temps
    n=int(R/dr)+1
    err_t_tdt=10e-7 #Condition d'arret
    k=4e-9
    tf=1e9

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
    C_num1,tps1=CAC_fct(prm)
    plt.plot(r,C_num1)
    plt.grid()
    
    with open('1e7_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])


'''# ============================================================================= 
# ============================= MMS ======================
# ==========================================================================''' 
    
def MMS():    
     
    prm=param2()
    r=np.linspace(0,prm.R,prm.n)
    C_num1,tps1,r=MMS_fct(prm)
    
    r,C_analy=MMS_analy(prm)
    plt.figure()
    plt.plot(r,C_num1,label='numerique')
    plt.plot(r,C_analy,'--',label='analytique')
    plt.legend()
    plt.grid()
    

print("Veuillez attendre la vérification du code est en cours.")    
# unittest.main(module=__name__)  
Cac()
MMS()
