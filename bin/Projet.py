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
    n=int(R/dr)
    err_t_tdt=10e-7 #Condition d'arret 
    k=4e-9   
    tf=10000000
    
prm=param()
r=np.linspace(0,prm.R,prm.n)
C_num,tps=PbF(prm)
plt.plot(r,C_num)
plt.grid()

'''# ==========================================================================
# =============================================================================
# =====================  Deuxieme cas : Schema d'ordre 2  =====================
# =============================================================================
# =============================================================================

# ============================================================================= 
# ============================= Regime transitoire ============================
# ==========================================================================''' 
    
def PartieF():    
     
    # Initialisation des paramètres'
    prm = param()
    dr_testee = [0.1,0.04,0.02,0.01,0.005]   
      
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= []   
    
    # Calcul de de l'erreur
    for dr_act in dr_testee:
       
        prm.dr = dr_act
        prm.n  = int(prm.R/prm.dr)
        print(prm.dr)
        r,C_analy=C_analytique(prm)
        C_num,tps=PbF(prm)
        
        c_ordre2.append(C_num)
        r_ordre2.append(r)
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/prm.n)
        erreur_L2.append(np.linalg.norm(epsilon_h))
        erreur_Linf.append(max(abs(epsilon_h)))
        
      
    print("Erreur L2")
    for i in range(len(dr_testee)):
        print(dr_testee[i],erreur_L2[i])
    for i in range(len(dr_testee)-1):    
        print(log(erreur_L2[i-1]/erreur_L2[i])/log(dr_testee[i-1]/dr_testee[i]))
    
    # Graphiques
    plt.figure(1)
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'.',label='$L_{1}$')  
    plt.loglog(dr_testee,erreur_L2,'.',label='$L_{2}$')
    plt.loglog(dr_testee,erreur_Linf,'.',label='$L_{\infty}$')
    plt.gca().invert_xaxis()
    plt.xlabel("$\delta$r")
    plt.ylabel("Erreur")
    plt.legend()
    plt.title("Convergence de l'erreur  en fonction de $\Delta$r")
    plt.savefig('Ordre2_Conv_dr', dpi=1000)

    plt.figure(2)
    for i in range(len(c_ordre2)):
        if dr_testee[i] in [0.25,0.0002,0.0001]:
            lab='dr='+str(dr_testee[i])
            plt.plot(r_ordre2[i],c_ordre2[i],'-.',label=lab)
    plt.plot(r,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("r [$m$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.title("Profil de concentration en fonction de $\Delta$r")
    plt.savefig('Ordre2_C_dr', dpi=1000)
    

print("Veuillez attendre la vérification du code est en cours.")    
unittest.main(module=__name__)  
