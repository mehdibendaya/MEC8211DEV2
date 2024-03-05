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
    dr=0.005  #Pas en espace
    dt=0.5*dr**2/(D_eff*100) # Pas en temps
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
# ===================================== MMS ====================================
# ==========================================================================''' 
#########################CONVERGENCE EN ESPACE#################################    
def MMS_Conv_Espace():    
     
    prm=param2()   
 # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= [] 
    
    # Initialisation des paramètres
    prm = param2()  
    dr_testee=[0.2,0.1,0.05,0.01,0.005]
    n_testee=[]

    # Calcul de de l'erreur
    for dr in dr_testee:
        
        prm.dr=dr
        prm.n=int(prm.R/dr)+1
        r1,C_analy=MMS_analy(prm)
        C_num,tps,r=MMS_fct(prm)
        
        c_ordre2.append(C_num)
        r_ordre2.append(r)
        n_testee.append(prm.n)
 
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/len(epsilon_h))
        erreur_L2.append((np.sum(epsilon_h**2)/len(epsilon_h))**0.5)
        erreur_Linf.append(max(abs(epsilon_h)))
    prm.dr=0.01    
    r1,C_analy=MMS_analy(prm)

        
    # Graphiques
    plt.figure(1)
    plt.rcParams['text.usetex'] = True
    plt.loglog(dr_testee,erreur_L1,'x-',label='$L_{1}$')  
    plt.loglog(dr_testee,erreur_L2,'x-',label='$L_{2}$')
    plt.loglog(dr_testee,erreur_Linf,'x-',label='$L_{\infty}$')
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.gca().invert_xaxis()
    plt.xlabel("$\Delta$r [m]")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid()
    plt.title("Convergence de l'erreur  en fonction de $\Delta$r")
    plt.savefig('Ordre2_Conv_dr', dpi=1000)
    
   
    plt.figure(2)
    for i in range(len(c_ordre2)):
        lab='$\Delta$r='+str(dr_testee[i])+"/ $N_{points}$="+str(n_testee[i])
        plt.plot(r_ordre2[i],c_ordre2[i],'-.',label=lab)
    plt.plot(r1,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("r [$m$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.title("Profil de concentration en fonction de r")
    plt.savefig('Ordre2_C_dr', dpi=1000)

    
    # plt.figure(3)
    # a,b = np.polyfit(np.log(dr_testee[-4:]), np.log(erreur_L2[-4:]), 1)
    # a1,b1 = np.polyfit((dr_testee[-4:]), (erreur_L2[-4:]), 1)
    # y=droite(a1,b1,dr_testee)
    # plt.rcParams['text.usetex'] = True
 
    # plt.loglog(dr_testee,erreur_L2,'o',label="L2")
    # plt.plot(dr_testee,y,'-.', label="droite de regression")
    # plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)

    # # Ajouter des étiquettes et un titre au graphique
    # plt.title('Convergence d\'ordre 2\n de l\'erreur $L_2$ en fonction de $Δr$',
    #         fontsize=12, y=1.02)  # Le paramètre y règle la position verticale du titre
    # plt.xlabel('$Δr$ [m]', fontsize=10)
    # plt.ylabel('Erreur $L_2$', fontsize=10)
    
    # # Afficher l'équation de la régression en loi de puissance
    # equation_text = f'$L_2 = {np.exp(b):.4f} \\times Δr^{{{a:.4f}}}$'
    # equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=10, transform=plt.gca().transAxes, color='k')

    # # Déplacer la zone de texte
    # equation_text_obj.set_position((0.2, 0.7))
    # plt.legend()    
    # plt.savefig('Ordre2_regression', dpi=1000)
    
    

#########################CONVERGENCE EN TEMPS#################################
def MMS_Conv_temps(): 
    prm=param2()
    prm.dr=0.001   
 # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= [] 
    
    # Initialisation des paramètres
    prm = param2()  
    dt_testee=[1e8,1e7,1e6,1e5,50000]


    # Calcul de de l'erreur
    for dt in dt_testee:
        
        prm.dt=dt

        r1,C_analy=MMS_analy(prm)
        C_num,tps,r=MMS_fct(prm)
        
        c_ordre2.append(C_num)
        r_ordre2.append(r)

 
        
        epsilon_h=C_num-C_analy
        
        erreur_L1.append(np.sum(abs(epsilon_h))/len(epsilon_h))
        erreur_L2.append((np.sum(epsilon_h**2)/len(epsilon_h))**0.5)
        erreur_Linf.append(max(abs(epsilon_h)))
    prm.dr=0.01    
    r1,C_analy=MMS_analy(prm)

        
    # Graphiques
    plt.figure(1)
    plt.rcParams['text.usetex'] = True
    plt.loglog(dt_testee,erreur_L1,'x-',label='$L_{1}$')  
    plt.loglog(dt_testee,erreur_L2,'x-',label='$L_{2}$')
    plt.loglog(dt_testee,erreur_Linf,'x-',label='$L_{\infty}$')
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.xlabel("$\Delta$t [m]")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid()
    plt.title("Convergence de l'erreur  en fonction de $\Delta$t")
    plt.savefig('Ordre2_Conv_dt', dpi=1000)
    
   
    plt.figure(2)
    for i in range(len(c_ordre2)):
        lab='$\Delta$t='+str(dt_testee[i])
        plt.plot(r_ordre2[i],c_ordre2[i],'-.',label=lab)
    plt.plot(r1,C_analy,'-.',label="Sol analytique",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("r [$m$]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.title("Profil de concentration en fonction de r")
    plt.savefig('Ordre2_C_dt', dpi=1000)

    
    # plt.figure(3)
    # a,b = np.polyfit(np.log(dr_testee[-4:]), np.log(erreur_L2[-4:]), 1)
    # a1,b1 = np.polyfit((dr_testee[-4:]), (erreur_L2[-4:]), 1)
    # y=droite(a1,b1,dr_testee)
    # plt.rcParams['text.usetex'] = True
 
    # plt.loglog(dr_testee,erreur_L2,'o',label="L2")
    # plt.plot(dr_testee,y,'-.', label="droite de regression")
    # plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)

    # # Ajouter des étiquettes et un titre au graphique
    # plt.title('Convergence d\'ordre 2\n de l\'erreur $L_2$ en fonction de $Δr$',
    #         fontsize=12, y=1.02)  # Le paramètre y règle la position verticale du titre
    # plt.xlabel('$Δr$ [m]', fontsize=10)
    # plt.ylabel('Erreur $L_2$', fontsize=10)
    
    # # Afficher l'équation de la régression en loi de puissance
    # equation_text = f'$L_2 = {np.exp(b):.4f} \\times Δr^{{{a:.4f}}}$'
    # equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=10, transform=plt.gca().transAxes, color='k')

    # # Déplacer la zone de texte
    # equation_text_obj.set_position((0.2, 0.7))
    # plt.legend()    
    # plt.savefig('Ordre2_regression', dpi=1000)    

print("Veuillez attendre la vérification du code est en cours.")    
# unittest.main(module=__name__)  
# Cac()
MMS_Conv_temps()
