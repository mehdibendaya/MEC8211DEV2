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
    dr=0.001  #Pas en espace
    dt=0.5*dr**2/(D_eff*10) # Pas en temps
    n=int(R/dr)+1
    #err_t_tdt=10e-7 #Condition d'arret 
    k=4e-9   
    tf=10e9
    


# parametres modifiés pour la MMS
class param2():
    S=8e-9      #Terme source [mol/m3/s]
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
    nt=5 # hardcoding pour resultats rapide : fonctionne pour tf=1e9 avec nt=5,6,9 noeuds car donne des valeurs de temps qui sont des multiples de dt
    # nt=int(tf/(10000*dt))+1 

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
    
    prm.tf=tf=10e8
    
    plt.grid()
    
    
    prm.dr=0.001
    prm.n=int(prm.R/prm.dr)+1
    r=np.linspace(0,prm.R,prm.n)
    C_num1,tps1=CAC_fct(prm)
    lab='$\Delta$r='+str(0.001)
    plt.plot(r,C_num1,label=lab)
    with open('0.001_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])
            
    prm.dr=0.002
    prm.n=int(prm.R/prm.dr)+1
    r=np.linspace(0,prm.R,prm.n)
    C_num1,tps1=CAC_fct(prm)
    lab='$\Delta$r='+str(0.002)
    plt.plot(r,C_num1,label=lab)
    with open('0.002_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])        
    
    
    prm.dr=0.005
    prm.n=int(prm.R/prm.dr)+1
    r=np.linspace(0,prm.R,prm.n)
    C_num1,tps1=CAC_fct(prm)
    lab='$\Delta$r='+str(0.005)
    plt.plot(r,C_num1,label=lab)
    with open('0.005_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])
    
    prm.dr=0.01
    prm.n=int(prm.R/prm.dr)+1
    r=np.linspace(0,prm.R,prm.n)
    C_num1,tps1=CAC_fct(prm)
    lab='$\Delta$r='+str(0.01)
    plt.plot(r,C_num1,label=lab)      
    with open('0.01_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])
            
    prm.dr=0.02
    prm.n=int(prm.R/prm.dr)+1
    r=np.linspace(0,prm.R,prm.n)
    C_num1,tps1=CAC_fct(prm)
    lab='$\Delta$r='+str(0.02)
    plt.plot(r,C_num1,label=lab)        
    with open('0.02_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])   
            
    prm.dr=0.05
    prm.n=int(prm.R/prm.dr)+1
    r=np.linspace(0,prm.R,prm.n)
    C_num1,tps1=CAC_fct(prm)
    lab='$\Delta$r='+str(0.05)
    plt.plot(r,C_num1,label=lab)      
    with open('0.05_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])  
            
    prm.dr=0.1
    prm.n=int(prm.R/prm.dr)+1
    r=np.linspace(0,prm.R,prm.n)
    C_num1,tps1=CAC_fct(prm)
    lab='$\Delta$r='+str(0.1)
    plt.plot(r,C_num1,label=lab)      
    with open('0.1_D.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for element2, element1 in zip(C_num1, r):
            writer.writerow([element1, element2])
            
    plt.xlabel("r [$m$]")
    plt.ylabel("C [mol/$m^3$]")
    plt.legend()
    plt.title("Convergence de l'erreur en fonction de $\Delta$r - Comparaison code a code")
            


'''# ======================================================================================================== 
# ================================================= MMS =====================================================
# ========================================================================================================''' 
#############################################################################################################
########################################  CONVERGENCE EN ESPACE  ############################################   
def MMS_Conv_Espace():    
       
 # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= [] 
    
    # Initialisation des paramètres
    prm = param2()  
    dr_testee=[0.1,0.05,0.01,0.005,0.0025,0.00125,0.001]
    n_testee=[]
    prm.tf=1e9
    tdom = np.linspace(0,prm.tf,prm.nt)
    print("tdom=",tdom)

    # Calcul de de l'erreur
    for dr in dr_testee:
        
        prm.dr=dr
        prm.n=int(prm.R/dr)+1
        rdom = np.linspace(0,prm.R,prm.n)
        # print("Rdom=",rdom)
        C_analy,z_source, f_source=MMS_analy(prm,rdom,tdom)
        C_num,tps,r=MMS_fct(prm,rdom,tdom) 
        # print("Canalyt=",C_analy)
        # print("Cnum=",C_num)
        
        c_ordre2.append(C_num[-1])
        r_ordre2.append(r)
        n_testee.append(prm.n)
 
        # print("C_num[-1]=",C_num[-1])
        # print("C_analy[-1]=",C_analy[-1])
        epsilon_h=C_num[-1]-C_analy[-1]
        
        erreur_L1.append(np.sum(abs(epsilon_h))/len(epsilon_h))
        erreur_L2.append((np.sum(epsilon_h**2)/len(epsilon_h))**0.5)
        erreur_Linf.append(max(abs(epsilon_h)))
  
    C_analy,z_source, f_source=MMS_analy(prm,rdom,tdom)

        
    # Graphiques
    plt.figure(1)
    plt.loglog(dr_testee,erreur_L1,'x-',label='$L_1$')  
    plt.loglog(dr_testee,erreur_L2,'x-',label='$L_2$')
    plt.loglog(dr_testee,erreur_Linf,'x-',label='$L_\infty$')
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.gca().invert_xaxis()
    plt.xlabel("$\Delta$r [m]")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid()
    plt.title(f"Convergence de l'erreur en fonction de $\Delta$r à $t={tdom[-1]}$ sec.")
    # plt.savefig('Ordre2_Conv_dr', dpi=1000)
    
   
    plt.figure(2)
    for i in range(len(c_ordre2)):
        lab='$\Delta$r='+str(dr_testee[i])+"/ $N_{points}$="+str(n_testee[i])
        plt.plot(r_ordre2[i],c_ordre2[i],'-.',label=lab)
    plt.plot(rdom,C_analy[-1],'-.',label="Sol analytique (MMS)",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("r [$m$]")
    plt.ylabel("C [mol/$m^3$]")
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.title(f"Profil de concentration en fonction de $r$ à $t={tdom[-1]}$ sec.")
    # plt.savefig('Ordre2_C_dr', dpi=1000)

    
    plt.figure(3)
    a,b = np.polyfit(np.log(dr_testee[-4:]), np.log(erreur_L2[-4:]), 1)
    # Fonction de régression en termes de logarithmes
    fit_function_log = lambda x: a * x + b

    # Fonction de régression en termes originaux
    fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

 
    plt.loglog(dr_testee,erreur_L2,'o',label="L2")
    plt.plot(dr_testee, fit_function(dr_testee), linestyle='--', color='r', label='Régression en loi de puissance')
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.gca().invert_xaxis()
    plt.grid()

    # Ajouter des étiquettes et un titre au graphique
    plt.title('Convergence d\'ordre 2\n de l\'erreur $L_2$ en fonction de $Δr$',fontsize=12, y=1.02)  # Le paramètre y règle la position verticale du titre
    plt.xlabel('$Δr$ [m]', fontsize=10)
    plt.ylabel('Erreur $L_2$', fontsize=10)
    
    # Afficher l'équation de la régression en loi de puissance
    equation_text = f'$L_2 = {np.exp(b):.4f} \\times Δr^{{{a:.4f}}}$'
    equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=10, transform=plt.gca().transAxes, color='k')

    # Déplacer la zone de texte
    equation_text_obj.set_position((0.35, 0.8))
    
    plt.legend()    
    # plt.savefig('Ordre2_regression', dpi=1000)
    plt.show()
    


#############################################################################################################
########################################  CONVERGENCE EN TEMPS  #############################################

def MMS_Conv_Temps(): 
    
    # Initialisation des vecteurs
    erreur_L1 = []
    erreur_L2 = []
    erreur_Linf = []
    c_ordre2= []
    r_ordre2= [] 
    
    # Initialisation des paramètres
    prm=param2()
    prm.tf=1e7
    prm.dr=0.01   # dr fixé à une valeur de maillage tres fin en espace
    dt_testee=[1e7,1e6,1e5,50000,25000,12500,6250,5000]

    rdom = np.linspace(0,prm.R,prm.n)

    # Calcul de de l'erreur
    for dt in dt_testee:
        
        prm.dt=dt
        prm.nt=int(prm.tf/dt)+1
        tdom = np.linspace(0,prm.tf,prm.nt)

        C_analy,z_source, f_source=MMS_analy(prm,rdom,tdom)
        C_num,tps,r=MMS_fct(prm,rdom,tdom) 
        
        c_ordre2.append(C_num[-1])
        r_ordre2.append(r)
        
        epsilon_h=C_num[-1]-C_analy[-1] # on extrait les derniers vecteurs de solution car c'est ceux qui correspondent à tf
        
        erreur_L1.append(np.sum(abs(epsilon_h))/len(epsilon_h))
        erreur_L2.append((np.sum(epsilon_h**2)/len(epsilon_h))**0.5)
        erreur_Linf.append(max(abs(epsilon_h)))

    C_analy,z_source, f_source=MMS_analy(prm,rdom,tdom)

        
    # Graphiques
    # Erreurs L1, L2 et L_inf sur echelle log/log
    plt.figure(1)
    plt.loglog(dt_testee,erreur_L1,'x-',label='$L_1$')  
    plt.loglog(dt_testee,erreur_L2,'x-',label='$L_2$')
    plt.loglog(dt_testee,erreur_Linf,'x-',label='$L_\infty$')
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.xlabel("$\Delta$t [$s$]")
    plt.ylabel("Erreur")
    plt.legend()
    plt.grid()
    plt.gca().invert_xaxis()
    plt.title(f"Convergence de l'erreur en fonction de $\Delta$t pour $\Delta$r={prm.dr}")
    # plt.savefig('Ordre2_Conv_dt', dpi=1000)
    
   
    plt.figure(2)
    for i in range(len(c_ordre2)):
        lab='$\Delta$t='+str(dt_testee[i])
        plt.plot(r_ordre2[i],c_ordre2[i],'-.',label=lab)
    plt.plot(rdom,C_analy[-1],'-.',label="Sol analytique (MMS)",linewidth=1.1)       
    plt.legend()
    plt.grid()
    plt.xlabel("r [m]")
    plt.ylabel("C [mol/$m^{3}$]")
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.title(f"Profil de concentration en fonction de r pour $\Delta$r={prm.dr}")
    # plt.savefig('Ordre2_C_dt', dpi=1000)

    
    plt.figure(3)
    a,b = np.polyfit(np.log(dt_testee[-4:]), np.log(erreur_L2[-4:]), 1)
    # Fonction de régression en termes de logarithmes
    fit_function_log = lambda x: a * x + b

    # Fonction de régression en termes originaux
    fit_function = lambda x: np.exp(fit_function_log(np.log(x)))

 
    plt.loglog(dt_testee,erreur_L2,'o',label="L2")
    plt.plot(dt_testee, fit_function(dt_testee), linestyle='--', color='r', label='Régression en loi de puissance')
    plt.tick_params(width=1.5, which='both', direction='in', top=True, right=True, length=5)
    plt.gca().invert_xaxis()
    plt.grid()
 

    # Ajouter des étiquettes et un titre au graphique
    plt.title('Convergence d\'ordre 2\n de l\'erreur $L_2$ en fonction de $Δt$',fontsize=12, y=1.02)  # Le paramètre y règle la position verticale du titre
    plt.xlabel('$Δt$ [s]', fontsize=10)
    plt.ylabel('Erreur $L_2$', fontsize=10)
    
    # Afficher l'équation de la régression en loi de puissance
    equation_text = f'$L_2 = {np.exp(b)} \\times $Δt$^{{{a:.4f}}}$'
    equation_text_obj = plt.text(0.05, 0.05, equation_text, fontsize=10, transform=plt.gca().transAxes, color='k')

    # Déplacer la zone de texte
    equation_text_obj.set_position((0.4, 0.7))
    plt.legend()    
    # plt.savefig('Ordre2_regression', dpi=1000)    
    plt.show()

print("Veuillez attendre la vérification du code est en cours.")    
# unittest.main(module=__name__)  
Cac()
# MMS_Conv_Espace()
# MMS_Conv_Temps()
