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
import numpy as np

# =============================================================================
# =============================================================================
# =====================  Fonctions analytiques     ============================
# =============================================================================
# =============================================================================

def C_analytique(prm):
        
        """ Fonction qui calcule la solution analytique
        Entrée : 
        - prm : classe contenant les donnees du probleme
        Sortie :
        - y : vecteur contenant la valeur numérique de la fonction 
        - r : vecteur contenant le domaine descretise 
        """
        r = np.linspace(0, prm.R, prm.n)
        y=(0.25*(prm.S/prm.D_eff)*(prm.R*prm.R)*(((r*r)/(prm.R*prm.R))-1))+prm.Ce
          
        return r,y
 
def C_analytique2(prm,n):
        
        """ Fonction qui calcule la solution analytique
        Entrée : 
        - prm : classe contenant les donnees du probleme
        - n : nombre de noeuds desires
        Sortie :
        - y : vecteur contenant la valeur numérique de la fonction 
        - r : vecteur contenant le domaine descretise 
        """
        r = np.linspace(0, prm.R, n)
        y=(0.25*(prm.S/prm.D_eff)*(prm.R*prm.R)*(((r*r)/(prm.R*prm.R))-1))+prm.Ce
          
        return y,r
    
def droite(a,b,x):
     """ Fonction qui calcule l'équation d'une droite
        Entrée :
        - a : un scalaire, la pente
        - b : un scalaire, la valeur à l'origine
        - x : vecteur des abscisses
        Sortie :
        - y : vecteur de sortie  
        """
     y=np.empty(len(x))
     for i in range(len(x)):
        y[i]=a*x[i]+b
     
     
     return y    
    


# =============================================================================
# =============================================================================
# ========================== Schema d'ordre 2  ================================
# =============================================================================
# =============================================================================  
 
# ============================================================================= 
# ===================== Comparaison code à code ===============================
# ============================================================================= 

def PbF(prm):
    from time import time
    """ Fonction qui résout le systeme  pour le deuxième cas
    Entrées:
        - prm : vecteur contenant la position 

    Sorties :
        - c_tdt : Matrice (array) qui contient la solution numérique la plus a jour
        - tps : vecteur (liste) qui contient les différents temps de résolution"""        

    dr = prm.dr #Pas en espace
    dt = prm.dt #Pas en temps
    D_eff=prm.D_eff 
    n  = prm.n  #Nombre de noeuds
    r = np.linspace(0, prm.R, n) #Discrétisation en espace
    A = np.zeros([prm.n, prm.n]) #Matrice A
    b = np.zeros(prm.n) #Vecteur b
    t=0         #Temps intial
    tps=[0]
    err_t_tdt=10 #Initialisation de l'erreur
    # Initialisation de c_t
    c_t=np.ones(n)
    c_t[:-1] = [0 for i in range(n-1)]
    c_t[-1]=prm.Ce

    # Remplissage du centre de la matrice A et du vecteur b    
    dr_inv=0.5/dr
    dt_D_eff=dt*D_eff
    dr2_inv=1/dr**2
    
    for i in range(1, prm.n-1):
        A[i, i+1] = -dt_D_eff*(dr_inv/r[i]+dr2_inv)
        A[i, i] = 1+dt_D_eff*(2*dr2_inv)
        A[i, i-1] = -dt*D_eff*(dr2_inv-dr_inv/r[i])
    #Condition de Dirichlet
    A[-1, -1] = 1 
    #Condition de Neumann
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    
    while t<prm.tf:

        b[1:n-1]=-dt*prm.k*c_t[1:n-1]+c_t[1:n-1]
        b[0] = 0
        b[-1] = prm.Ce
        
        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)

        c_t[:]=c_tdt[:]
        
        t+=prm.dt
        tps.append(t)
        i+=1
        if i%10000==0:
            duration = time() - start
            print(duration,err_t_tdt)
            start = time()

    return c_tdt,tps

