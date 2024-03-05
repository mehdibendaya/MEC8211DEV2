# =============================================================================
# =============================================================================
# ============ MEC 8211 - DEVOIR 1 - Verification de code - H24 ===============
# Redigé par:
# Mohammed Mahdi Sahbi Ben Daya
# Acile Sfeir
# =============================================================================
# =============================================================================

# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp

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

def CAC_fct(prm):
    from time import time
    """ Fonction qui résout le systeme  pour le deuxième cas
    Entrées:
        - prm : classe contenant les donnees du probleme 

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
        A[i, i] = 1+dt_D_eff*(2*dr2_inv)+prm.k*dt
        A[i, i-1] = -dt*D_eff*(dr2_inv-dr_inv/r[i])
    #Condition de Dirichlet
    A[-1, -1] = 1 
    #Condition de Neumann
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    
    start=time()
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



# ============================================================================= 
# ===========================      MMS 1   ====================================
# ============================================================================= 
def MMS_analy(prm,rdom,tdom):
    """ Fonction qui calcule la solution MMS "analytiquement"
    Entrée : 
        - prm : classe contenant les donnees du probleme
        - rdom : vecteur de discretisation en espace
        - tdom : vecteur de discretisation en temps

    Sortie :
        - z_MMS : vecteur contenant la valeur numérique de la fonction MMS en chaque noeud
        - z_source : vecteur contenant la valeur numérique du terme source en chaque noeud
        - f_source :  fonction "callable" du terme source

    """
    r,t=sp.symbols('r t')

    R=prm.R
    D=prm.D_eff
    k=prm.k

    # Expression symbolique de la fonction MMS choisie
    C_MMS=(r**2)*(R-r)*(t/1e9)*sp.exp(-D*t)
    # Create callable function for symbolic expression
    f_C_MMS = sp.lambdify([r,t], C_MMS, "numpy")

    # Appliquer l'opérateur différentiel sur la fonction MMS (calcul symbolique)
    source = sp.diff(C_MMS,t) -D*((1/r)*sp.diff(C_MMS,r)+sp.diff(sp.diff(C_MMS,r),r))+k*C_MMS
    # print("source=",source)
    # Create callable function for symbolic expression
    f_source = sp.lambdify([r,t], source, "numpy")

    z_MMS=np.zeros((len(tdom),len(rdom)))
    z_source=np.zeros((len(tdom),len(rdom)))
    
    # Calcul de la fonction MMS et du terme source en chaque noeud de discretisation
    for j in range(len(tdom)):
        t=tdom[j]
        for i in range(len(rdom)):
            z_MMS[j,i]    = f_C_MMS(rdom[i],t)
            z_source[j,i] = f_source(rdom[i], t)
        # print("ZMMS_i=",z_MMS)
    # print("ZMMS_ij=", z_MMS)

    return z_MMS, z_source, f_source

    
def MMS_fct(prm,rdom, tdom):
    from time import time
    """ Fonction qui résout le systeme  pour le deuxième cas
    Entrées:
        - prm : classe contenant les donnees du probleme
        - tdom : vecteur de discretisation en temps

    Sorties :
        - c_global : Matrice (array) qui contient les solutions numériques c_tdt à chaque t de tdom
        - tps : vecteur (liste) qui contient les différents temps de résolution
        - r : vecteur de discretisation en espace
    """     
      
       

    dr = prm.dr #Pas en espace
    dt = prm.dt #Pas en temps
    D_eff=prm.D_eff 
    n  = prm.n  #Nombre de noeuds
    r = rdom #Discrétisation en espace
    A = np.zeros([prm.n, prm.n]) #Matrice A
    b = np.zeros(prm.n) #Vecteur b
    t=0         #Temps initial
    tps=[0]
    err_t_tdt=10 #Initialisation de l'erreur
    # Initialisation de c_t
    c_t=np.ones(n)
    c_global=[]
    c_t[:-1] = [0 for i in range(n-1)]
    c_t[-1]=prm.Ce

    # Remplissage du centre de la matrice A et du vecteur b    
    dr_inv=0.5/dr
    dt_D_eff=dt*D_eff
    dr2_inv=1/dr**2
    
    for i in range(1, prm.n-1):
        A[i, i+1] = -dt_D_eff*(dr_inv/r[i]+dr2_inv)
        A[i, i] = 1+dt_D_eff*(2*dr2_inv)+prm.k*dt
        A[i, i-1] = -dt*D_eff*(dr2_inv-dr_inv/r[i])
    #Condition de Dirichlet
    A[-1, -1] = 1 
    #Condition de Neumann
    A[0, 0] = -3
    A[0, 1] = 4
    A[0, 2] = -1
    g,gg,ggg=MMS_analy(prm,r,tdom)
    j=0
    start=time()
    while t<=prm.tf:
        b[1:n-1]= dt*ggg(r[1:n-1],t) +c_t[1:n-1]
        b[0] = 0
        b[-1] = prm.Ce

        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)
        if t in tdom:
            c_global.append(list(c_tdt))
            # print("C_GLOBAL=",c_global)
        c_t[:]=c_tdt[:]

        t+=prm.dt
        tps.append(t)
        j+=1
        if j%100000==0:
            duration = time() - start
            print(duration,t)
            start = time()

    return c_global,tps, r

