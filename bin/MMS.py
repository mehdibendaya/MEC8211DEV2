# =============================================================================
# =============================================================================
# ============ MEC 8211 - DEVOIR 2 - Verification de code - H24 ===============
# Redigé par:
# Mohammed Mahdi Sahbi Ben Daya
# Acile Sfeir
# =============================================================================
# =============================================================================

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

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
    nt=5
    # nt=int(tf/(10000*dt))+1 # hardcoding pour resultats rapide : fonctionne pour tf=1e9 avec nt=5,6,9 noeuds car donne des valeurs de temps qui sont des multiples de dt

prm=param2()

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
    #solution MMS
    # C_MMS=(r**2)*(R-r)*t
    C_MMS=(r**2)*(R-r)*(t/1e9)*sp.exp(-D*t)
    # create callable function for symbolic expression
    f_C_MMS = sp.lambdify([r,t], C_MMS, "numpy")

    # Appliquer l'opérateur sur la solution MMS
    source = sp.diff(C_MMS,t) -D*((1/r)*sp.diff(C_MMS,r)+sp.diff(sp.diff(C_MMS,r),r))+k*C_MMS
    # source = D*sp.exp(-D*t)*((k/D-1)*(r**2)*(R-r)-4*R+9*r) #formule obtenue a la main pour simplifier les r et eviter la division par 0
    # print("source=",source)
    # create callable function for symbolic expression
    f_source = sp.lambdify([r,t], source, "numpy")


    z_MMS=np.zeros((len(tdom),len(rdom)))
    z_source=np.zeros((len(tdom),len(rdom)))
    # Evaluate MMS function and source term on the grid
    for j in range(len(tdom)):
        t=tdom[j]
        for i in range(len(rdom)):
            z_MMS[j,i]    = f_C_MMS(rdom[i],t)
            z_source[j,i] = f_source(rdom[i], t)
        # print("ZMMS_i=",z_MMS)
    # print("ZMMS_ij=", z_MMS)


    return z_MMS, z_source, f_source




# =============================================================================
# =============================================================================
# =======================  DEVOIR2 : Schema d'ordre 2  ========================
# =============================================================================
# =============================================================================

# =============================================================================
# ============================= Regime transitoire ============================
# =============================================================================

def MMS_fct(prm,rdom,tdom):
    from time import time
    from time import time
    """ Fonction qui résout le systeme  pour le deuxième cas
    Entrées:
        - prm : classe contenant les donnees du probleme
        - tdom : vecteur de discretisation en temps

    Sorties :
        - c_globalc_tdt : Matrice (array) qui contient la solution numérique la plus a jour
        - tps : vecteur (liste) qui contient les différents temps de résolution
        - r : vecteur de discretisation en espace
    """            

    dr = prm.dr #Pas en espace
    dt = prm.dt #Pas en temps
    print ("dt=",dt)
    D_eff=prm.D_eff 
    n  = prm.n  #Nombre de noeuds
    r = rdom#Discrétisation en espace
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
        # b[1:n-1]= dt*(D_eff*np.exp(-D_eff*t)*((prm.k/D_eff-1)*(r[1:n-1]**3)*(prm.R-r[1:n-1])-9*prm.R*r[1:n-1]+16*r[1:n-1]**2)) +c_t[1:n-1] #avec r**3
        # b[1:n-1]= dt*(D_eff*np.exp(-D_eff*t)*((prm.k/D_eff-1)*(r[1:n-1]**2)*(prm.R-r[1:n-1])-4*prm.R+9*r[1:n-1])) +c_t[1:n-1] # avec r**2
        # b[1:n-1]= dt*((1+prm.k*t)*(r[1:n-1]**2)*(prm.R-r[1:n-1])-D_eff**t*(4*prm.R-9*r[1:n-1])) +c_t[1:n-1] #sans exp
        b[1:n-1]= dt*ggg(r[1:n-1],t) +c_t[1:n-1]
        b[0] = 0
        b[-1] = prm.Ce

        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)
        if t in tdom:
            c_global.append(list(c_tdt))
            print("C_GLOBAL=",len(c_global))
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



#Visualiser les fonctions sur un maillage
# Discretisation du domaine
rdom = np.linspace(0,prm.R,prm.n)
tdom = np.linspace(0,prm.tf,prm.nt)

z_MMS, z_source,g=MMS_analy(prm,rdom,tdom)
concentration, temps, rayon=MMS_fct(prm, rdom, tdom)

# Plot the results
for j in range(len(tdom)):
    # print("XDOM=",rdom)
    # print("Z=",z_MMS[j,:])
    plt.figure(1)
    plt.plot(rdom,z_MMS[j,:], label=f"t={tdom[j]} sec")
    plt.legend()
    plt.title('Fonction MMS')

    plt.figure(2)
    plt.plot(rdom,z_source[j,:], label=f"t={tdom[j]} sec")
    plt.legend()
    plt.title('Terme source')


    plt.figure(3)
    plt.plot(rdom,concentration[j], label=f"t={tdom[j]} sec")
    plt.legend()
    plt.title('Application MMS au code')

    plt.figure(4)
    plt.plot(rdom,concentration[j], label=f"t={tdom[j]} sec")
    plt.plot(rdom,z_MMS[j,:], '--', label=f"t={tdom[j]} sec")
    plt.legend()
    plt.title('Superposition MMS et solution numerique')

plt.show()

# def MMS():    
     
#     prm=param2()
#     r=np.linspace(0,prm.R,prm.n)
#     C_num1,tps1,r=MMS_fct(prm)
    
#     r,C_analy=MMS_analy(prm)
#     plt.figure()
#     plt.plot(r,C_num1,label='numerique')
#     plt.plot(r,C_analy,'--',label='MMS')
#     plt.legend()
#     plt.grid()
#     plt.show()