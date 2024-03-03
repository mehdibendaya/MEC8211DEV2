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

prm=param2()

def MMS_diffusion(prm,rdom,tdom):

    r,t=sp.symbols('r t')

    R=prm.R
    D=prm.D_eff
    k=prm.k
    #solution MMS
    C_MMS=(r**2)*(R-r)*sp.exp(-t*D)
    # create callable function for symbolic expression
    f_C_MMS = sp.lambdify([r,t], C_MMS, "numpy")

    # Appliquer l'opérateur sur la solution MMS
    # source = sp.diff(C_MMS,t) -D*((1/r)*sp.diff(C_MMS,r)+sp.diff(sp.diff(C_MMS,r),r))+k*C_MMS
    source = D*sp.exp(-D*t)*((k/D-1)*(r**2)*(R-r)-4*R+9*r) #formule obtenue a la main pour simplifier les r et eviter la division par 0
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


    return z_MMS, z_source




#Visualiser les fonctions sur un maillage
# taille du domaine
rmin = 0
rmax = prm.R
tmin = 0
tmax = prm.tf
# Discretisation du domaine
rdom = np.linspace(rmin,rmax,prm.n)
tdom = np.linspace(tmin,tmax,5)

z_MMS, z_source=MMS_diffusion(prm,rdom,tdom)

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


plt.show()



# =============================================================================
# =============================================================================
# =======================  DEVOIR2 : Schema d'ordre 2  ========================
# =============================================================================
# =============================================================================

# =============================================================================
# ============================= Regime transitoire ============================
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
    print("n=",n)
    r = np.linspace(0, prm.R, n) #Discrétisation en espace
    print("r=",r)
    A = np.zeros([prm.n, prm.n]) #Matrice A
    b = np.zeros(prm.n) #Vecteur b
    t=0         #Temps initial
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


    #Calcul du premier pas de temps pour eviter une division par 0

    # b[1:n-1]=-dt*prm.S+c_t[1:n-1]
    t0=0
    b[1:n-1]=D_eff*np.exp(-D_eff*t)*((prm.k/D_eff-1)*(r[0]**2)*(prm.R-r[0])-4*prm.R+9*r[0]) +c_t[1:n-1]
    b[0] = 0   #Condition de Neumann
    b[-1] = prm.Ce #Condition de Dirichlet
    c_tdt = np.linalg.solve(A, b) #Resolution du systeme matriciel
    c_t[:]=c_tdt[:]
    t+=prm.dt #incrementation en temps
    tps.append(t)
    j=0
    start = time()

    while t<prm.tf:
        terme_source=np.zeros(len(r))
        for h in range(len(r)):
            terme_source[h]=D_eff*np.exp(-D_eff*t)*((prm.k/D_eff-1)*(r[h]**2)*(prm.R-r[h])-4*prm.R+9*r[h])
        b[1:n-1]= terme_source[1:n-1] +c_t[1:n-1]
        b[0] = 0
        b[-1] = prm.Ce

        # Résolution du système matriciel
        c_tdt = np.linalg.solve(A, b)
        #Calcul de l'erreur
        err_t_tdt=np.linalg.norm(c_t-c_tdt)
        c_t[:]=c_tdt[:]

        t+=prm.dt
        tps.append(t)
        j+=1
        if j%10000==0:
            duration = time() - start
            print(duration,err_t_tdt)
            start = time()

    return c_tdt,tps, r


concentration, temps, rayon=PbF(prm)
# Plot the results
for t in temps:
    plt.figure(3)
    plt.plot(rayon,concentration, label=f"t={t} sec")
    plt.legend()
    plt.title('Fonction MMS')


plt.show()