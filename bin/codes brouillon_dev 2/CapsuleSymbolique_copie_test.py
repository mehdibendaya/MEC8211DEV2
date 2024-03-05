# -*- coding: utf-8 -*-
"""
CapsuleSymbolique.py : Petite démonstration de calcul symbolique en Python

Created on 23 juillet 2020

@author: Jean-Yves Trépanier
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

r,t=sp.symbols('r t')

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

prm=param()

#solution MMS
C_MMS=r*(1-r)*sp.exp(-t*prm.D_eff)
# create callable function for symbolic expression
f_C_MMS = sp.lambdify([r,t], C_MMS, "numpy")

# Appliquer l'opérateur sur la solution MMS
source = prm.D_eff*((1/r)*sp.diff(C_MMS,r)+sp.diff(sp.diff(C_MMS,r),r))-prm.k*C_MMS
# create callable function for symbolic expression
f_source = sp.lambdify([r,t], source, "numpy")

#Visualiser les fonctions sur un maillage  
# taille du domaine
xmin = 0
xmax = 5
ymin = 0
ymax = 2
# Set up a regular grid of interpolation points    
xdom = np.linspace(xmin,xmax,50)
ydom = np.linspace(ymin,ymax,50)
xi, yi = np.meshgrid(xdom, ydom)

# Evaluate MMS function and source term on the grid
z_MMS    = f_C_MMS(xi,yi)
z_source = f_source(xi, yi)

# Plot the results
plt.contourf(xi,yi,z_MMS)
plt.colorbar()
plt.title('Fonction MMS')
plt.show()

plt.contourf(xi,yi,z_source)
plt.colorbar()
plt.title('Terme source')
plt.show()

