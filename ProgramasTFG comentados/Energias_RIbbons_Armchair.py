# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:47:43 2024

@author: Asus
"""

'''
CALCULO DE LAS BANDAS DE ENERGIA RIBBON ARMCHAIR
    Se hace igual que con las zigzag pero cambiando el hamiltoniano.
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import random as random

def eix(x):
    if np.isclose(x % np.pi, 0):
        return np.cos(x)
    elif np.isclose(x % (np.pi / 2), 0) and not np.isclose(x % np.pi, 0):
        return 1j*np.sin(x)
    else:
        return np.cos(x)+1j*np.sin(x)

def generar_color_aleatorio():
    # Generar componentes de color aleatorios en el rango [0, 255]
    color_rojo = random.randint(0, 255)
    color_verde = random.randint(0, 255)
    color_azul = random.randint(0, 255)
    
    # Convertir los componentes a formato hexadecimal y concatenarlos
    color_hexadecimal = '#{0:02x}{1:02x}{2:02x}'.format(color_rojo, color_verde, color_azul)
    
    return color_hexadecimal
eix=np.vectorize(eix)

a=1

My=8
t=0.7
h1=sp.diags((t*np.ones(3),0,t*np.ones(3)),[-1,0,1]).toarray()
h2=np.array([np.zeros(4),np.array([0,0,t,0]),np.zeros(4),np.zeros(4)])
h3=np.transpose(h2)
HM0y=sp.kron(np.eye(My-1),h1).toarray()+sp.kron(sp.diags(np.ones(My-2),-1).toarray(),h2).toarray()+sp.kron(sp.diags(np.ones(My-2),1).toarray(),h3).toarray()
H0=np.zeros((4*(My-1)+2,4*(My-1)+2))
H0[:4*(My-1),:4*(My-1)]=HM0y
hborde=np.zeros((2,4*(My-1)+2))
hborde[0][-1]=t
hborde[1][-2]=t
hborde[1][-4]=t
hborde2=np.zeros((4*(My-1)+2,2))
hborde2[-4][1]=t
H0[:,-2:]=hborde2
H0[-2:, :]=hborde

H1=np.zeros((4*(My-1)+2,4*(My-1)+2))
h5=np.array([np.zeros(4),np.zeros(4),np.zeros(4),np.array([t,0,0,0])])
H1[:4*(My-1),:4*(My-1)]=sp.kron(sp.diags((np.ones(My-2),np.ones(My-1)),(1,0)),h5).toarray()
H1[:,-2:][-3][0]=t

Hmenos1=np.transpose(H1)

ky=np.linspace(0,np.pi,200)

x=ky[0]
Hk=Hmenos1*eix(-x)+H0+eix(x)*H1

energies, states = np.linalg.eig(Hk)
energies=np.sort(energies.real)
EnergiesBig=np.sort(energies.real)

for i in range(1,len(ky)-1):
    x=ky[i]
    Hk=Hmenos1*eix(-x)+H0+eix(x)*H1
    energies, states = np.linalg.eig(Hk)
    energies=np.sort(energies.real)
    EnergiesBig=np.vstack((EnergiesBig,energies))

plt.figure()
plt.title(r'$M_a$= '+str(My),fontsize=18) 
plt.xlabel(r'$k$',fontsize=18)
plt.ylabel(r'$\varepsilon(k)$',fontsize=18)
x_ticks_locations = [np.pi/2,np.pi]
x_ticks_labels = [r'$\pi/2$',r'$\pi$']
x_ticks_labels.append(r'$\pi$')
plt.xticks(x_ticks_locations, x_ticks_labels,fontsize=12)
plt.yticks([])
plt.xlim(0,np.pi)
x_ticks_locations = [n*np.pi/2 for n in range(1, 5)]


for j in range(len(energies)-1):
    plt.plot(ky[:-1],EnergiesBig[:, j],color='red')
    #plt.plot(ky[:-1],-EnergiesBig[:, j],color=color)
    


    
   