# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:08:58 2024

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
'''
ESTUDIO DE LOS ESTADOS DE BORDE EN CADENAS SSH CLOSED CELL

En este programa:
    1. Calculamos las autoenergias de una cadena SSH con 
    condiciones de celda cerrada diagonalizando
    el hamiltoniano
    2. Determinamos cuales son los estados de borde
    3. "Perturbamos" el hamiltoniano alterando los valores
    de las integrales de salto t1 y t2 y las energias on-site
    con el propósito de verificar la permanencia de los estados
    de borde
        3.1. Generar un ruido en todas las integrales de salto
        de manera que para cada celda t1 y t2 dejan de ser valores
        uniformes
        3.2. Alterar las integrales de salto de los bordes, lo que 
        podría interpretarse como contaminacion del ambiente
        3.3. Alterar las energias on-site, lo que podría interpretarse
        como defectos en la red
'''
#Algunas funciones previas
def seq(t1, t2, n):
    sequence = []
    for i in range(n-1):
        if i % 2 == 0:
            sequence.append(t1)
        else:
            sequence.append(t2)
    return sequence
def f(t1,t2,k,ea=0,eb=0):
    return t1*((ea-eb)**2/(2*t1)**2+1+(t2/t1)**2+2*t2/t1*np.cos(k))**0.5
k=np.linspace(0,np.pi+0.03,1000)

#Celdas cerradas (terminacion A-B)
M=50 #Numero de celdas unidad
t1=1#integral de salto intracelda 
t2=1.5 #integral de sato entre celdas
dm=t2/t1 #parametro delta
n=2*M #numero total de atomos y por lo tanto, dimension del hamiltoniano
ejex=np.concatenate((np.arange(-n/2,0,1),np.arange(1,n/2+1,1)))
ejek = (ejex[:M] - ejex[0]) * (np.pi / (ejex[M-1] - ejex[0]))
#Introducimos los arrays que marcan las diagonales suerior e inferior
lu=seq(t1,t2,n) #diagonal superior
ld=seq(t1,t2,n) #diagonal inferior (tecnicamente no son iguales pero facilita
                #el calculo del hamiltoniano con sparse.diag
H=sp.diags((ld,0,lu),[-1,0,1]).toarray() #Hamiltoniano
def energias(energies,Ef=0,Ea=0,Eb=0):
    plt.plot(k,f(t1,t2,k,Ea,Eb)+Ef,color='black')
    plt.plot(k,-f(t1,t2,k,Ea,Eb)+Ef,color='black')
    plt.ylabel(r'$\varepsilon_k$',fontsize=18)
    plt.xlabel(r'$k$',fontsize=14)
    plt.xticks([0,np.pi/2,np.pi],['0',r'$\pi/2$',r'$\pi$'],fontsize=13)
    plt.yticks([Ef],[r'$E_F$'],fontsize=13)
    funcion2=f(t1,t2,np.pi,Ea,Eb)
    plt.fill_between(k, -funcion2+Ef, funcion2+Ef, color='green', alpha=0.2)
    plt.xlim(k[0],k[-1])
    plt.title(r'$\Delta =$'+str(round(dm,3))+ r'  $M =$' +str(M))
    plt.plot(ejek[:-1],energies[:M-1],'r.',label='Bulk')
    plt.plot(ejek[-1],energies[M-1],'b.',label='Borde',zorder=1000)
    plt.plot(ejek[-1],energies[M],'b.',zorder=1000)
    plt.plot(np.flip(ejek[:-1]),energies[M+1:],'r.')

#Ahora diagonalizamos la matriz
energies, states = np.linalg.eig(H)
idx=np.argsort(energies)
energies=energies[idx]
states=states[:,idx]
ejex=np.concatenate((np.arange(-n/2,0,1),np.arange(1,n/2+1,1)))
#y ahora hacemos un plot de las energías

plt.figure()
if dm>1:
    Mc=1/(dm-1)
    if M>Mc:
        print('Hay estados de borde')
plt.axhline(0,color='grey',linestyle='--',label='Nivel de Fermi',alpha=0.5)
energias(energies)
plt.title(r'Closed Cell con $\; \Delta$ = ' +str(round(dm,3))+
          r'$\;$ y $\; M$ =' + str(M))
plt.legend(loc='upper right')

#3.1. Ruido en todas las integrales de salto.
#Creamos dos amplitudes para los valores tipicos de t1 y t2
A1=1
A2=1

def seqruido(t1,t2,A1=A1,A2=A2,n=n,s=1):
    #Esta funcion nos crea un valor de ruido para cada
    #integral de salto siguiendo una distribucion de 
    #normas las amplitudes y moduladas por los valores
    #previos de t1 y t2
    sequence = []
    for i in range(n-1):
        if i % 2 == 0:
            sequence.append(np.random.normal(loc=t1*A1,scale=s))
        else:
            sequence.append(np.random.normal(loc=t2*A2,scale=s))
    return sequence

plt.figure()
sigma=np.array([0.1,0.25,0.5,1])
for i in range(len(sigma)):
    plt.subplot(2,2,i+1)
    ruido= seqruido(t1,t2,A1,A2,s=sigma[i])
    dH=sp.diags((ruido,0,ruido),[-1,0,1]).toarray()
    
    Hperb1=dH
    
    #Ahora diagonalizamos la matriz
    energiesP1, statesP1 = np.linalg.eig(Hperb1)
    idxP1=np.argsort(energiesP1)
    energiesP1=energiesP1[idxP1]
    statesP1=statesP1[:,idxP1]
    
    
    
    energias(energiesP1)
    plt.axhline(0,color='grey',linestyle='--',label='Nivel de Fermi',alpha=0.5)
    plt.title(r'$\sigma=$ '+str(sigma[i]))


#3.2. Cambio de las integrales en los bordes
#Para modificar las integrales de los bordes tenemos que simplemente
#cambiar el primer y el segundo valor del array lu 
tl=t1*0.05
tr=t1*0.05

lp2=np.concatenate((np.array([tl]),lu[1:-1],np.array([tr])))
Hperb2=sp.diags((lp2,0,lp2),[-1,0,1]).toarray()

#Ahora diagonalizamos la matriz
energiesP2, statesP2 = np.linalg.eig(Hperb2)
idxP2=np.argsort(energiesP2)
energiesP2=energiesP2[idxP2]
statesP2=statesP2[:,idxP2]

plt.figure()

energias(energiesP2)
plt.axhline(0,color='grey',linestyle='--',label='Nivel de Fermi',alpha=0.5)
plt.title(r'Closed Cell perturbado2 con $\; \Delta$ = ' +str(round(dm,3))+
          r'$\;$ y $\; M$ =' + str(M))
plt.text(ejex[-int(M/2)],energiesP2[0],'tl = ' +str(round(tl,2))+
              '\ntr = '+str(round(tr,2)))
plt.legend(loc='upper left')


    
    

#3.3. Cambiar las energias onsite
#Aqui tenemos varias opciones:
    #a)Podemos cambiar las energias por separado de los atomos B y A
    #b)Podemos hacer un ruido en general para ambos atomos
    #c)Podemos cambiar las de los vertices
#Programamos la primera y las otras dos son un caso particular de esta

#a)
Ea=0 #Energia on site atomos A
Eb=0 #Energia on site atomos B
#♥d=seq(Ea,Eb,n+1) #Diagonal

AE1=1
AE2=1
d=seqruido(Ea,Eb,AE1,AE2,n+1,s=8)
'''
#c)
E1=Ea*2
E2M=Eb*2
d[0]=E1
d[-1]=E2M
'''

Hdia=sp.diags(d,0).toarray()
Hperb3=H+Hdia

#Ahora diagonalizamos la matriz
energiesP3, statesP3 = np.linalg.eig(Hperb3)
idxP3=np.argsort(energiesP3)
energiesP3=energiesP3[idxP3]
statesP3=statesP3[:,idxP3]


plt.figure()
energias(energiesP3,Ef=(Ea+Eb)/2,Ea=Ea,Eb=Eb)
plt.axhline((Ea+Eb)/2,color='grey',linestyle='--',
            label='Nivel de Fermi',alpha=0.5)
plt.title(r'Closed Cell perturbado3 con $\; \Delta$ = ' 
          +str(round(dm,3))+ r'$\;$ y $\; M$ =' + str(M))
plt.ylabel('E')
plt.xticks([])
#plt.text(ejex[-int(M/2)],energiesP2[0],'tl = ' +str(round(tl,2))
#               +'\ntr = '+str(round(tr,2)))
plt.legend(loc='upper left')

sigma=np.array([0.25,1,5,10])
Eedge=np.zeros_like(sigma)
fig1=plt.figure(15)
fig2=plt.figure(16)
for i in range(0,len(sigma)):
    plt.figure(15)
    plt.subplot(2,2,i+1)
    d4=d=seqruido(Ea,Eb,AE1,AE2,n+1,s=sigma[i])
    H4=H+sp.diags(d4,0).toarray()
    energiesP4,statesP4=np.linalg.eig(H4)
    idxP4=np.argsort(energiesP4)
    energiesP4=energiesP4[idxP4]
    statesP4=statesP4[:,idxP4]

    Eedge[i]=energiesP4[M]
    energias(energiesP4)
    plt.axhline(0,color='grey',linestyle='--',label='Nivel de Fermi',alpha=0.5)
    plt.title(r'$\sigma$ = ' +str(round(sigma[i],3)),fontsize=18)
    plt.figure(16)
    plt.subplot(2,2,i+1)
    plt.title(r'$\sigma$ = ' +str(round(sigma[i],3)),fontsize=18)
    plt.bar(ejex,np.abs(statesP4[:,M-1])**2)
    plt.xticks([ejex[0],ejex[-1]],[r'$A0$',r'$BM$'],fontsize=12)
    
    


    
    