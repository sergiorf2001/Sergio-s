# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:47:43 2024

@author: Asus
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import random as random

'''
CALCULO DE LAS BANDAS DE ENERGIAS DE RIBBONS INFINITOS (ZIGZAG)
    Seguimos los pasos expuestos en el TFG.
'''

#Parametros Inciales
a=1   #constante de red
t=2.7 #integral de salto

#Definimos la exponencial imaginaria.
def eix(x):
    if np.isclose(x % np.pi, 0):
        return np.cos(x)
    elif np.isclose(x % (np.pi / 2), 0) and not np.isclose(x % np.pi, 0):
        return 1j*np.sin(x)
    else:
        return np.cos(x)+1j*np.sin(x)


eix=np.vectorize(eix) # y la vectorizamos

#Por convenencia vamos a llamar M=Mx
Mx=9 #4*Mx atomos en la celda unidad

#Defino las matrices que forman parte del hamiltoniano de la supercelda
H0=sp.diags((t*np.ones(4*Mx-1),0,t*np.ones(4*Mx-1)),[-1,0,1]).toarray()

h1=np.array([np.array([0,t,0,0]),np.zeros(4),np.zeros(4),np.array([0,0,t,0])])
H1=sp.kron(np.eye(Mx),h1).toarray()
Hmenos1=np.transpose(H1)

#Pasamos al calculo de las bandas de energia. Lo que hacemos es resolver
#HkC=ekC para un mallado en k lo suficientemente grande para emular que
#es una variable continua. Por trasladar el codigo mas facilmente a el
#caso de las cintas, llamamos ky=k
ky=np.linspace(0,np.pi,200)#
x=ky[0]
Hk=Hmenos1*eix(-x)+H0+eix(x)*H1 #Hamiltoniano de la super celda
 #numero de cedas en sentido vertical (es solo para visualizarlo mejor)
#pues en todas las "filas" se vera lo mismo


#Diagonalizamos el Hamiltoniano para un prmier valor de k
energies, states = np.linalg.eig(Hk)
idx=np.argsort(energies.real)
energies=energies[idx].real #nos quedamos con la parte real de las energias
                            #Se puede comprobar que la imaginaria es infima
states=states[:,idx]
EnergiesBig=energies
#Y pasamos a hacerlo para el resto
for i in range(1,len(ky)):
    x=ky[i]
    Hk=Hmenos1*eix(-x)+H0+eix(x)*H1
    energies, states = np.linalg.eig(Hk)
    idx=np.argsort(energies.real)
    energies=energies[idx].real
    states=states[:,idx]
    EnergiesBig=np.vstack((EnergiesBig,energies))

plt.figure()

#Finalmente, creamos la figura
Myrepr=3
N=2*Myrepr-1 #numero de lineas dimeras en el eje y

#ky permitidos
ky2=np.zeros(Myrepr)
for i in range(0,Myrepr):
    ky2[i]=2*(i+1)*np.pi/(N+1)

plt.title(r'$M$= '+str(Mx),fontsize=18) 
plt.xlabel(r'$k$',fontsize=18)
plt.ylabel(r'$\varepsilon(k)$',fontsize=18)
x_ticks_locations = ky2
x_ticks_labels = [f'{n}$\pi/{int(Myrepr)}$' for n in range(1, Myrepr)]
x_ticks_labels.append(r'$\pi$')
plt.xticks(x_ticks_locations, x_ticks_labels,fontsize=12)
plt.yticks([])
plt.xlim(0,np.pi)

idedge=[]
for j in range(len(energies)):
    plt.plot(ky[:],EnergiesBig[:, j],color='red')
    if abs(EnergiesBig[:, j][-1])<=0.5:
        idedge.append(j)


'''
#ESTAS LINEAS SON PARA REPRESENTAR LAS DISTRIBUCIONES DE PROBABILIDAD
#Ahora con la misma tactica podemos pintar  la distribucion espacial de la funcion
#de onda Nos centramos en las dos bandas centrales  vamos a escoger cuatro valores
#para K: 1.5, 2*np.pi/3, 3*npi/4 y np.pi



#Primero debemos de dibujar la red directa de los rectángulos del grafeno 
#utilizando los parámetros anteriores

My=3

c=1
a=c/(2*np.sin(60/180*np.pi))
d2x=np.sqrt(3)/2*c
d2y=c/2
d1x=0
d1y=0

#En las siguientes lineas calculo las coordenadas de cada uno
#de los atomos de carbono del ribbon, separando los bordes
#y los atomos que no interaccionan
A1x = np.sqrt(3)*np.arange(Mx+1)
A2x=A1x+d2x
A1y = np.arange(My+1)
A2y=A1y+d2y
A1=np.meshgrid(A1x[1:],A1y[1:])

A2=np.meshgrid(A2x,A2y)
A20x=np.delete(A2[0][0],0) 
A2Mx=np.delete(A2[0][-1],0)
A2bordesx=np.concatenate((A20x,A2Mx,A2[0][1:-1,0]),axis=0)
A20y=np.delete(A2[1][0],0)
A2My=np.delete(A2[1][-1],0)

A2bordesy=np.concatenate((A20y,A2My,A2[1][1:-1,0]),axis=0)
A1lista=np.vstack([A1[0].ravel(), A1[1].ravel()]).T
A2lista=np.vstack([A2[0][1:-1,1:].ravel(),A2[1][1:-1,1:].ravel()]).T

B1x = A1x+a
B2x=B1x+d2x
B1y = A1y
B2y=B1y+d2y
B1=np.meshgrid(B1x[1:],B1y[1:])
B2=np.meshgrid(B2x,B2y)
B20x=np.delete(B2[0][0],-1)
B2Mx=np.delete(B2[0][-1],-1)
B2bordesx=np.concatenate((B20x,B2Mx,B2[0][1:-1,-1]),axis=0)
B20y=np.delete(B2[1][0],-1)
B2My=np.delete(B2[1][-1],-1)
B2bordesy=np.concatenate((B20y,B2My,B2[1][1:-1,-1]),axis=0)
Bordesx=np.concatenate((A2bordesx,B2bordesx),axis=0)
Bordesy=np.concatenate((A2bordesy,B2bordesy),axis=0)
Bordes=np.column_stack((Bordesx,Bordesy))
B1lista=np.vstack([B1[0].ravel(), B1[1].ravel()]).T
B2lista=np.vstack([B2[0][1:-1,:-1].ravel(),B2[1][1:-1,:-1].ravel()]).T


k=np.array([2*np.pi/3,3*np.pi/4,9*np.pi/10,np.pi])
augbig=[0.5*1e3,0.5*1e3,0.5*1e3,1e2]
kstr = [r'$2\pi/3$',r'$3\pi/4$',r'$9\pi/10$',r'$\pi$']
B2x=B2x[:-1]
A1x=A1x[1:]
A2x=A2x[1:]
B1x=B1x[1:]
points=np.concatenate((A1lista,A2lista,B1lista,B2lista))



plt.figure()
for i in range(len(k)):
    aug=augbig[i]
    plt.subplot(2,2,i+1)
    plt.axis('equal')      


    plt.title(r'$k$ = ' + kstr[i],fontsize=18)
    plt.xticks([])
    plt.yticks([])
    x=k[i]
    Hk=Hmenos1*eix(-x)+H0+eix(x)*H1
    energies, states = np.linalg.eig(Hk)
    idx=np.argsort(energies.real)
    energies=energies[idx].real
    states=states[:,idx]
    
    
    
    #♣points=np.concatenate((A1lista,A2lista,B1lista,B2lista))
    
    for i in range(len(points)):
        #Parq pintar las lineas simplemente fijarse
        #en cuales son los primero vecinos de cada atomo
        point1=points[i]
        points2=[]
        for j in range(i,len(points)):
            if  abs(np.linalg.norm(point1-points[j])-a)<1e-3:
                points2.append(points[j])
        for p in points2:
            plt.plot([p[0],point1[0]],[p[1],point1[1]],'-',color='black')
    
   #plt.scatter(A1[0],A1[1],color='b')
    #plt.scatter(B1[0],B1[1],color='b')
    #plt.scatter(A2[0][1:-1,1:],A2[1][1:-1,1:],color='b')
    #plt.scatter(A20x,A20y,color='orange')
    #plt.scatter(A2Mx,A2My,color='orange')
    #plt.scatter(B2[0][1:-1,:-1],B2[1][1:-1,:-1],color='b')
    #plt.scatter(B20x,B20y,color='orange')
    #plt.scatter(B2Mx,B2My,color='orange')
    #plt.scatter(Bordesx,Bordesy,color='orange')
    
    estado=np.absolute(states[:,idedge[-1]])**2
    

    
    for i in range(1,My):
        B1y=B1[1][i][1:]
        B2y=B2[1][i][1:-1]
        A2y=A2[1][i][1:-1]
        A1y=A1[1][i][1:]
    
        ib2=0
        ia1=1
        ia2=3
        ib1=2
        for nx in range(len(B2x)):  
            plt.scatter(B2x[nx],B2y[0],s=aug*estado[ib2],color='red',alpha=0.7)
            ib2+=4
        for nx in range(len(A1x)):  
            plt.scatter(A1x[nx],A1y[0],s=aug*estado[ia1],color='red',alpha=0.7)
            ia1+=4
        for nx in range(len(B1x)):  
            plt.scatter(B1x[nx],B1y[0],s=aug*estado[ib1],color='red',alpha=0.7)
            ib1+=4
        for nx in range(len(A2x)):  
            plt.scatter(A2x[nx],A2y[0],s=aug*estado[ia2],color='red',alpha=0.7)
            ia2+=4
'''
              
        
        



    
    
        
    
   