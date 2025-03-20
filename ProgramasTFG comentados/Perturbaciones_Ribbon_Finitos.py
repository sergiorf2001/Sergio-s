# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:04:19 2024

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

#Parametros inciales
c=1
a=c/(2*np.sin(60/180*np.pi))
t=1 

#hagamos primero el de los bordes zigzag
'''
PERTURBACIONES EN LOS RIBBONS FINITOS:
    En este programa construimos el Hamiltoniano de un ribbon de gradfeno como
    se indica en el texto. A continuacion introducimos perturbaciones en las 
    integrales de salto de los bordes verticales para comprobar la robustez de 
    los estados de borde. Posteriormente lo hacemos introducimos ruido en las
    energias on-site. AVISO: tarda un poco en ejecutarse (matrices muy grandes)
'''

#Variables de entrada (tamaño del ribbon)
Mx=20#4*Mx atomos en la celda unidad
My=9

#Como sabemos que vamos a necesitar deshacernos de las soluciones que no sean 
#nulas en las posiciones de los atomos falsos, introducimos una funcion que
#nos va a servir para filtrar el esspectro. Si se cumple alguna de las 
#condiciones que mencionamos en el texto, sustituimos la energia correspondiente
#por 1e6 y el ayutovector por una array de unos. Mas tarde en el codigo nos 
#aplicaremos esta funcion y descartaremos estos elementos.
def filtrado(energies,states):
    for j in range(len(energies)):
        for k in range(1,Mx+1):
            if states[:,j][4*Mx*(My-1)+4*k-1]!=0:
                states[:,j]=np.ones_like(states[:,j])
                energies[j]=1e6
            else: 
                continue
        for m in range(0,Mx):
            if states[:,j][4*Mx*(My-1)+4*m]!=0:
                states[:,j]=np.ones_like(states[:,j])
                energies[j]=1e6
            else:
                continue
    
  
    
#Construimos el Hamiltoniano como se indica en el texto
H0=sp.diags((t*np.ones(4*Mx-1),0,t*np.ones(4*Mx-1)),[-1,0,1]).toarray()

h1=np.array([np.array([0,t,0,0]),np.zeros(4),np.zeros(4),np.array([0,0,t,0])])
H1=sp.kron(np.eye(Mx),h1).toarray()
Hmenos1=np.transpose(H1)

H=np.kron(np.eye(My),H0)+np.kron(sp.diags(np.ones(My-1),-1).toarray(),
          Hmenos1)+np.kron(sp.diags(np.ones(My-1),1).toarray(),H1)

h0gorro=np.array([np.zeros(4),np.array([0,0,t,0]),
                  np.array([0,t,0,0]),np.zeros(4)])
H0gorro=sp.kron(np.eye(Mx),h0gorro).toarray()

H[-(4*Mx):,-(4*Mx):]=H0gorro


#Y a continuacion diagonalizamos el hamiltoniano entero
energies, states = np.linalg.eig(H)

filtrado(energies,states) #aplicamos la funcion de la que hablabamos antes
states=states[:,energies<1e6] #filtramos los estados
energies=energies[energies<1e6] #iltramos las energias
#Y las ordenamos y las pintamos
idx=np.argsort(energies.real)
energies=energies[idx].real
states=states[:,idx]
Edge=states[:,int(len(energies)/2)] #ESTE ESTADO LO ESCOJO MANUALMENTE
ejex=np.arange(0,len(energies)/2)

def bandas(energies=energies,s=12):
    plt.plot(ejex,energies[:int(len(energies)/2)],'b.',markersize=s)
    plt.plot(np.flip(ejex),energies[int(len(energies)/2):],'b.',markersize=s)
    plt.axhline(energies[int(len(energies)/2)-4])
    plt.axhline(energies[int(len(energies)/2)+3])
    plt.fill_between([ejex[0],ejex[-1]+50],energies[int(len(energies)/2)-4],
                           energies[int(len(energies)/2)+3],
                           color='green', alpha=0.2)
    plt.xlim(ejex[0],ejex[-1]+10)
    plt.xticks([])
    plt.yticks([0],[r'$E_f$'],fontsize=18)
    plt.ylabel(r'$\varepsilon$',fontsize=20)
    plt.axhline(0,color='grey', linestyle='--')
plt.figure()
plt.subplot(1,2,1)
bandas()
plt.subplot(1,2,2)
bandas()
plt.xlim(ejex[-6],ejex[-1]+1)
plt.ylim(energies[int(len(energies)/2)-5],energies[int(len(energies)/2)+4])

#Las siguientes lineas son solo para poder pintar el ribbon finito y la distri
#buccion de probabilidad. Son las mismas que en los otros codigos. EL codigo
#continua en la linea 205
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
A2lista=np.vstack([A2[0][1:-1,1:].ravel(),
                   A2[1][1:-1,1:].ravel()]).T

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
B2lista=np.vstack([B2[0][1:-1,:-1].ravel(),
                   B2[1][1:-1,:-1].ravel()]).T

B2x=B2x[:-1]
A1x=A1x[1:]
A2x=A2x[1:]
B1x=B1x[1:]

    
    
points=np.concatenate((A1lista,A2lista,B1lista,B2lista))

def pintar(aug,es=Edge):   #funcion para pintar las distribucion de probabilidad
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
    
    kmas=4*Mx
    kmenos=-4*Mx
    
    for i in range(0,My):
            
        kmenos+=4*Mx
        estado=np.abs(es[kmenos:kmas])
        B1y=B1[1][i][1:]
        B2y=B2[1][i+1][1:-1]
        A2y=A2[1][i+1][1:-1]
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
        kmas+=4*Mx
plt.figure()
pintar(aug=5e2,es=Edge)

#Ahora podemos perturbar las cadenas SSH
#nos centramos en modificar las integrales de salto de los bordes, para lo que
#solo tenemos que cambiar los elementos relvantes en H0 y H1 (Hmenos1)         

plt.figure()
#Lo vamos a hacer para cuatro valores distintos de la desviacion estandar.
#que recogemos en el siguiente array.
sigma=np.array([0.25,0.5,0.75,1])

#A continuacion construimos el hamiltoniano perturbado. Es el mismo que antes
#pero estamos cambiando las integrles de salto de los bordes laterales, de ma_
#nera que cada una de ellas es una muestra de una distribuccion normal de norma
#t y desvaicion estandar sigma. Lo hacemos a traves de un bucle sobre cada una
#de las sigmas que proponemos en el array.
Hp=np.zeros((4*Mx*My,4*Mx*My)) 
for i in range(len(sigma)):
    for j in range(0,My-1):
        #Construccion del Hamiltoniano perturbado.
        Mprod=np.zeros((My,My))
        Mprodcha=1*Mprod
        Mprodizq=1*Mprod
        Mprod[j,j]=1
        Mprodcha[j,j+1]=1
        tperb1=np.random.normal(t,sigma[i])
        tperb2=np.random.normal(t,sigma[i])
        tperb3=np.random.normal(t,sigma[i])
        tperb4=np.random.normal(t,sigma[i])
        H0p=1*H0
        H0p[0,1]=tperb1
        H0p[1,0]=tperb1
        
        H0p[-1,-2]=tperb2
        H0p[-2,-1]=tperb2
        
        H1p=1*H1
        H1p[0,1]=tperb3
        H1p[-1,-2]=tperb4
        Hmenos1p=np.transpose(H1)
        if j>0:
            Mprodizq[j,j-1]=1
        Hmenos1p=np.transpose(H1p)
        Hp= Hp +np.kron(Mprod,H0p)+np.kron(Mprodizq,Hmenos1p)+np.kron(Mprodcha,H1p)
    
    Hp[-(4*Mx):,-(4*Mx):]=H0gorro
    #Hp es nuestro nuevo Hamiltoniano. Lo diagonalizamos en las siguientes
    #lineas        
    energies2, states2 = np.linalg.eig(Hp)
    filtrado(energies2,states2)
    states2=states2[:,energies2<1e6]
    energies2=energies2[energies2<1e6]
    idx2=np.argsort(energies2.real)
    energies2=energies2[idx2].real
    states2=states2[:,idx2]
    Edge2=states2[:,int(len(energies2)/2)]
    #Y a continuacion hacemos el plot de las energias
    #plt.subplot(2,2,i+1)
    '''
    plt.subplot(2,4,i+1)
    bandas()
    plt.title(r'$\sigma=$ '+str(sigma[i]),fontsize=18)
    plt.plot(ejex,energies2[:int(len(energies2)/2)],'r.',
                            label='Estados perturbados')
    plt.plot(np.flip(ejex),energies2[int(len(energies2)/2):],'r.')
    plt.subplot(2,4,i+1+4)
    bandas()
    #plt.title(r'$\sigma=$ '+str(sigma[i]))
    plt.plot(ejex,energies2[:int(len(energies2)/2)],'r.')
    plt.plot(np.flip(ejex),energies2[int(len(energies2)/2):],'r.')
    plt.xlim(ejex[-6],ejex[-1]+1)
    plt.ylim(energies[int(len(energies)/2)-5],energies[int(len(energies)/2)+4])
    #plt.figure()
    #plt.title(r'$\sigma= $'+str(sigma))
    #pintar(aug=5e2,es=Edge2)
    '''
    plt.figure(num=6)
    plt.subplot(2,2,i+1)
    plt.axis('off')
    plt.axis('equal')
    plt.title(r'$\sigma= $'+str(sigma[i]),fontsize=18)
    pintar(aug=5e3,es=Edge2)
'''
#Por ultimo, lo hacemos metiendo ruido en las energías on site:
for i in range(len(sigma)):
    d=np.random.normal(scale=sigma[i],size=4*Mx*My) #solo hace falta cambiar
    #la diagonal
    
    #En los atomos fantasma le ponemos la energia igual a cero para evitar 
    #problemas
    for k in range(1,Mx+1):
        d[4*Mx*(My-1)+4*k-1]=0
    for m in range(0,Mx):
        d[4*Mx*(My-1)+4*m]=0
    
    Hperb3=H+sp.diags(d,0).toarray()
    #Hperb3 es nuestro nuevo hamiltoniano. Pasamos a diagonalizarlo.
    energies3, states3 = np.linalg.eig(Hperb3)
    filtrado(energies3,states3)
    states3=states3[:,energies3<1e6]
    energies3=energies3[energies3<1e6]
    idx3=np.argsort(energies3.real)
    energies3=energies3[idx3].real
    states3=states3[:,idx3]
    Edge3=states3[:,int(len(energies3)/2)]
    plt.figure(num=5)
    plt.subplot(2,4,i+1)
    bandas()
    plt.title(r'$\sigma=$ '+str(sigma[i]),fontsize=18)
    plt.plot(ejex,energies3[:int(len(energies2)/2)],'r.',
                            label='Estados perturbados')
    plt.plot(np.flip(ejex),energies3[int(len(energies2)/2):],'r.')
    plt.subplot(2,4,i+1+4)
    bandas()
    #plt.title(r'$\sigma=$ '+str(sigma[i]))
    plt.plot(ejex,energies3[:int(len(energies2)/2)],'r.')
    plt.plot(np.flip(ejex),energies3[int(len(energies2)/2):],'r.')
    plt.xlim(ejex[-6],ejex[-1]+1)
    plt.ylim(energies[int(len(energies)/2)-5],energies[int(len(energies)/2)+4])
    #plt.title(r'$\sigma= $'+str(sigma))
    pintar(aug=5e2,es=Edge3)
    plt.figure(num=6)
    plt.subplot(2,2,i+1)
    plt.axis('off')
    plt.axis('equal')
    plt.title(r'$\sigma= $'+str(sigma[i]),fontsize=18)
    pintar(aug=5e2,es=Edge3)
    
'''















'''
#No hacer caso a esto, no llego a estar el trabajo. 
#En primer lugar vamos a aplicar ruido a todas las integrales de salto.
sigma=0.25 #Escogemos un valor de sigma

#El Hamiltoniano que queremos construir esta compuesto por tres 
Hp=np.zeros((4*Mx*My,4*Mx*My))
for j in range(0,My-1):
    Mprod=np.zeros((My,My))
    Mprodcha=1*Mprod
    Mprodizq=1*Mprod
    Mprod[j,j]=1
    Mprodcha[j,j+1]=1
    
    diagprip=np.random.normal(loc=t,scale=sigma,size=4*Mx-1)
    H0p=sp.diags((diagprip,0,diagprip),[-1,0,1]).toarray()
    H1p=np.zeros((4*Mx,4*Mx))
    for i in range(0,Mx): 
        h1p=np.array([np.array([0,np.random.normal(loc=t,scale=sigma),0,0]),
                     np.zeros(4),np.zeros(4),
                     np.array([0,0,np.random.normal(loc=t,scale=sigma),0])])
        mprod=np.zeros((Mx,Mx))
        mprod[i,i]=1
        H1p= H1p + sp.kron(mprod,h1p).toarray()
    if j>0:
        Mprodizq[j,j-1]=1
    Hmenos1p=np.transpose(H1p)
    Hp= Hp +np.kron(Mprod,H0p)+np.kron(Mprodizq,Hmenos1p)+np.kron(Mprodcha,H1p)

H0gorrop=np.zeros((4*Mx,4*Mx))

for i in range(0,Mx): 
        h0gorrop=np.array([np.array([0,np.random.normal(loc=t,scale=sigma),0,0]),
                     np.zeros(4),np.zeros(4),
                     np.array([0,0,np.random.normal(loc=t,scale=sigma),0])])
        mprod=np.zeros((Mx,Mx))
        mprod[i,i]=1
        H0gorrop= H0gorrop + sp.kron(mprod,h0gorrop).toarray()        


Hp[-(4*Mx):,-(4*Mx):]=H0gorrop   
energiesp, statesp = np.linalg.eig(Hp)
statesp=statesp.real
#filtrado(energiesp,statesp)
#Aauí vamos a filtrar los estados con las condiciones que se comentan 
#en el texto
#filtrado(energiesp,statesp)
energiesp=energiesp[energiesp<1e6]

idx=np.argsort(energies.real)
energies=energies[idx].real
states=states[:,idx]
idxfermi=[abs(energies)<1e-6]
Edge=states[:,int(len(energies)/2-4)
'''
