# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:14:41 2024

@author: Asus
"""

import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import newton
import matplotlib.pyplot as plt
import scipy.sparse as sp



'''
VARIABLES DE ENTRADA: introducir en Mx y My las dimensiones de la cinta
'''
Mx=9#Numero de dobles celdas en el eje x
My=8 #numero de dobles celdas verticales 
N=2*My-1 #numero de lineas dimeras en el eje y


#Parametros dimensionales del grafeno
c=1 #Constante honeycomb
a=c/(2*np.sin(60/180*np.pi)) #distancia entre primeros vecinos
t=2.7 #valor de la integral de salto
'''
CALCULO DE LOS KX PERMITIDOS 
'''

#En primer lugar calculamos los ky permitidos
ky=np.zeros(My-1)
for i in range(0,My-1):
    ky[i]=(i+1)*(np.pi/My)
dmBig=2*np.cos(ky/2) #deltas 

#para cada conjunto de ky habrá ciertos kx permitidos que vienen dados
#por la condicion de cuantización. Calcularemos en primer lugar 
#los estados de bulk.

#Importamos las funciones que nos dan la condicion de cuantizacion
thk= lambda dm,x: np.arctan((dm*np.sin(x/2))/(1+dm*np.cos(x/2))) #\theta_k
def gk(dm,M,x): #funcion g(k) que cuando vale n*pi indica el k permitido
    if dm<1:
        return x*(M)+thk(dm,x)
    else:
        breakp=2*np.arccos(-1/dm)
        
        if x==breakp:
            return x*(M)-np.pi/2
        elif x>breakp:
            return x*(M)+thk(dm,x)+np.pi
        else: 
            return x*(M)+thk(dm,x)
gk2=np.vectorize(gk)

#Para los estados de borde, cuando los haya, los q permitidos son los cruces
#entre f3 y f4
f3= lambda Mx, x: np.tanh(Mx*x) 
f4= lambda dm, x: (dm*np.sinh(x/2))*(1-dm*np.cosh(x/2))**(-1)
#por lo tanto ahora para cada ky debemos calcular los puntos correspondientes
#kx 

roots=[] #Aqui vamos a guardar todos los estados de bulk
rootsedge=[] #Aqui los de borde
npi=np.arange(1,2*Mx+1,1)*np.pi #Este vector lo uso para calculos

#INICIALIZAMOS CON UN BUCLE EL CALCULO DE LOS KX PERMITIDOS
for j in range(0,len(dmBig)):
    #para cada valor de ky, y por lo tanto de delta 
    #calculamos los kx permitidos
    dm=dmBig[j]
    Mc=np.ceil(dm/(2*(1-dm))) #valor critico de dm cuando dm<1
    #Primero calculo los de borde si es que existen
    if Mx>=Mc and abs(dm)<=1:
        #si se dan las condiciones buscamos el valor de q para
        #el estado de brode
        def func2(z):
             return f3(2*Mx,z)-f4(dm,z)
        #lo hacemos con el metodo de newton
        raizborde=newton(func2,-2*np.log(dm))
        #y lo agregamos a rootsedge acompañado de el borde 
        #de la zona de brillouin del ejex y de el valor del ky
        #para el que se encuentra
        rootsedge.append(np.array([np.pi*2,ky[j],raizborde]))
    #y ahora los de bulk
    rootstemp=np.array([]) #array temporal

    for p in npi:
        
        #para cada multiplo de pi, miramos donde se cruza
        #la funcion gk
        def func(x):
            return gk(dm,Mx,x)-p #notar que M=2*Mx es el numero de celdas SSH
        raiz=root_scalar(func,bracket=[0,2*np.pi],method='brentq').root
        if raiz==np.pi:
            #evitamos que nos coja valores de la raiz que corresponden al borde
            #de la zona de brillouin
            continue
        else:
            #y añadimos en otro caso
            rootstemp=np.append(rootstemp,raiz)
    rootstemp=np.array(rootstemp) #deshacemos el cambio, pues k=kx/2
    roots.append(rootstemp)


roots=np.array(roots) #cada array i-esimo contenidos en roots 
                      #contiene las raices de los estados de bulk que se 
                      #encuentran para la componente i-esima de ky
lastroots=np.array([])
for i in range(1,Mx+1):
    #el caso ky=pi es especial y lo tratamos aparte
    lastroots=np.append(lastroots,np.pi*i/(Mx))
rootsedge=np.array(rootsedge) #cada una de las filas de este array contiene 
#tres elementos: el primero es 2pi, para recordarnos que la parte real de k
#es igual a 2pi. El segundo es el valor de ky corresponiente a este estado
#de borde y el último es el valor de q (parte imaginaria de kx)

#Ya tenemos todos los estados calculados
#Aqui hacemos el plot de la primera figura
fig=plt.figure()
plt.ylim(0,np.pi+2*np.pi/(N+1))
x_ticks_locations = [n*np.pi/2 for n in range(1, 5)]
x_ticks_labels = [r'$\pi/2$',r'$\pi$',r'$3\pi/4$',r'$2\pi$']
plt.xticks(x_ticks_locations, x_ticks_labels,fontsize=16)
y_ticks_locations = np.append(ky,np.pi)*1
y_ticks_labels = [f'{n}$\pi/{int(My)}$' for n in range(1, My)]
y_ticks_labels.append(r'$\pi$')
plt.yticks(y_ticks_locations, y_ticks_labels,fontsize=16)
plt.title(r'$M_x=$'+str(Mx)+r', $M_y=$'+str(My),fontsize=20)
for i in range(0,len(roots[0])):
    plt.axvline(roots[0][i],linestyle='--',color='green')
plt.xlabel(r'$\bar{k}_{x}$',fontsize=18)
plt.ylabel(r'$\bar{k}_{y}$',fontsize=18)

for i in range(0,len(ky)):
    plt.axhline(ky[i],color='red')
    plt.plot(roots[i],ky[i]*np.ones_like(roots[i]),'.',color='black',
             markersize=12)
plt.axhline(np.pi,color='red')
plt.plot(lastroots,np.pi*np.ones_like(lastroots),'.',color='black',
         markersize=12)

for i in range(0,len(rootsedge)):
    plt.plot(rootsedge[i][0],rootsedge[i][1],marker='.',color='b',
             markersize=12)
    


'''
REPRESENTACIÓN DE LA DISTRIBUCIÓN DE PROBABILIDAD
'''
#Representamos la distribuccion de probabilidad del electron para los estados
#de bulk.

#Primero debemos de dibujar la red directa de los rectángulos del grafeno.
#Utilizando los parámetros anteriores definimos las componentes "x" e "y"
#de los vectores d que definimos en el texto. En este caso, vamos a inter
#cambiar las etiquetas y los atomos de la pareja (A1,B1) se posicionan en
#la pareja inferior de la celda unidad y los (A2,B2) en la superior.

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

B1x = A1x+a
B2x=B1x+d2x
B1y = A1y
B2y=B1y+d2y

#A partir de aqui y hasta la linea 213 volvemos estos datos mas manejables,
#apartamos los atomos fantasma de los de verdad...
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

#Pasamos a representar la figura en la que se muestran tanto los atomos
#falsos como los verdaderos.
plt.figure()
#plt.title('Representacion grafica del ribbon')
points=np.concatenate((A1lista,A2lista,B1lista,B2lista))
plt.axis('off')
plt.axis('equal')
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

plt.scatter(A1[0],A1[1],color='b')
plt.scatter(B1[0],B1[1],color='b')
plt.scatter(A2[0][1:-1,1:],A2[1][1:-1,1:],color='b')
#plt.scatter(A20x,A20y,color='orange')
#plt.scatter(A2Mx,A2My,color='orange')
plt.scatter(B2[0][1:-1,:-1],B2[1][1:-1,:-1],color='b')
#plt.scatter(B20x,B20y,color='orange')
#plt.scatter(B2Mx,B2My,color='orange')
plt.scatter(Bordesx,Bordesy,color='orange')

for i in range(len(Bordes)):
    #este bucle habría que optimizarlo pues es evidente que no me 
    #hace falta recorrer cada uno de los puntos pero bueno
    point1=Bordes[i]
    points2=[]
    for j in range(len(points)):
        if  abs(np.linalg.norm(point1-points[j])-a)<1e-3:
            points2.append(points[j])
    for p in points2:
        plt.plot([p[0],point1[0]],[p[1],point1[1]],'--',color='grey')



#Por fin podemos pasar a pintar las probabilidades. Primero pintamos tan solo
#los atomos verdaeros.
plt.figure()
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


#Escogemos aqui el estado que nos interesa
m=3
alfa=5
Kx= roots[m][alfa]
Ky=ky[m]
plt.title(r'$\bar{k}_x=$ ' +str(round(Kx,3))+ 
          r', $\bar{k}_y=$'+str(m+1)+r'$\pi/$'  + str(My),fontsize=20 )

#Y aqui introducimos los modulos cuadrados de la funion de onda para cada 
#atomo de la celda unidad
C=4/(My*8*Mx)
def pA2(nx,ny,Kx=Kx,Ky=Ky):
    return C*np.sin(Kx*nx)**2*np.sin(Ky*ny)**2
def pA1(nx,ny,Kx=Kx,Ky=Ky):
    return C*np.sin(Kx*(nx+0.5))**2*np.sin(Ky*(ny+0.5))**2
def pB2(nx,ny,Kx=Kx,Ky=Ky):
    return C*np.sin(Kx*(Mx-nx))**2*np.sin(Ky*ny)**2
def pB1(nx,ny,Kx=Kx,Ky=Ky):
    return C*np.sin(Kx*(Mx-(nx+0.5)))**2*np.sin(Ky*(ny+0.5))**2

aug=1.5e5

#Finalmente con dos bucles (uno para cada los atomos A2,B2  y otro para los
#(A1,B1), pintamos la distribuccion de probabilidadd del estado de Bulk
#escogido.

for nx in range(0,Mx+1):
    if nx==0:  
        for ny in range(1,My+1):
            pb2=pB2(nx,ny,Kx,Ky)
            plt.scatter(B2[0][0,nx],B2[1][ny][0],
                        s=aug*pb2,color='red',alpha=0.7)
    elif nx==Mx:
        for ny in range(1,My+1):
            pa2=pA2(nx,ny,Kx,Ky)
            plt.scatter(A2[0][0,nx],A2[1][ny][0],
                        s=aug*pa2,color='red',alpha=0.7)
    else:
        for ny in range(1,My):
            pa2=pA2(nx,ny,Kx,Ky)
            plt.scatter(A2[0][0,nx],A2[1][ny][0],
                        s=aug*pa2,color='red',alpha=0.7)
            pb2=pB2(nx,ny,Kx,Ky)
            plt.scatter(B2[0][0,nx],B2[1][ny][0],
                        s=aug*pb2,color='red',alpha=0.7)


for nx in range(0,Mx):
    for ny in range(0,My):
        pa1=pA1(nx,ny,Kx,Ky)
        plt.scatter(A1[0][0,nx],A1[1][ny][0],
                    s=aug*pa1,color='red',alpha=0.7)
        pb1=pB1(nx,ny,Kx,Ky)
        plt.scatter(B1[0][0,nx],B1[1][ny][0],
                    s=aug*pb1,color='red',alpha=0.7)
    pa1=pA1(nx,My,Kx,Ky)
    plt.scatter(A1[0][0,nx],A1[1][len(A1[1])-1][0],
                s=aug*pa1,color='red',alpha=0.7)
    pb1=pB1(nx,My,Kx,Ky)
    plt.scatter(B1[0][0,nx],B1[1][len(A1[1])-1][0],
                s=aug*pb1,color='red',alpha=0.7)

plt.axis('off')
plt.axis('equal')


'''
ESTRUCTURA DE BANDAS EN RIBBONS FINITOS
'''
#Conocemos las exoresiones analiticas de las energias, tanto de los estados de 
#borde como de los de bulk. Son las siguientes.
def ek(ky,kx,t=t): 
    #Energia de los estados de bulk
    dm=2*np.cos(ky/2)
    return t*np.sqrt(1+dm**2+2*dm*np.cos(kx/2))

def eq(ky,q,t=t):
    #Energia de los estados de borde
    dm=2*np.cos(ky/2)
    return t*np.sqrt(np.abs(1+dm**2-2*dm*np.cosh(q/2)))

#Para pintarlas, primero calculamos las de los ribbons infinitos
#tanto de zigzag como de armchair (igual que en los respectivos codigos).
def eix(x):
    if np.isclose(x % np.pi, 0):
        return np.cos(x)
    elif np.isclose(x % (np.pi / 2), 0) and not np.isclose(x % np.pi, 0):
        return 1j*np.sin(x)
    else:
        return np.cos(x)+1j*np.sin(x)
eix=np.vectorize(eix)

#CODIGO PARA LAS BANDAS ZIGZAG

#Defino las matrices que forman parte del hamiltoniano de la supercelda
H0=sp.diags((t*np.ones(4*Mx-1),0,t*np.ones(4*Mx-1)),[-1,0,1]).toarray()

h1=np.array([np.array([0,t,0,0]),np.zeros(4),np.zeros(4),np.array([0,0,t,0])])
H1=sp.kron(np.eye(Mx),h1).toarray()
Hmenos1=np.transpose(H1)

kyRepr=np.linspace(0,np.pi+0.05,200)
x=kyRepr[0]
Hk=Hmenos1*eix(-x)+H0+eix(x)*H1 #Hamiltoniano de la super celda



#Diagonalizamos el Hamiltoniano para un prmier valor de k
energies, states = np.linalg.eig(Hk)
idx=np.argsort(energies.real)
energies=energies[idx].real #nos quedamos con la parte real de las energias
                            #Se puede comprobar que la imaginaria es infima
states=states[:,idx]
EnergiesBig=energies

for i in range(1,len(kyRepr)):
    x=kyRepr[i]
    Hk=Hmenos1*eix(-x)+H0+eix(x)*H1
    energies, states = np.linalg.eig(Hk)
    idx=np.argsort(energies.real)
    energies=energies[idx].real
    states=states[:,idx]
    EnergiesBig=np.vstack((EnergiesBig,energies))

#Pasamos a hacer las figuras
plt.figure()

#Primero pintamos las energias de la cinta de grafeno.
for i in range(0,len(ky)):
    r=roots[i]
    for j in range(len(r)):
        plt.scatter(ky[i],ek(ky[i],r[j],t=2.7),color='black',zorder=1000)
        plt.scatter(ky[i],-ek(ky[i],r[j],t=2.7),color='black',zorder=1000)

for j in range(len(lastroots)):
    plt.scatter(np.pi,ek(np.pi,lastroots[j],t=2.7),color='black',zorder=1000)
    plt.scatter(np.pi,-ek(np.pi,lastroots[j],t=2.7),color='black',zorder=1000)

for n in range(len(rootsedge)):
        plt.scatter(rootsedge[n][1],eq(rootsedge[n][1],
                    rootsedge[n][2],t=1),color='blue',zorder=1000)
        plt.scatter(rootsedge[n][1],-eq(rootsedge[n][1],
                    rootsedge[n][2],t=1),color='blue',zorder=1000)
plt.title(r'$Mx$= '+str(Mx)+r', $M_y$= '+str(My),fontsize=18) 
plt.xlabel(r'$\bar{k}_y$',fontsize=18)
plt.ylabel(r'$\varepsilon$',fontsize=18)
x_ticks_locations = [0,np.pi/3,2*np.pi/3,np.pi]
x_ticks_labels = [r'$0$',r'$\pi/3$',r'$2\pi/3$',r'$\pi$']
plt.xticks(x_ticks_locations, x_ticks_labels,fontsize=12)
plt.yticks([])
#Y aqui las de los ribbons infinitos
for j in range(len(energies)-1):
    plt.plot(kyRepr[:],EnergiesBig[:, j],color='red')
plt.xlim(0,np.pi)


#CODIGO PARA PINTAR LAS BANDAS ARMCHAIR

#Primero hago una figura en la que se dejan ver todas
#las bandas. EL codigo es el mismo que para el caso en
#el que calculamos las bandas de los ribbons armchair infinitos
#Van a haber bandas que nos tienen autoestados encima.

h1=sp.diags((t*np.ones(3),0,t*np.ones(3)),[-1,0,1]).toarray()
h2=np.array([np.zeros(4),np.array([0,0,t,0]),np.zeros(4),np.zeros(4)])
h3=np.transpose(h2)
HM0y=sp.kron(np.eye(My-1),h1).toarray()+sp.kron(sp.diags(np.ones(My-2),
                    -1).toarray(),h2).toarray()+sp.kron(sp.diags(np.ones(My-2),
                               1).toarray(),h3).toarray()
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

Hmenos1=np.zeros((4*(My-1)+2,4*(My-1)+2))
h5=np.array([np.zeros(4),np.zeros(4),np.zeros(4),np.array([t,0,0,0])])
Hmenos1[:4*(My-1),:4*(My-1)]=sp.kron(sp.diags((np.ones(My-2),np.ones(My-1)),
       (1,0)),h5).toarray()
Hmenos1[:,-2:][-3][0]=t

H1=np.transpose(Hmenos1)

kxRepr=np.linspace(0,2*np.pi+1,200)

x=kxRepr[0]
Hk=Hmenos1*eix(-x)+H0+eix(x)*H1

energies, states = np.linalg.eig(Hk)
energies=np.sort(energies.real)
EnergiesBig=np.sort(energies.real)

for i in range(1,len(kxRepr)-1):
    x=kxRepr[i]
    Hk=Hmenos1*eix(-x)+H0+eix(x)*H1
    energies, states = np.linalg.eig(Hk)
    energies=np.sort(energies.real)
    EnergiesBig=np.vstack((EnergiesBig,energies))

plt.figure()
for i in range(0,len(ky)):
    r=roots[i]
    for j in range(len(r)):
        plt.scatter(r[j],ek(ky[i],r[j],t=2.7),color='black',zorder=1000)
        plt.scatter(r[j],-ek(ky[i],r[j],t=2.7),color='black',zorder=1000)
for n in range(len(rootsedge)):
        plt.scatter(rootsedge[n][0],eq(rootsedge[n][1],
                    rootsedge[n][2],t=1),color='blue',zorder=1000)
        plt.scatter(rootsedge[n][0],-eq(rootsedge[n][1],
                    rootsedge[n][2],t=1),color='blue',zorder=1000)
for j in range(len(lastroots)):
    plt.scatter(lastroots[j],ek(np.pi,lastroots[j],t=2.7),
                color='black',zorder=1000)
    plt.scatter(lastroots[j],-ek(np.pi,lastroots[j],t=2.7),
                color='black',zorder=1000)
plt.title(r'$Mx$= '+str(Mx)+r', $M_y$= '+str(My),fontsize=18) 
plt.xlabel(r'$\bar{k}_x$',fontsize=18)
plt.ylabel(r'$\varepsilon$',fontsize=18)
x_ticks_locations = [np.pi/2,np.pi,3*np.pi/2,2*np.pi]
x_ticks_labels = [r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$']
plt.xticks(x_ticks_locations, x_ticks_labels,fontsize=12)
plt.yticks([])
plt.xlim(0,2*np.pi+0.09)



for j in range(len(energies)):
    plt.plot(kxRepr[:-1],EnergiesBig[:, j],color='red')

plt.figure()
for ks in ky:
    plt.plot(kxRepr,ek(ks,kxRepr,t=2.7),color='red')
    plt.plot(kxRepr,-ek(ks,kxRepr,t=2.7),color='red')

plt.plot(kxRepr,ek(np.pi,kxRepr,t=2.7),color='red')
plt.plot(kxRepr,-ek(np.pi,kxRepr,t=2.7),color='red')
for i in range(0,len(ky)):
    r=roots[i]
    for j in range(len(r)):
        plt.scatter(r[j],ek(ky[i],r[j],t=2.7),color='black',zorder=1000)
        plt.scatter(r[j],-ek(ky[i],r[j],t=2.7),color='black',zorder=1000)
for n in range(len(rootsedge)):
        plt.scatter(rootsedge[n][0],eq(rootsedge[n][1],
                    rootsedge[n][2],t=1),color='blue',zorder=1000)
        plt.scatter(rootsedge[n][0],-eq(rootsedge[n][1],
                    rootsedge[n][2],t=1),color='blue',zorder=1000)
for j in range(len(lastroots)):
    plt.scatter(lastroots[j],ek(np.pi,lastroots[j],t=2.7),
                color='black',zorder=1000)
    plt.scatter(lastroots[j],-ek(np.pi,lastroots[j],t=2.7),
                color='black',zorder=1000)
plt.title(r'$Mx$= '+str(Mx)+r', $M_y$= '+str(My),fontsize=18) 
plt.xlabel(r'$\bar{k}_x$',fontsize=18)
plt.ylabel(r'$\varepsilon$',fontsize=18)
x_ticks_locations = [np.pi/2,np.pi,3*np.pi/2,2*np.pi]
x_ticks_labels = [r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$']
plt.xticks(x_ticks_locations, x_ticks_labels,fontsize=12)
plt.yticks([])
plt.xlim(0,2*np.pi+0.08)


    