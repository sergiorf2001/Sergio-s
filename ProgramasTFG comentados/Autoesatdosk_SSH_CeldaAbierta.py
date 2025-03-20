# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:06:55 2024

@author: Asus
"""
import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import newton
import matplotlib.pyplot as plt

#PARAMETROS DE ENTRADA
'''
CALCULO DE LOS AUTOESTADOS DE UNA CADENA SSH DE CELDAS CERRADAS
    Este codigo es identico al de la cadena de celdas cerradas
    pero hacemos los cambios pertinentes para resolver el caso
    de celdas abiertas.
'''

M=20 #Celdas en la cadena
dm=0.8 #valor de delta (imponemos que t1=1) y hacemos que t2=delta
t2=dm
t1=1


def plot_discontinuous_function(x, y, ax=None,col='b', **kwargs):
    if ax is None:
        ax = plt.gca()

    # Encontrar índices donde hay discontinuidades
    discont_indices = np.where(np.abs(np.diff(y)) > 1)[0]

    if len(discont_indices) == 0:
        # Si no hay discontinuidades, simplemente trazar la función
        ax.plot(x, y,color=col, **kwargs)
    else:
        # Separar la función en segmentos y trazar cada segmento por separado
        start_idx = 0
        for idx in discont_indices:
            ax.plot(x[start_idx:idx+1], y[start_idx:idx+1], color=col,**kwargs)
            start_idx = idx + 1
        ax.plot(x[start_idx:], y[start_idx:],color=col, **kwargs)

    return ax



#estas funciones solo usarlas si se quiere representar
thk= lambda dm,x: np.arctan((dm*np.sin(x))/(1+dm*np.cos(x)))
f3= lambda M, x: np.tanh((M)*x) 
f4= lambda dm, x: (dm*np.sinh(x))*(1-dm*np.cosh(x))**(-1)


#Defino la funcion gk
def gk(x):
    if dm<1:
        return x*(M)+thk(dm,x)
    else:
        breakp=np.arccos(-1/dm)
        if x==breakp:
            return x*(M)+np.pi/2
        elif x>breakp:
            return x*(M)+thk(dm,x)+np.pi
        else: 
            return x*(M)+thk(dm,x)

Mc=dm/(1-dm) #Valor de M critico, del que depende la existencia de estados
#de borde
npi=np.arange(1,M+1,1)*np.pi #multiplos de pi
rootsedge=[]
#Primero calculo los de borde si es que existen
if abs(dm)<1 and M>=Mc:
        #pi=np.delete(npi,-1)
        print('Existen estados de borde')
        def func2(z):
             return f3(M,z)-f4(dm,z)
        raizborde=newton(func2,-np.log(dm),maxiter=100000)
        rootsedge.append(raizborde)

#y ahora los de bulk

rootstemp=np.array([])
plt.figure()
if M>Mc and dm<1:
    plt.title(r'$\delta =$'+str(dm)+ r'  $M =$' +str(M) +r'$>Mc$',fontsize=12)
elif M<Mc and dm<1:
    plt.title(r'$\delta = $'+str(dm) + r'  $Mc>M =$' +str(M),fontsize=12 )
else:
    plt.title(r'$\delta=$'+str(dm)+ r'  $M =$' +str(M),fontsize=12)
plt.xlabel(r'$k$',fontsize=12)
plt.ylabel(r'$g(k)$',fontsize=12)
pi_multiples = np.arange(0, M*np.pi + np.pi, np.pi)
plt.yticks(pi_multiples, [f'{int(i/np.pi)}$\pi$' for i in pi_multiples])

for p in npi:
    #los estados de bulk aparecen cuando gk toma valor de multiplo de pi
    plt.axhline(p,linestyle='--',color='black')
    def func(x):
        return gk(x)-p
    raiz=root_scalar(func,bracket=[0,np.pi],method='brentq').root
    if raiz==np.pi:
        continue
    else:
        rootstemp=np.append(rootstemp,raiz)
        plt.scatter(raiz,p,color='red')

gk=np.vectorize(gk)
inter=np.linspace(0,np.pi,1000)
plot_discontinuous_function(inter,gk(inter))
plt.ylim(0,npi[-1]+0.5)
plt.xlim(0,np.pi)


'''
DISTRIBUCION DE PROBABILIDAD
'''
def generar_lista_arrays(n):
    lista_arrays = np.array(['A0','B0'])

    for i in range(1,n):
        lista_arrays=np.concatenate((lista_arrays,np.array(['A' + str(i), 'B' + str(i)])),axis=0)

    return lista_arrays

#Probabilidades de los estados de bulk
def Pn(M,k,n):
    A=1/(M+0.5*(1-np.sin((2*M+1)*k)/np.sin(k)))
    pb=np.sin(k*(M-n))**2
    pa=np.sin(k*n)**2
    return A*np.array([pa,pb])

#y de borde
def Pedgen(M,k,n,dm):
    #g=dm**4
    #A=(2*(1-g)*g**M)/(1-g**(2*M+1)-(2*M+1)*(1-g)*g**M)
    A=0.5*sum(np.sinh(k*np.arange(1,M+1))**2)**(-1)
    pa=np.sinh(k*(n))**2
    pb=np.sinh(k*(M-n))**2
    return A*np.array([pa,pb])

fig=plt.figure()

k=rootstemp[2]
plt.title(r'$\Delta =$'+str(dm)+ r'  $M =$' +str(M) +r'; $\bar{k}= $' +str(round(k,2)))
probbulk=Pn(M,k,0)
for i in range(1,M+1):
    probbulk=np.concatenate((probbulk,Pn(M,k,i)),axis=0)

histo=generar_lista_arrays(M+1)

plt.bar(histo,probbulk)

if len(rootsedge) != 0:
    plt.figure()
    q=rootsedge[0]
    plt.title(r'$\Delta =$'+str(dm)+ r'  $M =$' +str(M) +r'; $\bar{z}= \pi + $'+str(round(q,2))+'i')

    probedge=Pedgen(M,q,0,dm)
    for i in range(1,M+1):
        probedge=np.concatenate((probedge,Pedgen(M,q,i,dm)),axis=0)
    plt.bar(histo,probedge)
    

f= lambda t1,t2,k: t1*(1+(t1/t2)**2+2*t1/t2*np.cos(k))**0.5
fedge= lambda t1,t2,q: t1*(1+(t1/t2)**2-2*t1/t2*np.cosh(q))**0.5
plt.figure()
k=np.linspace(0,np.pi+0.03,1000)
funcion2=f(t1,t2,np.pi)

#plt.plot(k,e1,color='red')
#plt.plot(k,e2,color='red')
plt.fill_between(k, -funcion2, funcion2, color='green', alpha=0.2)
#plt.vlines(np.pi, -funcion2+0.01, funcion2-0.01, color='blue', label='Anchura', linewidth=2)
#plt.text(s=r'$E_{gap}$',x=np.pi-0.6,y=0.5,fontsize=15,color='blue')
plt.xlim(k[0],k[-1])
#plt.axhline(-funcion2+0.01,linestyle='--',color='grey')
#plt.axhline(funcion2-0.01,linestyle='--',color='grey')
plt.plot(k,f(t1,t2,k),color='black')
plt.plot(k,-f(t1,t2,k),color='black')
plt.ylabel(r'$\varepsilon_k$',fontsize=18)
plt.xlabel(r'$k$',fontsize=14)
plt.xticks([0,np.pi/2,np.pi],['0',r'$\pi/2$',r'$\pi$'],fontsize=13)
plt.yticks([0],[r'$E_F$'],fontsize=13)
plt.axhline(0,linestyle='--',color='grey')
plt.scatter(rootstemp,f(t1,t2,rootstemp),color='red',zorder=100,label='Estados de Bulk')
plt.scatter(rootstemp,-f(t1,t2,rootstemp),color='red',zorder=100)
plt.scatter(np.pi,fedge(t1,t2,rootsedge),color='blue',zorder=100,label='Estados de Borde')
plt.scatter(np.pi,-fedge(t1,t2,rootsedge),color='blue',zorder=100)

plt.legend()
plt.title(r'$\Delta =$'+str(dm)+ r'  $M =$' +str(M))
        