import sys
import numpy as np
from numpy import linalg as LA
import pylab as plt
import matplotlib.mlab as mlab
import scipy

def morse(x, alp, re, de, j, mass ):
    return (de * (np.exp(-2*alp*(x-re))-2*np.exp(-alp*(x-re))) + j*(j+1)/(2.0*mass*x*x))

def kin_cobdvr(dvr,mass):
	n = len(dvr)
	dx = dvr[1]-dvr[0]
	mat = np.zeros((n,n))
	for i in range(n):
		mat[i][i] = np.pi**2/(6.0*mass*dx**2)
		ia = i + 1
		for j in range(i+1,n):
			ja = j + 1
			mat[i][j] = (-1.0)**(ia-ja)/(mass*dx**2*(ia-ja)**2)
		for j in range(i):
			ja = j + 1
			mat[i][j] = (-1.0)**(ia-ja)/(mass*dx**2*(ia-ja)**2)
	return mat

def pot_dvr(dvr,mass,alp,re,de,j):
	n = len(dvr)
	dx = dvr[1]-dvr[0]
	mat = np.zeros((n,n))
	for i in range(n):
		x = dvr[i]
		mat[i][i] = de * (np.exp(-2*alp*(x-re))-2*np.exp(-alp*(x-re))) + j*(j+1)/(2.0*mass*x*x)
	return mat

def cobdvr(dvr,mass,alp,re,de,nvib,j):
 matkin = kin_cobdvr(dvr,mass)
 matpot = pot_dvr(dvr,mass,alp,re,de,j)
 mat = matkin + 1.0*matpot
 eig, vec = LA.eig(mat)
 idx = eig.argsort()
 eig = eig[idx]
 vec = vec[:,idx]
 tvec = np.transpose(vec)
 return eig, tvec

def fcfactors(p1,p2,mass):
    j=0 # no rotation implemented
    alp1= p1[0]
    re1 = p1[1]
    alp2= p2[0]
    re2 = p1[2]
    nvib1 = p1[3]
    nvib2 = p2[3]
    rmin=0.1
    rmax = max(re1+10.0/alp1,re2+10.0/alp2)
    npt = 50*max(nvib1+1,nvib2+1)
    dr = (rmax-rmin)/npt
    #print(rmax,dr)
    dvr = np.arange(rmin,rmax+0.1,dr)
    eig1, vec1 = cobdvr(dvr,mass,*p1,j)
    eig2, vec2 = cobdvr(dvr,mass,*p2,j)
    if eig1[nvib1] > 0.0:
        print('Reduced number of vibrational states')
        print(*eig1[:nvib1])
        sys.exit()
    if eig2[nvib2] > 0.0:
        print('Reduced number of vibrational states')
        print(*eig2[:nvib2])
        sys.exit()
    eig1+=p1[2] # add De, i.e. shift the zero to the electronic state energy
    eig2+=p2[2] # add De, i.e. shift the zero to the electronic state energy
    fcf = np.zeros((nvib2,nvib1))
    for iv in range(nvib1):
      for jv in range(nvib2):
          fcf[jv,iv] = np.sum(vec2[jv][:]*vec1[iv][:])

    return eig1, fcf

if __name__ == '__main__':
    rmax = 6.0
    dvr = np.arange(1.0,rmax+0.1,0.01)
    nvib = [2,3,5] #nb of vibrational levels per electronic state
    mass = 1836.15
    alp1=3.0
    re1=0.74
    de1=0.2
    j1=0 # ground rotational level
    alp2=3.0
    re2=0.74
    de2=0.2
    j2=0 # ground rotational level
    alp3=3.0
    re3=0.74
    de3=0.2
    j3=0 # ground rotational level

    parMorse = [
    [mass,alp1,re1,de1,j1],
    [mass,alp2,re2,de2,j2],
    [mass,alp3,re3,de3,j3],
    ]
    nsta = len(nvib)

    feig = open('eigenvalues','w')
    ffc = open('fcfactors','w')
    print(nsta,file=feig)
    print(*nvib,file=feig)
    for i in range(nsta):
      eig1, vec1 = cobdvr(dvr,*parMorse[i])
      for iv in range(nvib[i]):
        print(eig1[iv],file=feig)
      for j in range(nsta):
         eig2, vec2 = cobdvr(dvr,*parMorse[j])
         for iv in range(nvib[i]):
           for jv in range(nvib[j]):
               print(i,j,iv,jv,np.sum(vec2[jv][:]*vec1[iv][:])**2,file=ffc)
    feig.close()
    ffc.close()

