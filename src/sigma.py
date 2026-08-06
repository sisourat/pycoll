#!/usr/bin/env python3
import sys
import numpy as np


outdir = sys.argv[1]
probdat = np.loadtxt(outdir+'/prob')
stadat = np.loadtxt(outdir+'/sta')

eindex = stadat[:,0]
vindex = stadat[:,1]
esta = stadat[:,2]
nsta = len(esta)

bimp = probdat[:,0]
nbimp = len(bimp)
bprob = np.zeros((nbimp+1,nsta))
b = np.zeros((nbimp+1))
for ib in range(nbimp):
    for i in range(nsta):
        p = probdat[ib,i+1]
        bprob[ib+1,i] = bimp[ib]*p
        b[ib+1] = bimp[ib]

sig = []
k=0
for i in range(nsta):
 sig.append(np.trapz(bprob[:,i],b)*2.0*np.pi*0.28)
 print(k,'   ', int(eindex[i]),int(vindex[i]),esta[i],'     ',sig[i])
 k+=1

print()
sexc = np.sum(sig[1:5])
scapt = np.sum(sig[5:])
#s2s = np.sum(sig[12:18])
#s2p = np.sum(sig[18:])
print('Tot',sexc,scapt)#2s,s2p)
print()
for i in range(nsta):
 print(i,sig[i]/scapt)


