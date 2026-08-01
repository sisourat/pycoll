import sys
import numpy as np


dat = np.loadtxt(sys.argv[1])
bimp = dat[:,0]
nbimp = len(bimp)
nsta = len(dat[0,:])-2
bprob = np.zeros((nbimp,nsta))
bpbiem = np.zeros((nbimp,nsta))
for ib in range(nbimp):
    for i in range(nsta):
        bprob[ib,i] = bimp[ib]*dat[ib,1+i]
        p = dat[ib,1+i]
        piem = 2.0*p*(1.0-p)
        bpbiem[ib,i] = bimp[ib]*piem

sig = []
for i in range(nsta):
 sig.append(np.trapz(bprob[:,i],bimp)*2.0*np.pi*0.28)
 print(i,sig[i])

print()
sexc = np.sum(sig[1:6])
s1s = np.sum(sig[6:12])
s2s = np.sum(sig[12:18])
s2p = np.sum(sig[18:])
print('Tot',s1s,sexc)#2s,s2p)
print()
for i in range(nsta):
 print(i,sig[i]/s1s)


