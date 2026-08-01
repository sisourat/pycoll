from generate_csfs import *
import numpy as np
import sys

from lowdin_nb import *
from numba import jit, njit

# generate the CSFs

def cimat(ovmo, h1emo, r12mo, r12mo_antisym, ne, nmo, csfs, phase):

 ncsfs = len(csfs)
 hmat = np.zeros((ncsfs,ncsfs), dtype=complex)
 smat = np.zeros((ncsfs,ncsfs), dtype=complex)

 # Precompute determinants and phase-adjusted coefficients for all CSFs
 csf_data = []
 for i, csf in enumerate(csfs):
     csf_terms = []
     for j in range(csf.nterms):
         c, alp, bet = csf.terms[j]
         # Precompute phase-adjusted coefficient
         c_phase = c * phase[i]
         # Precompute determinant object
         det = Sdeterminant(len(alp), len(bet), np.array(alp), np.array(bet))
         csf_terms.append((c_phase, det))
     csf_data.append(csf_terms)

 # Optimized computation using precomputed data
 for i1, csf1_data in enumerate(csf_data):
    for i2, csf2_data in enumerate(csf_data):
        hmat_val = 0.0 + 0.0j
        smat_val = 0.0 + 0.0j

        for j1 in range(len(csf1_data)):
            c1_phase, det1 = csf1_data[j1]
            for j2 in range(len(csf2_data)):
                c2_phase, det2 = csf2_data[j2]

                ov, h1e, r12 = lowdin(ovmo, h1emo, r12mo, r12mo_antisym, det1, det2)

                # Accumulate results
                hmat_val += c1_phase * np.conj(c2_phase) * (h1e + r12)
                smat_val += c1_phase * np.conj(c2_phase) * ov

        hmat[i1, i2] = hmat_val
        smat[i1, i2] = smat_val

 return hmat, smat
