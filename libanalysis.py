import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from lowdin_nb import *

def evaluate_gto(mu, r):
    # Implement GTO evaluation for basis function mu at point r
    return g_mu

def plot_rho():
    # Define grid
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    z = np.linspace(-10, 10, 100)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)

    # Evaluate GTOs on the grid
    # Compute density at each grid point
    rho = np.zeros(len(grid_points))
    for i, r in enumerate(grid_points):
        for mu in range(n_gto):
            for nu in range(n_gto):
                rho[i] += P[mu, nu] * evaluate_gto(mu, r) * evaluate_gto(nu, r)

    #NICO P is the 1RDM to be implemented
    # Reshape and plot
    rho_grid = rho.reshape(X.shape)
    # Plot a slice (e.g., z=0)
    plt.imshow(rho_grid[:, :, 50], cmap='viridis')
    plt.colorbar()
    plt.show()


def det_overlap(occ_I, occ_J, S):
     """
     Computes determinant of the submatrix S[I_occ, J_occ]
     occ_I, occ_J: list of occupied orbital indices
     S: orbital-overlap matrix
     """
     sub = S[np.ix_(occ_I, occ_J)]
     return np.linalg.det(sub)

def one_rdm_nonorth(csfs, cicoeffs, ovlmo):
    """
    csfs : (ncsfs,)
    cicoeffs : (ncsfs,)        CI coefficients
    ovlm    : (nmo, nmo)   global orbital overlap matrix

    Returns gamma[p,q] in the same (non-orthogonal) orbital basis.
    """
    ncsfs = len(csfs)
    nmo = len(ovlmo)
    gamma = np.zeros((2*nmo, 2*nmo),dtype=np.complex128)
    gamma2 = np.zeros((2*nmo, 2*nmo, 2*nmo, 2*nmo),dtype=np.complex128)

    ovl = np.zeros((2*nmo,2*nmo))
    ovl[0:nmo,0:nmo] = ovlmo
    ovl[nmo:2*nmo,nmo:2*nmo] = ovlmo

    #cicoeffs[0] = 0.707106781
    #cicoeffs[1] = 0.707106781
    #cicoeffs[2] = 0.0

    for i1, csf1 in enumerate(csfs):
        nterm1 = csf1.nterms
        for i2, csf2 in enumerate(csfs):
            nterm2 = csf2.nterms
            for j1 in range(nterm1):
                c1, alp, bet = csf1.terms[j1]
                alp = np.array(alp)
                bet = np.array(bet) + nmo
                indices = np.concatenate([alp, bet])
                c1 = c1 * cicoeffs[i1]
                det1 = [1 if i in indices else 0 for i in range(2*nmo)]
                for j2 in range(nterm2):
                     c2, alp, bet = csf2.terms[j2]
                     alp = np.array(alp)
                     bet = np.array(bet) + nmo
                     indices = np.concatenate([alp, bet])
                     c2 = c2 * cicoeffs[i2]
                     det2 = [1 if i in indices else 0 for i in range(2*nmo)]

                     # Compute TDM
                     #print(i1,i2)
                     #print(det1,det2)
                     tdm = compute_tdm(det1, det2, ovl)
                     #tdm2 = compute_2tdm(det1, det2, det1, det2, ovl)

                     # Accumulate contributions to the RDM
                     gamma += c2 * np.conj(c1) * tdm
                     #gamma2 += c2 * np.conj(c1) * tdm2

    return gamma#, gamma2

def compute_tdm(det_i, det_j, S):
    """
    Compute the transition density matrix (TDM) between two Slater determinants
    in a non-orthogonal orbital basis.

    Args:
        det_i: Occupation vector for determinant i (e.g., [1,1,0,0]).
        det_j: Occupation vector for determinant j (e.g., [1,0,1,0]).
        S: Overlap matrix of the non-orthogonal orbitals.

    Returns:
        TDM between det_i and det_j (nmo x nmo).
    """
    n_orbitals = len(det_i)
    tdm = np.zeros((n_orbitals, n_orbitals), dtype=complex)

    # Check if the determinants have the same number of electrons
    if sum(det_i) != sum(det_j):
        return tdm  # TDM is zero if different electron numbers

    # Find the indices of the differing orbitals
    diff_indices = [k for k, (di, dj) in enumerate(zip(det_i, det_j)) if di != dj]

    # If more than two orbitals differ, TDM is zero (Slater rules)
    if len(diff_indices) > 2:
        return tdm

    # Case 1: det_i == det_j (diagonal TDM)
    if len(diff_indices) == 0:
        for p in range(n_orbitals):
            if det_i[p] == 1:
                tdm[p, p] = 1.0
        return tdm

    # Case 2: det_i and det_j differ by one orbital (TDM is zero)
    if len(diff_indices) == 1:
        return tdm

    # Case 3: det_i and det_j differ by two orbitals (p and q)
    p, q = diff_indices
    if det_i[p] == 1 and det_j[q] == 1:
        # Compute the cofactor for the (p,q) element
        minor = np.delete(np.delete(S, p, axis=0), q, axis=1)
        cofactor = ((-1) ** (p + q)) * np.linalg.det(minor)
        tdm[p, q] = cofactor / np.sqrt(np.linalg.det(S))
    elif det_i[q] == 1 and det_j[p] == 1:
        # Compute the cofactor for the (q,p) element
        minor = np.delete(np.delete(S, q, axis=0), p, axis=1)
        cofactor = ((-1) ** (p + q)) * np.linalg.det(minor)
        tdm[q, p] = cofactor / np.sqrt(np.linalg.det(S))

    return tdm

def compute_2tdm(det_i1, det_j1, det_i2, det_j2, S):
    """
    Compute the transition 2-RDM between two pairs of Slater determinants
    in a non-orthogonal orbital basis.

    Args:
        det_i1: Occupation vector for determinant i (spin 1).
        det_j1: Occupation vector for determinant j (spin 1).
        det_i2: Occupation vector for determinant i (spin 2).
        det_j2: Occupation vector for determinant j (spin 2).
        S: Overlap matrix of the non-orthogonal orbitals.

    Returns:
        Transition 2-RDM between the determinants (nmo x nmo x nmo x nmo).
    """
    n_orbitals = len(det_i1)
    tdm_2 = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals), dtype=complex)

    # Check if the determinants have the same number of electrons
    if sum(det_i1) != sum(det_j1) or sum(det_i2) != sum(det_j2):
        return tdm_2

    # Find the indices of the differing orbitals for spin 1 and spin 2
    diff_indices1 = [k for k, (di, dj) in enumerate(zip(det_i1, det_j1)) if di != dj]
    diff_indices2 = [k for k, (di, dj) in enumerate(zip(det_i2, det_j2)) if di != dj]

    # If more than two orbitals differ for either spin, the 2-TDM is zero
    if len(diff_indices1) > 2 or len(diff_indices2) > 2:
        return tdm_2

    # Implement the Slater rules for the 2-TDM here
    # This is a simplified placeholder; the actual implementation depends on the specific form of the CSFs
    # For a full implementation, you would need to account for all possible cases of differing orbitals
    # and compute the appropriate cofactors using the overlap matrix S

    # Placeholder: For identical determinants, the 2-TDM is the wedge product of the 1-TDMs
    if len(diff_indices1) == 0 and len(diff_indices2) == 0:
        for p in range(n_orbitals):
            for q in range(n_orbitals):
                for r in range(n_orbitals):
                    for s in range(n_orbitals):
                        if det_i1[p] == 1 and det_i1[q] == 1 and det_i2[r] == 1 and det_i2[s] == 1:
                            tdm_2[p, q, r, s] = 1.0

    return tdm_2
