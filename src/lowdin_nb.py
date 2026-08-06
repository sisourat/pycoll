import numpy as np
from numba import njit, int64, prange
from numba.experimental import jitclass
from numba.typed import List
from numba import optional
import sys

# -------------------- 1. Jitclass for Slater Determinant --------------------
spec = [
    ('nalpha', int64),
    ('nbeta', int64),
    ('alpha', int64[:]),
    ('beta', int64[:]),
]

@jitclass(spec)
class Sdeterminant:
    def __init__(self, nalpha, nbeta, alpha, beta):
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.alpha = np.asarray(alpha, dtype=np.int64)
        self.beta = np.asarray(beta, dtype=np.int64)

# -------------------- 2. Determinant Calculation --------------------
@njit(fastmath=True)
def compute_det(mat):
    return np.linalg.det(mat)

# -------------------- 3. Copy Matrix Excluding Rows/Columns --------------------
@njit(fastmath=True)
def copy_excluding(mat, out, row_indices, col_indices):
    n_out, m_out = out.shape
    row_skip = set(row_indices)
    col_skip = set(col_indices)

    ri = 0
    for i in range(mat.shape[0]):
        if i in row_skip:
            continue
        ci = 0
        for j in range(mat.shape[1]):
            if j in col_skip:
                continue
            out[ri, ci] = mat[i, j]
            ci += 1
        ri += 1
    out += 1e-15

# -------------------- 4. Precompute Minor Indices and Signs --------------------
@njit(fastmath=True)
def precompute_minor_info(det1, det2):
    ne = det1.nalpha + det1.nbeta

    # Use Numba-compatible lists
    alpha_alpha_info = List()
    alpha_beta_info = List()
    beta_beta_info = List()
    h1e_alpha_info = List()
    h1e_beta_info = List()

    # One-electron minor info (alpha)
    for i in range(det1.nalpha):
        for j in range(det2.nalpha):
            sign = (-1) ** (i + j)
            h1e_alpha_info.append((i, j, sign))

    # One-electron minor info (beta)
    for i in range(det1.nbeta):
        for j in range(det2.nbeta):
            sign = (-1) ** (det1.nalpha + det2.nalpha + i + j)
            h1e_beta_info.append((i, j, sign))
    if ne<2:
     return alpha_alpha_info, alpha_beta_info, beta_beta_info, h1e_alpha_info, h1e_beta_info

    # Two-electron minor info (alpha-alpha)
    for i in range(det1.nalpha):
        for k in range(i + 1, det1.nalpha):
            for j in range(det2.nalpha):
                for l in range(j + 1, det2.nalpha):
                    sign = (-1) ** (i + j + k + l)
                    alpha_alpha_info.append((i, k, j, l, sign))

    # Two-electron minor info (alpha-beta)
    for i in range(det1.nalpha):
        for k in range(det1.nbeta):
            for j in range(det2.nalpha):
                for l in range(det2.nbeta):
                    sign = (-1) ** (i + j + k + l + det1.nalpha + det2.nalpha)
                    alpha_beta_info.append((i, k, j, l, sign))

    # Two-electron minor info (beta-beta)
    for i in range(det1.nbeta):
        for k in range(i + 1, det1.nbeta):
            for j in range(det2.nbeta):
                for l in range(j + 1, det2.nbeta):
                    sign = (-1) ** (i + j + k + l + 2 * det1.nalpha + 2 * det2.nalpha)
                    beta_beta_info.append((i, k, j, l, sign))

    return alpha_alpha_info, alpha_beta_info, beta_beta_info, h1e_alpha_info, h1e_beta_info

# -------------------- 6. Compute Overlap and One-Electron Term --------------------
@njit(fastmath=True)
def compute_ov_and_h1e(ovmo, h1emo, det1, det2, h1e_alpha_info, h1e_beta_info):
    ne = det1.nalpha + det1.nbeta
    ovmat = np.zeros((ne, ne), dtype=np.complex128)

    # Overlap matrix
    for i in range(det1.nalpha):
        ia = det1.alpha[i]
        for j in range(det2.nalpha):
            ja = det2.alpha[j]
            ovmat[j, i] = ovmo[ja, ia] + 1e-15
    for i in range(det1.nbeta):
        ib = det1.beta[i]
        for j in range(det2.nbeta):
            jb = det2.beta[j]
            ovmat[j + det2.nalpha, i + det1.nalpha] = ovmo[jb, ib] + 1e-15

    ov = compute_det(ovmat)

    # One-electron term
    h1e = 0.0 + 0.0j
    comat = np.zeros((ne - 1, ne - 1), dtype=np.complex128)

    # Alpha-alpha
    for (i, j, sign) in h1e_alpha_info:
        ia = det1.alpha[i]
        ja = det2.alpha[j]
        copy_excluding(ovmat, comat, np.array([j], dtype=np.int64), np.array([i], dtype=np.int64))
        h1e += sign * h1emo[ja, ia] * compute_det(comat)

    # Beta-beta
    for (i, j, sign) in h1e_beta_info:
        ib = det1.beta[i]
        jb = det2.beta[j]
        copy_excluding(ovmat, comat,
                      np.array([j + det2.nalpha], dtype=np.int64),
                      np.array([i + det1.nalpha], dtype=np.int64))
        h1e += sign * h1emo[jb, ib] * compute_det(comat)

    return ov, h1e, ovmat

# -------------------- 7. Compute Two-Electron Term --------------------
@njit(fastmath=True)
def compute_r12(r12mo, r12mo_antisym, ovmat, det1, det2, alpha_alpha_info, alpha_beta_info, beta_beta_info):
    ne = det1.nalpha + det1.nbeta
    r12 = 0.0 + 0.0j
    comat2 = np.zeros((ne - 2, ne - 2), dtype=np.complex128)
#    return r12

    # Alpha-alpha
    for idx, (i, k, j, l, sign) in enumerate(alpha_alpha_info):
        ia = det1.alpha[i]
        ka = det1.alpha[k]
        ja = det2.alpha[j]
        la = det2.alpha[l]
        r12int = r12mo_antisym[la, ka, ja, ia]
        copy_excluding(ovmat, comat2,
                      np.array([j, l], dtype=np.int64),
                      np.array([i, k], dtype=np.int64))
        r12 += sign * r12int * compute_det(comat2)

    # Alpha-beta
    for idx, (i, k, j, l, sign) in enumerate(alpha_beta_info):
        ia = det1.alpha[i]
        kb = det1.beta[k]
        ja = det2.alpha[j]
        lb = det2.beta[l]
        r12int = r12mo[lb, kb, ja, ia]
        copy_excluding(ovmat, comat2,
                      np.array([j, l + det2.nalpha], dtype=np.int64),
                      np.array([i, k + det1.nalpha], dtype=np.int64))
        r12 += sign * r12int * compute_det(comat2)

    # Beta-beta
    for idx, (i, k, j, l, sign) in enumerate(beta_beta_info):
        ib = det1.beta[i]
        kb = det1.beta[k]
        jb = det2.beta[j]
        lb = det2.beta[l]
        r12int = r12mo_antisym[lb, kb, jb, ib]
        copy_excluding(ovmat, comat2,
                      np.array([j + det2.nalpha, l + det2.nalpha], dtype=np.int64),
                      np.array([i + det1.nalpha, k + det1.nalpha], dtype=np.int64))
        r12 += sign * r12int * compute_det(comat2)

    return r12

# -------------------- 8. Main Lowdin Function --------------------
@njit
def lowdin(ovmo, h1emo, r12mo, r12mo_antisym, det1, det2):
    # Precompute minor info and zero masks
    alpha_alpha_info, alpha_beta_info, beta_beta_info, h1e_alpha_info, h1e_beta_info = precompute_minor_info(det1, det2)

    # Compute overlap and one-electron term
    ov, h1e, ovmat = compute_ov_and_h1e(ovmo, h1emo, det1, det2, h1e_alpha_info, h1e_beta_info)

    # Compute two-electron term
    ne = det1.nalpha + det1.nbeta
    if ne > 1:
      r12 = compute_r12(r12mo, r12mo_antisym, ovmat, det1, det2, alpha_alpha_info, alpha_beta_info, beta_beta_info)
    return ov, h1e, r12
