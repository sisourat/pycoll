import sys
import numpy as np

from pyscf import gto, scf, ao2mo, lo, dft
from pyscf import lib
from scipy.linalg import eigh
from scipy.linalg import orth
from scipy.sparse.csgraph import connected_components

def find_blocks(matrix, threshold=1e-10):
    # Create a graph where non-zero elements indicate connections
    n = matrix.shape[0]
    adjacency = np.abs(matrix) > threshold

    # Find connected components (blocks)
    num_blocks, labels = connected_components(adjacency, directed=False)

    # Group indices by block
    blocks = []
    for block_id in range(num_blocks):
        block_indices = np.where(labels == block_id)[0]
        blocks.append(block_indices)

    return blocks

# Define and compute the model potential
def model_potential(alp, coef, center, power, grid):
    pot = np.zeros_like(grid[:, 0])  # Initialize potential array
    #print(alp,coef,center,power)
    for a, c, r, p in zip(alp, coef, center, power):
        dist = np.linalg.norm(grid - r, axis=1)  # Distance from each grid point to center
        pot += c * np.exp(-a * dist) * (dist + 1e-14)**p  # Avoid division by zero
    return pot

def compute_model_potential(mol, alp, coef, center, power):
    grid = dft.gen_grid.Grids(mol)
    grid.prune = None  # Disable pruning
    grid.level = 9
    #grid.radi_method = dft.delley
    grid.build()
    #for l in grid.coords:
    #    print(l)
    ao_value = dft.numint.eval_ao(mol, grid.coords)
    modpot = model_potential(alp, coef, center, power, grid.coords)
    nao = mol.nao_nr()
    modpot_matrix = np.zeros((nao, nao), dtype=float)
    for i in range(nao):
        for j in range(nao):
            integrand = ao_value[:, i] * ao_value[:, j] * modpot
            modpot_matrix[i, j] = np.dot(grid.weights, integrand)
    #print(modpot_matrix)
    return modpot_matrix

def modpot_system(mol, alp, coef, center, power, debug=False):
  '''
    Computes target or projectile model potential orbitals
  '''

  ao_labels = mol.ao_labels()

  # Compute the one-electron integrals
  T = mol.intor('int1e_kin')  # Kinetic energy
  #V_nuc = mol.intor('int1e_nuc')  # Nuclear attraction
  V_mod = compute_model_potential(mol, alp, coef, center, power)  # Nuclear attraction
  S = mol.intor('int1e_ovlp')   # Overlap matrix
  #F = T + V_nuc + V_mod
  V_mod[np.abs(V_mod) < 1e-12] = 0.00
  #print(V_mod)
  F = T + V_mod
  blocks = find_blocks(F)

  # Initialize eigenvectors as a full zero matrix
  n = F.shape[0]
  mo_coeff = np.zeros((n, n))
  mo_energies = np.zeros(n)
  # Initialize lists to store eigenvalues and eigenvectors
  all_eigenvalues = []
  all_eigenvectors = [] # np.eye(F.shape[0])  # Identity matrix to store eigenvectors

  # Diagonalize each block
  for block_indices in blocks:
    # Extract the block
    F_block = F[np.ix_(block_indices, block_indices)]
    S_block = S[np.ix_(block_indices, block_indices)]

    # Diagonalize the block
    eigvals, eigvecs = eigh(F_block, S_block)

    # Store eigenvalues and eigenvectors
    all_eigenvalues.extend(eigvals)
    all_eigenvectors.extend([(block_indices, eigvecs[:, i]) for i in range(eigvecs.shape[1])])

  #print(all_eigenvalues)
  #print(all_eigenvectors)

  # Convert eigenvalues to a 1D numpy array
  all_eigenvalues = np.array(all_eigenvalues)
  for i, e in enumerate(all_eigenvalues):
    pos, coeff = all_eigenvectors[i]
    mo_coeff[pos,i] = coeff
    mo_energies[i] = e

  sorted_indices = np.argsort(mo_energies)
  mo = mo_coeff[:, sorted_indices]
  mo_e = mo_energies[sorted_indices]
  #mo = mo_coeff
  #mo_e = mo_energies

  # Print results
  print("Orbital energies (Hartree):")
  for i, energy in enumerate(mo_e):
      print(f"MO {i}: {energy:.6f}")
      print()
      for j in range(len(mo)):
        print(j,mo[j,i],ao_labels[j])
      print()

  return mo, mo_e

def modpot_erf_system(mol, alp, coef, center, debug=False):
  '''
    Computes target or projectile orbitals for -q/r + ne_core*erf(r)/r
  '''

  ao_labels = mol.ao_labels()

  # Compute the one-electron integrals
  T = mol.intor('int1e_kin')  # Kinetic energy
  S = mol.intor('int1e_ovlp')   # Overlap matrix

  pot = 0.0
  for i, a in enumerate(alp):
    with mol.with_rinv_zeta(zeta=a), mol.with_rinv_origin(center[i]):
      pot += coef[i]*mol.intor('int1e_rinv')

  F = T + pot
  blocks = find_blocks(F)

  # Initialize eigenvectors as a full zero matrix
  n = F.shape[0]
  mo_coeff = np.zeros((n, n))
  mo_energies = np.zeros(n)
  # Initialize lists to store eigenvalues and eigenvectors
  all_eigenvalues = []
  all_eigenvectors = [] # np.eye(F.shape[0])  # Identity matrix to store eigenvectors

  # Diagonalize each block
  for block_indices in blocks:
    # Extract the block
    F_block = F[np.ix_(block_indices, block_indices)]
    S_block = S[np.ix_(block_indices, block_indices)]

    # Diagonalize the block
    eigvals, eigvecs = eigh(F_block, S_block)

    # Store eigenvalues and eigenvectors
    all_eigenvalues.extend(eigvals)
    all_eigenvectors.extend([(block_indices, eigvecs[:, i]) for i in range(eigvecs.shape[1])])

  #print(all_eigenvalues)
  #print(all_eigenvectors)

  # Convert eigenvalues to a 1D numpy array
  all_eigenvalues = np.array(all_eigenvalues)
  for i, e in enumerate(all_eigenvalues):
    pos, coeff = all_eigenvectors[i]
    mo_coeff[pos,i] = coeff
    mo_energies[i] = e

  sorted_indices = np.argsort(mo_energies)
  mo = mo_coeff[:, sorted_indices]
  mo_e = mo_energies[sorted_indices]
  #mo = mo_coeff
  #mo_e = mo_energies

  # Print results
  print("Orbital energies (Hartree):")
  for i, energy in enumerate(mo_e):
      print(f"MO {i}: {energy:.6f}")
      print()
      for j in range(len(mo)):
        print(j,mo[j,i],ao_labels[j])
      print()

  return mo, mo_e


def hcore(mol, mo):
   """ Computes the one-electron terms for given mol and mo """
   T = mol.intor('int1e_ovlp')
   ovl = mo.T @ T @ mo
   T = mol.intor('int1e_kin')
   kin = mo.T @ T @ mo
   T = mol.intor('int1e_nuc')
   pot = mo.T @ T @ mo
   return ovl, kin, pot

def hcore_modpot(alp, coef, center, power, mol, mo):
   """ Computes the one-electron terms for given mol and mo """
   T = mol.intor('int1e_ovlp')
   ovl = mo.T @ T @ mo
   T = mol.intor('int1e_kin')
   kin = mo.T @ T @ mo
   T = compute_model_potential(mol, alp, coef, center, power)
   pot = mo.T @ T @ mo
   return ovl, kin, pot

def hcore_modpot_erf(alp, coef, center, mol, mo):
   """ Computes the one-electron terms for given mol and mo """
   # Compute the one-electron integrals
   T = mol.intor('int1e_ovlp')   # Overlap matrix
   ovl = mo.T @ T @ mo
   T = mol.intor('int1e_kin')  # Kinetic energy
   kin = mo.T @ T @ mo

   T = 0.0
   for i, a in enumerate(alp):
     with mol.with_rinv_zeta(zeta=a), mol.with_rinv_origin(center[i]):
       T += coef[i]*mol.intor('int1e_rinv')
   pot = mo.T @ T @ mo

   return ovl, kin, pot


def twoeints(mol,mo):
    """ Compute the two electron integrals in MO basis """

# saves the two-electron integrals in the file ftmp.name
    ao2mo.kernel(mol, mo, erifile = 'hf.h5', dataname = 'test')
# load 2e integrals by filename and dataname
    with ao2mo.load('hf.h5', 'test') as eri:
      erimo = ao2mo.restore(1, np.asarray(eri), mo.shape[1])

    return erimo

def system(mol,debug=False):
  '''
    Computes target or projectile HF orbitals
  '''
  conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))
  nmo = len(mo_e)

  #mf = scf.RHF(mol).run()
  #lomo = lo.orth_ao(mf, 'nao')

  print()
  print(e)
  print()
  for i in range(len(mo_e)):
     print(i,mo_e[i])
  print()

  #return lomo, mo_e
  return mo, mo_e
