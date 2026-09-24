import sys
import numpy as np
import os
import time

from pyscf import gto, scf, ao2mo, lo, dft
from pyscf import lib
from scipy.linalg import eigh
from scipy.linalg import orth
from scipy.sparse.csgraph import connected_components

from numba import jit, njit
from scipy.spatial import cKDTree

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
   T = mol.intor('int1e_ipovlp')[2]
   deriv_z = mo.T @ T @ mo
   return ovl, kin, pot, deriv_z

def precompute_neighbors(coords, k=6):
    tree = cKDTree(coords)
    neighbors = []
    for i in range(len(coords)):
        distances, indices = tree.query(coords[i], k=k+1)  # k+1 to include the point itself
        neighbors.append(indices[1:])  # Exclude the point itself
    return np.array(neighbors)

def precompute_neighbors_index(coords, neighbors):
    n_points = len(coords)
    neighbors_list = []
    for i in range(n_points):
        neighbor_indices = neighbors[i]
        neighbors_list.append(neighbor_indices)
    return neighbors_list

@njit(fastmath=True)
def compute_laplacian(vproj, ntmo, npmo, mo_values, coords, neighbors_list):
    n_points, n_mos = mo_values.shape
    laplacian = np.zeros((n_points, n_mos),dtype=np.complex128)
    mo_values = mo_values.astype(np.complex128)
    #for i in range(n_points):
    #    print(neighbors_list[i])

    for p in range(ntmo):
        for i in range(n_points):
            f_center = mo_values[i, p]
            n_neighbors = len(neighbors_list[i])
            A = np.ones((n_neighbors, 7),dtype=np.complex128)
            b = np.zeros(n_neighbors,dtype=np.complex128)
            k = 0
            for j in neighbors_list[i]:

                dx = coords[j][0] - coords[i][0]
                dy = coords[j][1] - coords[i][1]
                dz = coords[j][2] - coords[i][2]

                A[k, 1] = dx
                A[k, 2] = dy
                A[k, 3] = dz
                A[k, 4] = dx ** 2
                A[k, 5] = dy ** 2
                A[k, 6] = dz ** 2
                b[k] = mo_values[j, p]  # Use the neighbor's MO value
                k += 1

            # Solve the normal equations: coeffs = (A^T A)^-1 A^T b
            ATA = np.dot(A.T, A)
            ATb = np.dot(A.T, b)

            # Add a small regularization term to avoid singularities
            ATA += 1e-8 * np.eye(7)

            coeffs = np.linalg.solve(ATA, ATb)

            # Laplacian = 2*(e + f + g)
            laplacian[i, p] = 2 * (coeffs[4] + coeffs[5] + coeffs[6])

    for p in range(ntmo, n_mos):
        for i in range(n_points):
            n_neighbors = len(neighbors_list[i])
            A = np.ones((n_neighbors, 7),dtype=np.complex128)
            b = np.zeros(n_neighbors,dtype=np.complex128)
            k = 0
            for j in neighbors_list[i]:

                dx = coords[j][0] - coords[i][0]
                dy = coords[j][1] - coords[i][1]
                dz = coords[j][2] - coords[i][2]

                A[k, 1] = dx
                A[k, 2] = dy
                A[k, 3] = dz
                A[k, 4] = dx ** 2
                A[k, 5] = dy ** 2
                A[k, 6] = dz ** 2
                b[k] = mo_values[j, p] * np.exp(1j*vproj*coords[i,2])  # Use the neighbor's MO value
                k += 1

            # Solve the normal equations: coeffs = (A^T A)^-1 A^T b
            ATA = np.dot(A.T, A)
            ATb = np.dot(A.T, b)

            # Add a small regularization term to avoid singularities
            ATA += 1e-8 * np.eye(7)

            coeffs = np.linalg.solve(ATA, ATb)

            # Laplacian = 2*(e + f + g)
            laplacian[i, p] = 2 * (coeffs[4] + coeffs[5] + coeffs[6])

    return laplacian

@njit(fastmath=True)
def ints_ETF(mo_value, neighbors_list, grid, weights, coul_pot, ntmo, npmo, vproj):
   #vproj = 0.0
   nmo = ntmo + npmo
   phase = np.exp(1j*vproj*grid[:,2])
   ovl = np.zeros((nmo, nmo), dtype=np.complex128)
   pot = np.zeros((nmo, nmo), dtype=np.complex128)
   kin = np.zeros((nmo, nmo), dtype=np.complex128)

   laplacian_mo = -0.5*compute_laplacian(vproj, ntmo, npmo, mo_value, grid, neighbors_list)
   laplacian_mo = laplacian_mo.astype(np.complex128)

   for i in range(ntmo):
      for j in range(ntmo,nmo):
         integrand = mo_value[:, i] * mo_value[:, j] * phase
         ovl[i, j] = np.dot(weights, integrand)
         ovl[j, i] = np.conj(ovl[i, j])
         integrand = mo_value[:, i] * mo_value[:, j] * phase * coul_pot
         pot[i, j] = np.dot(weights, integrand)
         pot[j, i] = np.conj(pot[i, j])
         integrand = mo_value[:, i] * laplacian_mo[:, j]
         kin[i, j] = np.dot(weights, integrand)
         integrand = mo_value[:, j] * laplacian_mo[:, i] * phase
         kin[j, i] = np.dot(weights, integrand)
   return ovl, pot, kin


def hcore_ETF(mol, mo, ntmo, npmo, vproj):
   """ Computes the one-electron terms for given mol and mo """
   grid = dft.gen_grid.Grids(mol)
   grid.prune = None  # Disable pruning
   grid.level = 3
   #grid.radi_method = dft.delley
   grid.build()
   ao_value = dft.numint.eval_ao(mol, grid.coords)
   mo_value = np.dot(ao_value, mo)
   neighbors = precompute_neighbors(grid.coords, k=48)
   neighbors_list = precompute_neighbors_index(grid.coords, neighbors)

   # Get nuclear charges (atomic numbers) and positions (in Bohr)
   charges = mol.atom_charges()  # Shape: (N_atoms,)
   positions = mol.atom_coords()  # Shape: (N_atoms, 3)
   # Get grid coordinates (shape: (N_grid, 3))
   grid_coords = grid.coords
   # Reshape positions for broadcasting: (1, N_atoms, 3)
   positions_reshaped = positions.reshape(1, -1, 3)
   # Reshape grid_coords for broadcasting: (N_grid, 1, 3)
   grid_coords_reshaped = grid_coords.reshape(-1, 1, 3)
   # Compute distances: ||grid_coords - positions|| (shape: (N_grid, N_atoms))
   distances = np.linalg.norm(grid_coords_reshaped - positions_reshaped, axis=2)
   # Avoid division by zero (for grid points at nucleus positions)
   distances[distances == 0] = 1e-10  # Small cutoff to prevent singularities
   # Compute Coulomb potential: sum(Z_A / |r - R_A|) (shape: (N_grid,))
   coulomb_potential = -np.sum(charges / distances, axis=1)

   # vproj is along z => thus grid.coords[:,2]
   ovletf, potetf, kinetf = ints_ETF(mo_value, neighbors_list, grid.coords[:,:], grid.weights.astype(np.complex128), coulomb_potential, ntmo, npmo, vproj)

   T = mol.intor('int1e_ovlp')
   ovl = mo.T @ T @ mo
   ovl = ovl.astype(np.complex128)

   T = mol.intor('int1e_kin')
   kin = mo.T @ T @ mo
   kin = kin.astype(np.complex128)

   T = mol.intor('int1e_nuc')
   pot = mo.T @ T @ mo
   pot = pot.astype(np.complex128)

   nmo = ntmo + npmo
   for i in range(ntmo):
      for j in range(ntmo, nmo):
           ovl[i,j] = ovletf[i,j]
           ovl[j,i] = ovletf[j,i]
           pot[i,j] = potetf[i,j]
           pot[j,i] = potetf[j,i]
           kin[i,j] = kinetf[i,j]
           kin[j,i] = kinetf[j,i]

   T = mol.intor('int1e_ipovlp')[2]
   deriv_z = mo.T @ T @ mo
   return ovl, kin, pot, deriv_z


def hcore_modpot(alp, coef, center, power, mol, mo):
   """ Computes the one-electron terms for given mol and mo """
   T = mol.intor('int1e_ovlp')
   ovl = mo.T @ T @ mo
   T = mol.intor('int1e_kin')
   kin = mo.T @ T @ mo
   T = compute_model_potential(mol, alp, coef, center, power)
   pot = mo.T @ T @ mo
   T = mol.intor('int1e_ipovlp', comp=2)
   deriv_z = mo.T @ T @ mo
   return ovl, kin, pot, deriv_z

def hcore_modpot_erf(alp, coef, center, mol, mo):
   """ Computes the one-electron terms for given mol and mo """
   # Compute the one-electron integrals
   T = mol.intor('int1e_ovlp')   # Overlap matrix
   ovl = mo.T @ T @ mo
   T = mol.intor('int1e_kin')  # Kinetic energy
   kin = mo.T @ T @ mo
   T = mol.intor('int1e_ipovlp', comp=2)
   deriv_z = mo.T @ T @ mo

   T = 0.0
   for i, a in enumerate(alp):
     with mol.with_rinv_zeta(zeta=a), mol.with_rinv_origin(center[i]):
       T += coef[i]*mol.intor('int1e_rinv')
   pot = mo.T @ T @ mo

   return ovl, kin, pot, deriv_z


def twoeints(mol,mo):
    """ Compute the two electron integrals in MO basis """

    timestamp = int(time.time() * 1000)  # Milliseconds for more uniqueness
    filename = f'hf_{timestamp}.h5'

    # saves the two-electron integrals in the file ftmp.name
    ao2mo.kernel(mol, mo, erifile = filename, dataname = 'test')
    # load 2e integrals by filename and dataname
    with ao2mo.load(filename, 'test') as eri:
      erimo = ao2mo.restore(1, np.asarray(eri), mo.shape[1])

    # Delete the file after loading the integrals
    if os.path.exists(filename):
        os.remove(filename)

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
