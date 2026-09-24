import numpy as np
from scipy.special import gamma
import sys
import os
import pathlib
import logging
from pyscf import gto
from scipy.interpolate import interp1d
import importlib
from libsystems import *
from generate_csfs import *
from fcfactors import *
from libdyn import *

# --- Input Validation ---
def validate_input(inp, required_vars):
    """Validate that the inpui defines all required variables."""
    for var in required_vars:
        assert hasattr(inp, var), f"Input {inp.__name__} must define {var}"

# --- System Setup ---
def setup_molecules(tgeom, tbasis, tcharge, tspin, pgeom, pbasis, pcharge, pspin, orb, debug=False):
    """Set up target, projectile, and super-system molecules."""
    sgeom = tgeom + pgeom
    sbasis = tbasis | pbasis
    scharge = tcharge + pcharge
    sspin = tspin + pspin

    # Build molecules
    tmol = gto.M(atom=tgeom, basis=tbasis, charge=tcharge, spin=tspin, symmetry=True)
    tmol.build(unit="Bohr")
    pmol = gto.M(atom=pgeom, basis=pbasis, charge=pcharge, spin=pspin, symmetry=True)
    pmol.build(unit="Bohr")
    mol = gto.M(atom=sgeom, basis=sbasis, charge=scharge, spin=sspin)
    mol.build(unit="Bohr")

    print("Molecules built successfully.")
    return tmol, pmol, mol, sgeom, sbasis, scharge, sspin

# --- Orbital Calculations ---
def compute_orbitals(mol, orb_type, **kwargs):
    """Compute orbitals based on the specified type (HF, modpot, modpot_erf)."""
    if orb_type == "modpot":
        mo, mo_e = modpot_system(mol, **kwargs)
    elif orb_type == "modpot_erf":
        mo, mo_e = modpot_erf_system(mol, **kwargs)
    elif orb_type == "HF" or "HF_ETF":
        mo, mo_e = system(mol, debug=kwargs.get("debug", False))
    else:
        raise NotImplementedError("Only HF or modpot orbitals are implemented.")
    print(f"Orbitals computed using {orb_type}.")
    return mo, mo_e

# --- Franck-Condon Factors ---

def compute_fcf(morse_p, mass):
    """Compute Franck-Condon factors for all pairs of CSFs."""
    nsta = len(morse_p)
    mat_fcf = [[None for _ in range(nsta)] for _ in range(nsta)]
    evib = [None for _ in range(nsta)]
    for i1, p1 in enumerate(morse_p):
        for i2, p2 in enumerate(morse_p):
           if i2 >= i1:
             evib[i1], mat_fcf[i2][i1] = fcfactors(p1,p2,mass)
             mat_fcf[i1][i2] = np.transpose(mat_fcf[i2][i1])

    return evib, mat_fcf

def print_asymp_eig(eig,eigv,morse_p,evib):

   nsta = len(eig)
   ncsfs = len(eig)

   k = 0
   print()
   for i in range(nsta):
     print('Electronic state ',i , eig[i])
     n_vib_i = morse_p[i][3]
     print('        Vibronic states')
     for j in range(n_vib_i):
         print("       ",k, eig[i].real+evib[i][j], eig[i].real, evib[i][j])
         k+=1
     print()

def print_asymp_elec_eig(eig,eigv,csfs):
    # Get the absolute values of the eigenvectors
    abs_eigv = np.abs(eigv)
    # Find the index of the largest component for each eigenvector (column)
    largest_component_indices = np.argmax(abs_eigv, axis=0)
    # Find the values of the largest components
    largest_component_values = abs_eigv[largest_component_indices, range(eigv.shape[1])]

    nsta = len(eig)
    ncsfs = len(eig)

    k = 0
    for i in range(nsta):
      _, alpe, betae =  csfs[largest_component_indices[i]].terms[0]
      print('Electronic state ',i , eig[i])
      for j in range(ncsfs):
       if(np.abs(eigv[j,i].real)>0.3):
        print("  ",j, eigv[j,i].real, csfs[j])
      print()


def select_sta(fulleig,fulleigv,sta_list):

    # Extract the keys from the dictionary
    selected_states = list(sta_list.keys())
    if len(selected_states) == 0:
        return len(fulleig), fulleig, fulleigv, None

    # Select the corresponding columns from eigv
    eigv = fulleigv[:, selected_states]
    eig = fulleig[selected_states]

    nsta = len(eig)
    morse_p = [[None for i in range(3)] for j in range(nsta)]
    k=0
    for i in selected_states:
      morse_p[k][:] = sta_list[i]
      k+=1

    # prepare the list of Morse parameter
    return nsta, eig, eigv, morse_p


# --- Collision Dynamics ---
def setup_collision_grid(bmin, bmax, nbb, zmax, ngrid, gridtype, vproj):
    """Set up impact parameters and z-grid for collision dynamics."""
    blist = np.linspace(bmin, bmax, nbb)
    if gridtype == "exp":
        zlist = np.sort(
            np.concatenate(
                (
                    -np.logspace(-18, 0, base=2, num=ngrid) * zmax,
                    np.logspace(-18, 0, base=2, num=ngrid) * zmax,
                )
            )
        )
    elif gridtype == "lin":
        zlist = np.linspace(-zmax, zmax, ngrid)
    else:
        raise NotImplementedError("Grid type must be 'exp' or 'lin'.")

    tlist = zlist / vproj
    print(f"Collision grid set up: bmin={bmin}, bmax={bmax}, nb={nbb}, zmax={zmax}, ngrid={ngrid}, gridtype={gridtype}.")
    return blist, zlist, tlist

# --- Vibronic Matrices ---
def build_vibronic_matrices(hmat, smat, mat_fcf, evib, morse_p, vib_indices):
    """Build vibronic Hamiltonian and overlap matrices using vectorization."""
    ncsfs = len(hmat)
    n_vib_total = sum(morse_p[i][3] for i in range(ncsfs))  # Total number of vibronic states
    hmatvib = np.zeros((n_vib_total, n_vib_total), dtype=complex)
    smatvib = np.zeros((n_vib_total, n_vib_total), dtype=complex)

    # Track the starting index for each electronic state's vibrational states
    vib_indices = []
    start = 0
    for i in range(ncsfs):
       n_vib_i = morse_p[i][3]
       vib_indices.append((start, start + n_vib_i))
       start += n_vib_i

    # Fill hmatvib and smatvib
    for i in range(ncsfs):
      for j in range(ncsfs):
        n_vib_i = morse_p[i][3]
        n_vib_j = morse_p[j][3]
        start_i, end_i = vib_indices[i]
        start_j, end_j = vib_indices[j]

        # Extract Franck-Condon factors for this electronic state pair
        fcf = mat_fcf[j][i]  # Shape: (n_vib_j, n_vib_i)

        # Fill the blocks for hmatvib and smatvib
        for k in range(n_vib_i):
           for l in range(n_vib_j):
              ev = 0.25*(evib[i][k]+evib[j][l])*smat[j, i]  # comes from the vib kinetic energy operator + Virial Theorem
              hmatvib[start_j + l, start_i + k] = (hmat[j, i] + ev) * fcf[l, k]
              smatvib[start_j + l, start_i + k] = smat[j, i] * fcf[l, k]

    #print("Vibronic matrices built.")
    return hmatvib, smatvib

# --- TDSE Propagation ---
def run_tdse(hmat_interp, psi0, t_grid):
    """Run TDSE propagation."""
    wf_t = solve_tdse_sequential(hmat_interp, psi0, t_grid)
    prob = np.abs(wf_t[-1, :]) ** 2
    #print(f"TDSE propagation completed. Total probability: {np.sum(prob):.6f}")
    return prob

