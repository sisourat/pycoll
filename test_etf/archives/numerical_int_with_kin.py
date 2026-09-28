from pyscf import gto, dft, scf
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define separate molecules for H and F ---
mol_H = gto.M(atom='H 0 0 0', basis='6-31g', charge=1)
mol_F = gto.M(atom='F 0 0 0', basis='6-31g', spin=1)

# --- Step 2: Perform SCF calculations for H and F separately ---
mf_H = scf.RHF(mol_H)
mf_H.kernel()
print(mf_H.mo_energy)
mf_F = scf.RHF(mol_F)
mf_F.kernel()
print(mf_F.mo_energy)

# Extract molecular orbitals (MOs) for H and F
mo_H = mf_H.mo_coeff  # Shape: (nao_H, nmo_H)
mo_F = mf_F.mo_coeff  # Shape: (nao_F, nmo_F)

# --- Step 3: Define the full molecule (HF) ---
z_positions = np.linspace(-2.0, 2.0, 40)  # 10 positions from 0.5 to 2.0 Å
hcore_modified_list = []  # Store hcore_modified for each position

# --- Step 4: Define the translation vector v ---
vz = 0.1  # Magnitude of v along z-axis
v = np.array([0.0, 0.0, vz])

# --- Step 5: Loop over F positions ---
for z_pos in z_positions:
    # Redefine the full molecule with F at (0, 0, 0) and H+ at (0, 0, z_pos)
    mol = gto.M(
        atom=f'H 0 0 {z_pos}; F 0 0 0',
        basis='6-31g'
    )

    # --- Step 6: Build the basis of MOs from H and F ---
    # Get the number of AOs for H and F in the full molecule
    nao_H = mol_H.nao_nr()
    nao_F = mol_F.nao_nr()
    nao = mol.nao_nr()

    # Get the overlap matrix between the full molecule and the fragments
    # We need to map the AOs of H and F in the full molecule to their respective MOs
    # This is non-trivial, so we assume the first nao_H AOs correspond to H and the rest to F
    # This is a simplification and may not be accurate for all basis sets
    # A more rigorous approach would use projection or localization methods
    S = mol.intor('int1e_ovlp')

    # Split the overlap matrix into blocks for H and F
    S_HH = S[:nao_H, :nao_H]
    S_FF = S[nao_H:, nao_H:]
    S_HF = S[:nao_H, nao_H:]

    # Transform the MOs of H and F to the basis of the full molecule
    # This is a simplified approach; a more accurate method would use projection
    mo_H_full = np.zeros((nao, mo_H.shape[1]))
    mo_H_full[:nao_H, :] = mo_H
    mo_F_full = np.zeros((nao, mo_F.shape[1]))
    mo_F_full[nao_H:, :] = mo_F

    # Combine the MOs of H and F to form the basis for the full molecule
    mo_basis = np.hstack((mo_H_full, mo_F_full))  # Shape: (nao, nmo_H + nmo_F)

    # --- Step 7: Generate the DFT grid ---
    grid = dft.gen_grid.Grids(mol)
    grid.level = 3  # Higher grid level for accuracy
    grid.build()

    # Get grid points and weights
    coords = grid.coords  # Shape: (N, 3)
    weights = grid.weights  # Shape: (N,)

    # --- Step 8: Evaluate basis functions and gradients on the grid ---
    nbas = mol.nbas  # Number of shells

    # Evaluate all basis functions on the grid
    ao_value = dft.numint.eval_ao(mol, coords, non0tab=None, shls_slice=(0, nbas))
    ao_value = ao_value.reshape(-1, nao)  # Shape: (N, nao)

    # Evaluate gradients of all basis functions on the grid
    ao_grad = dft.numint.eval_ao(mol, coords, non0tab=None, shls_slice=(0, nbas), deriv=1)
    ao_grad = ao_grad.reshape(-1, 3, nao)  # Shape: (N, 3, nao)

    # Ensure ao_value and ao_grad have the same number of grid points
    if ao_value.shape[0] != ao_grad.shape[0]:
        min_N = min(ao_value.shape[0], ao_grad.shape[0])
        ao_value = ao_value[:min_N, :]
        ao_grad = ao_grad[:min_N, :, :]
        coords = coords[:min_N, :]
        weights = weights[:min_N]

    # Recompute phase_factor
    phase_factor = np.exp(1j * np.dot(coords, v))  # Shape: (N,)

    # --- Step 9: Compute the standard kinetic energy matrix ---
    T = mol.intor('int1e_kin')  # Standard kinetic energy matrix

    # --- Step 10: Replace V_nuc_grid with 1/r ---
    r = np.linalg.norm(coords, axis=1)  # Shape: (N,)
    V_custom = -1.0 / r  # Shape: (N,)

    # --- Step 11: Compute the modified kinetic energy matrix ---
    momentum_term = np.zeros((nao, nao), dtype=complex)
    for p in range(nao):
        for q in range(nao):
            v_dot_grad_phi_p = np.einsum('ij,j->i', ao_grad[:, :, p], v)  # Shape: (N,)
            integrand = 1j * v_dot_grad_phi_p * ao_value[:, q] * phase_factor
            momentum_term[p, q] = np.sum(integrand * weights)

    # Phase-dependent term: 0.5 v^2 phi_p * phi_q
    phase_term = 0.5 * np.dot(v, v) * np.sum(
        np.conj(ao_value) * ao_value * phase_factor[:, np.newaxis] * weights[:, np.newaxis],
        axis=0
    )

    # Combine all terms for the modified kinetic energy
    T_modified = T + momentum_term + np.diag(phase_term)

    # --- Step 12: Compute the modified potential energy matrix ---
    V_modified = np.zeros((nao, nao), dtype=complex)
    for p in range(nao):
        for q in range(nao):
            integrand = np.conj(ao_value[:, p]) * ao_value[:, q] * V_custom * phase_factor
            V_modified[p, q] = np.sum(integrand * weights)

    # --- Step 13: Combine into the modified hcore matrix ---
    #hcore_modified = T_modified + V_modified
    hcore_modified = V_modified

    # Transform hcore_modified to the MO basis
    hcore_modified_mo = np.dot(mo_basis.T, np.dot(hcore_modified, mo_basis))
    hcore_modified_list.append(hcore_modified_mo.real)  # Store the real part

# --- Step 14: Plot the results ---
hcore_modified_array = np.array(hcore_modified_list)  # Shape: (10, nmo_H + nmo_F, nmo_H + nmo_F)

# Example: Plot the diagonal elements of hcore_modified in the MO basis
plt.figure(figsize=(10, 6))
for i in range(min(5, hcore_modified_array.shape[1])):  # Plot first 5 diagonal elements
    plt.plot(z_positions, hcore_modified_array[:, i, i], label=f'Diagonal Element {i+1}')

plt.xlabel('Position of F (Å)')
plt.ylabel('hcore_modified (a.u.)')
plt.title('Diagonal Elements of hcore_modified in MO Basis vs. F Position')
plt.legend()
plt.grid(True)
plt.savefig('hcore_modified_mo_vs_F_position.png', dpi=200, bbox_inches='tight')
plt.close()

print("Plot saved as 'hcore_modified_mo_vs_F_position.png'")
