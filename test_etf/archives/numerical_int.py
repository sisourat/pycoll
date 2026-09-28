from pyscf import gto, dft, scf
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Define separate molecules for H and He ---
mol_H = gto.M(atom='H 0 0 0', basis='aug-cc-pvdz', charge=1)
mol_He = gto.M(atom='He 0 0 0', basis='aug-cc-pvdz', spin=0)

# --- Step 2: Perform SCF calculations for H and He separately ---
mf_H = scf.RHF(mol_H)
mf_H.kernel()
print("H MO energies:", mf_H.mo_energy)

mf_He = scf.RHF(mol_He)
mf_He.kernel()
print("He MO energies:", mf_He.mo_energy)

# Extract molecular orbitals (MOs) for H and He
mo_H = mf_H.mo_coeff  # Shape: (nao_H, nmo_H)
mo_He = mf_He.mo_coeff  # Shape: (nao_He, nmo_He)

# --- Step 3: Define the full molecule (H-He) ---
z_positions = np.linspace(-10.0, 10.0, 150)  # Positions from -10 to 10 Å
hcore_modified_list = []  # Store hcore_modified for each position
hcore_modified_linear_list = []  # Store hcore_modified for each position

# --- Step 4: Define the translation vector v ---
vz = 0.6  # Magnitude of v along z-axis
v = np.array([0.0, 0.0, vz])

# --- Step 5: Define ETF approximation functions ---
def exact_etf(coords, v):
    """Exact ETF: exp(i v.r)"""
    return np.exp(1j * np.dot(coords, v))

def taylor_etf(coords, v, order=1):
    """Taylor expansion of ETF up to a given order"""
    z = coords[:, 2]  # Assuming v is along z-axis
    phase_factor = np.zeros_like(z, dtype=complex)
    for n in range(order + 1):
        phase_factor += (1j * v[2] * z) ** n / np.math.factorial(n)
    return phase_factor

def linear_etf(coords, v):
    """Linear approximation of ETF: 1 + i v.z"""
    z = coords[:, 2]
    return 1.0 + 1j * v[2] * z

def pade_etf(coords, v, zp):
    """Padé [1,1] approximation of ETF"""
    #zp = zp/4.0
    z = coords[:, 2]  # Assuming v is along z-axis
    v_dot_r = v[2] * z
    v_dot_zp = v[2] * zp
    numerator = 1.0 + 1j * v_dot_zp / 3.0
    denominator = 1.0 - 1j * 2.0 * v_dot_zp / 3.0 - v_dot_zp**2 / 6.0
    #return  numerator / denominator
    return np.exp(1j * v[2] * zp) #* numerator / denominator

def pade_etf_analytical(mol, v, pot, order=1):
    """
    Compute the modified hcore matrix using the [1,1] Padé approximation for ETF.
    The Padé approximant is expanded as a geometric series up to a given order.
    """
    # Standard integrals
    S = mol.intor('int1e_ovlp')  # Overlap integrals
    dipole_z = mol.intor('int1e_r')[2]  # Dipole integrals (z-component)

    # Compute the standard hcore matrix
    hcore = pot

    # Padé approximant: (1 + i v z / 2) / (1 - i v z / 2)
    # Expand the denominator as a geometric series: 1 / (1 - x) ≈ 1 + x + x^2 + ... (for |x| < 1)
    # Here, x = i v z / 2
    # We approximate the Padé approximant as (1 + i v z / 2) * (1 + i v z / 2 + (i v z / 2)^2 + ...)

    # First-order term: (1 + i v z / 2) * (1 + i v z / 2) ≈ 1 + i v z + (i v z / 2)^2
    # We only keep terms up to first order in v for simplicity
    # Higher-order terms can be included for better accuracy

    # First-order term: i v z (from the numerator and denominator)
    # This is the same as the linear approximation, but with a scaling factor
    # For the [1,1] Padé approximant, the first-order term is i v z
    # So we can reuse the linear approximation logic but with a scaling factor

    # For simplicity, we use the first-order term of the Padé approximant
    # This is equivalent to the linear approximation but with a scaling factor
    # A more rigorous approach would include higher-order terms
    additional_term = 1j * v[2] * dipole_z

    # Combine the standard hcore and the additional term
    hcore_modified = hcore + additional_term

    return hcore_modified

# Choose the ETF approximation function
etf_function = exact_etf  # Change this to test different approximations
#etf_function_taylor = taylor_etf  # Example: Taylor expansion up to order 1
#etf_function_linear = linear_etf   # Example: Linear approximation
etf_function_linear = pade_etf   # Example: Linear approximation

# --- Step 6: Loop over H positions ---
for z_pos in z_positions:
    # Redefine the full molecule with H at (0, 0, z_pos) and He at (0, 0, 0)
    mol = gto.M(
        atom=f'H 0 0 {z_pos}; He 0 0 0',
        basis='aug-cc-pvdz',
        charge=1
    )

    # --- Step 7: Build the basis of MOs from H and He ---
    nao_H = mol_H.nao_nr()
    nao_He = mol_He.nao_nr()
    nao = mol.nao_nr()

    # Simplified approach: Assume first nao_H AOs correspond to H, rest to He
    mo_H_full = np.zeros((nao, mo_H.shape[1]))
    mo_H_full[:nao_H, :] = mo_H
    mo_He_full = np.zeros((nao, mo_He.shape[1]))
    mo_He_full[nao_H:, :] = mo_He

    # Combine the MOs of H and He to form the basis for the full molecule
    mo_basis = np.hstack((mo_H_full, mo_He_full))  # Shape: (nao, nmo_H + nmo_He)

    # --- Step 8: Generate the DFT grid ---
    grid = dft.gen_grid.Grids(mol)
    grid.level = 3  # Higher grid level for accuracy
    grid.build()

    # Get grid points and weights
    coords = grid.coords  # Shape: (N, 3)
    weights = grid.weights  # Shape: (N,)

    # --- Step 9: Evaluate basis functions on the grid ---
    nbas = mol.nbas  # Number of shells
    ao_value = dft.numint.eval_ao(mol, coords, non0tab=None, shls_slice=(0, nbas))
    ao_value = ao_value.reshape(-1, nao)  # Shape: (N, nao)

    # --- Step 10: Compute the ETF phase factor ---
    dipole_z = mol.intor('int1e_r')[2]  # Dipole integrals (z-component)
    dipole_z_mo = np.dot(mo_basis.T.conj(), np.dot(dipole_z, mo_basis))
    #print(z_pos,dipole_z_mo[0,9])
    ovl = mol.intor('int1e_ovlp')  # Overlap integrals
    ovlmo = np.dot(mo_basis.T.conj(), np.dot(ovl, mo_basis))
    #zmo = z_pos-dipole_z_mo[1,10]#/ovlmo[1,10]
    zmo = z_pos+ovlmo[1,10]
    #print(z_pos,zmo,dipole_z_mo[1,10],ovlmo[1,10])
    #zmo = z_pos*ovlmo[1,10]
    phase_factor = etf_function(coords, v)  # Use the chosen ETF approximation
    phase_factor_linear = etf_function_linear(coords, v, zmo)  # Use the chosen ETF approximation

    # --- Step 11: Compute V_custom = -1/r ---
    r = np.linalg.norm(coords, axis=1)  # Shape: (N,)
    V_custom = -1.0 / r  # Shape: (N,)

    # --- Step 12: Compute the modified potential energy matrix ---
    V_modified = np.zeros((nao, nao), dtype=complex)
    V_modified_linear = np.zeros((nao, nao), dtype=complex)
    for p in range(nao):
        for q in range(nao):
            integrand = np.conj(ao_value[:, p]) * ao_value[:, q] * V_custom * phase_factor
            V_modified[p, q] = np.sum(integrand * weights)
            integrand = np.conj(ao_value[:, p]) * ao_value[:, q] * V_custom * phase_factor_linear
            V_modified_linear[p, q] = np.sum(integrand * weights)

    #V_modified_linear = pade_etf_analytical(mol, v, V_modified_linear, order=1)
    # --- Step 13: Combine into the modified hcore matrix ---
    hcore_modified = V_modified
    hcore_modified_linear = V_modified_linear


    # Transform hcore_modified to the MO basis
    hcore_modified_mo = np.dot(mo_basis.T.conj(), np.dot(hcore_modified, mo_basis))
    hcore_modified_mo_linear = np.dot(mo_basis.T.conj(), np.dot(hcore_modified_linear, mo_basis))
    hcore_modified_list.append(hcore_modified_mo)
    hcore_modified_linear_list.append(hcore_modified_mo_linear)

# --- Step 14: Plot the results ---
hcore_modified_array = np.array(hcore_modified_list)  # Shape: (150, nmo_H + nmo_He, nmo_H + nmo_He)
hcore_modified_linear_array = np.array(hcore_modified_linear_list)  # Shape: (150, nmo_H + nmo_He, nmo_H + nmo_He)

# Plot the real and imaginary parts of the H1s-He1s coupling
plt.figure(figsize=(10, 6))
#plt.plot(z_positions, hcore_modified_array[:, 0, 0].real, label='Real part H1s-H1s')
#plt.plot(z_positions, hcore_modified_array[:, 0, 0].imag, label='Imag part H1s-H1s')
plt.plot(z_positions, hcore_modified_array[:, 1, 10].real, label='Real part H1s-He1s')
plt.plot(z_positions, hcore_modified_array[:, 1, 10].imag, label='Imag part H1s-He1s')
plt.plot(z_positions, hcore_modified_linear_array[:, 1, 10].real, label='Real part H1s-He1s Linear approx')
plt.plot(z_positions, hcore_modified_linear_array[:, 1, 10].imag, label='Imag part H1s-He1s Linear approx')

plt.xlabel('Position of H (Å)')
plt.ylabel('hcore_modified (a.u.)')
plt.title('hcore_modified in MO Basis vs. H Position (Exact ETF)')
plt.legend()
plt.grid(True)
plt.show()
