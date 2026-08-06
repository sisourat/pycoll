
import numpy as np
from scipy.special import hyp1f1
from pyscf import gto

# =============================================================================
# Gaussian Product Rule (GPR)
# =============================================================================

def gaussian_product_center(alpha, A, beta, B):
    """
    Compute the new center and exponent for the product of two GTOs.
    """
    gamma = alpha + beta
    P = [
        (alpha * A[0] + beta * B[0]) / gamma,
        (alpha * A[1] + beta * B[1]) / gamma,
        (alpha * A[2] + beta * B[2]) / gamma,
    ]
    return gamma, P

def gaussian_product_normalization(alpha, A, beta, B):
    """
    Compute the normalization factor for the product of two GTOs.
    """
    gamma = alpha + beta
    AB_squared = np.sum([(A[i] - B[i]) ** 2 for i in range(3)])
    return np.exp(-alpha * beta * AB_squared / gamma)

# =============================================================================
# Helper Functions for Integrals (Eqs. 10-14)
# =============================================================================

def compute_S_lmn(l, m, n, alpha, vx, vy, vz):
    """
    Compute the overlap integral S_{lmn} using recursion (Eq. 11).
    """
    if l == 0 and m == 0 and n == 0:
        return (np.pi / alpha) ** (3/2) * np.exp(-(vx**2 + vy**2 + vz**2) / (4 * alpha))

    def S_recursive(l, m, n):
        if l < 0 or m < 0 or n < 0:
            return 0.0
        if l == 0 and m == 0 and n == 0:
            return (np.pi / alpha) ** (3/2) * np.exp(-(vx**2 + vy**2 + vz**2) / (4 * alpha))

        # Recursion for l
        if l >= 1:
            term1 = -((l - 1) / (2 * alpha)) * S_recursive(l - 2, m, n)
            term2 = -(vx / (2 * alpha)) * S_recursive(l - 1, m, n)
            return term1 + term2
        # Recursion for m
        elif m >= 1:
            term1 = -((m - 1) / (2 * alpha)) * S_recursive(l, m - 2, n)
            term2 = -(vy / (2 * alpha)) * S_recursive(l, m - 1, n)
            return term1 + term2
        # Recursion for n
        elif n >= 1:
            term1 = -((n - 1) / (2 * alpha)) * S_recursive(l, m, n - 2)
            term2 = -(vz / (2 * alpha)) * S_recursive(l, m, n - 1)
            return term1 + term2

    return S_recursive(l, m, n)

def compute_J0(alpha, beta, Ri, a, gamma):
    """
    Compute J^{(0)} (Eq. 13) for the potential integral.
    """
    A_sq = -np.sum([a_i**2 for a_i in a]) + 4 * beta * np.sum([Ri_i**2 for Ri_i in Ri]) + 4j * beta * np.dot(a, Ri)
    B_sq = -np.sum([a_i**2 for a_i in a]) + 4 * alpha * np.sum([Ri_i**2 for Ri_i in Ri]) + 4j * alpha * np.dot(a, Ri)
    M0 = hyp1f1(0.5, 1.5, -B_sq / (4 * gamma))
    return (2 * np.pi / gamma) * np.exp(-alpha * np.sum([Ri_i**2 for Ri_i in Ri]) + A_sq / (4 * gamma)) * M0

def compute_V_lmn_k(l, m, n, k, alpha, beta, Ri, vx, vy, vz):
    """
    Compute the potential integral V_{lmn}^{(k)} using recursion (Eq. 14).
    """
    gamma = alpha + beta
    a = [0.0, 0.0, 0.0]  # Placeholder for vector 'a' (Eq. 13)

    if l == 0 and m == 0 and n == 0 and k == 0:
        return compute_J0(alpha, beta, Ri, a, gamma)

    def V_recursive(l, m, n, k):
        if l < 0 or m < 0 or n < 0 or k < 0:
            return 0.0

        # Term 1: (l-1)/(2γ) * V_{(l-2)mn}^{(k)}
        term1 = ((l - 1) / (2 * gamma)) * V_recursive(l - 2, m, n, k)

        # Term 2: (i/(2γ)) * (vx + 2iβRs) * V_{(l-1)mn}^{(k)}
        Rs = Ri[0]  # Assuming Ri is [Rx, Ry, Rz]
        term2 = (1j / (2 * gamma)) * (vx + 2j * beta * Rs) * V_recursive(l - 1, m, n, k)

        # Term 3: -(l-1)/(2γ) * V_{(l-2)mn}^{(k+1)}
        term3 = -((l - 1) / (2 * gamma)) * V_recursive(l - 2, m, n, k + 1)

        # Term 4: (i/(2γ)) * (-vx + 2iαRs) * V_{(l-1)mn}^{(k+1)}
        term4 = (1j / (2 * gamma)) * (-vx + 2j * alpha * Rs) * V_recursive(l - 1, m, n, k + 1)

        return term1 + term2 + term3 + term4

    return V_recursive(l, m, n, k)

# =============================================================================
# Extract Primitive GTOs (PGTOs) from PySCF Basis
# =============================================================================

def get_primitive_gto_info(mol):
    """
    Extract primitive GTO exponents, coefficients, and centers from a PySCF Mole object.

    Args:
        mol: PySCF Mole object.

    Returns:
        list: List of tuples (exponent, coefficient, center) for each primitive GTO.
    """
    primitive_info = []
    atom_coords = mol.atom_coords()  # Call the method to get atom coordinates

    for i in range(mol.nbas):
        # Get basis information for shell i
        bas = mol._basis['He']  # Use _bas instead of bas
        print(bas)
        atom_idx = mol.bas_atom(i)  # Call bas_atom() as a method to get the atom index for shell i
        center = atom_coords[atom_idx]  # Center of the shell

        # Loop over the primitives in the shell
        print(mol.bas_nctr(i))
        for j in range(len(bas[0])):
            exponent = bas[0][j]  # Exponent for the primitive
            coeff = bas[1][j]    # Coefficient for the primitive

            # For simplicity, assume s-type orbitals (l=0, m=0, n=0)
            # In practice, you would need to handle higher angular momentum (l, m, n)
            primitive_info.append((exponent, coeff, center))

    return primitive_info

# =============================================================================
# Compute Overlap and Coupling Matrices for CGTOs
# =============================================================================

def compute_coupling_matrices_cgto(target, projectile, vx=0.1, vy=0.0, vz=0.0):
    """
    Compute the overlap (S) and coupling (M) matrices for the two-center system using CGTOs.
    """
    # Extract primitive GTO information
    target_primitives = get_primitive_gto_info(target)
    projectile_primitives = get_primitive_gto_info(projectile)

    # Number of basis functions (CGTOs)
    n_target = target.nao
    n_projectile = projectile.nao
    n_total = n_target + n_projectile

    # Initialize matrices
    S_matrix = np.zeros((n_total, n_total), dtype=complex)
    M_matrix = np.zeros((n_total, n_total), dtype=complex)

    # Fill overlap matrix (S) for CGTOs
    for i in range(n_target):
        for j in range(n_projectile):
            # Target-projectile overlap (two-center)
            # Sum over all primitive pairs
            S_ij = 0.0
            for (alpha_i, coeff_i, A) in target_primitives:
                for (alpha_j, coeff_j, B) in projectile_primitives:
                    # Apply Gaussian Product Rule (GPR)
                    gamma, P = gaussian_product_center(alpha_i, A, alpha_j, B)
                    K = gaussian_product_normalization(alpha_i, A, alpha_j, B)

                    # Compute S_{lmn} for the combined GTO
                    S_primitive = K * compute_S_lmn(0, 0, 0, gamma, vx, vy, vz)

                    # Multiply by coefficients and add to the total
                    S_ij += coeff_i * coeff_j * S_primitive

            S_matrix[i, j + n_target] = S_ij

    # Fill coupling matrix (M) for CGTOs
    for i in range(n_target):
        for j in range(n_projectile):
            # Target-projectile coupling
            # Sum over all primitive pairs
            M_ij = 0.0
            for (alpha_i, coeff_i, A) in target_primitives:
                for (alpha_j, coeff_j, B) in projectile_primitives:
                    # Apply Gaussian Product Rule (GPR)
                    gamma, P = gaussian_product_center(alpha_i, A, alpha_j, B)
                    K = gaussian_product_normalization(alpha_i, A, alpha_j, B)

                    # Position of the nucleus (Ri) in the new frame
                    Ri = [B[0] - P[0], B[1] - P[1], B[2] - P[2]]

                    # Compute V_{lmn}^{(k)} for the combined GTO
                    V_primitive = K * compute_V_lmn_k(0, 0, 0, 0, alpha_i, alpha_j, Ri, vx, vy, vz)

                    # Multiply by coefficients and add to the total
                    M_ij += coeff_i * coeff_j * V_primitive

            M_matrix[i, j + n_target] = M_ij

    return S_matrix, M_matrix

def setup_collision_system():
    """
    Define the target (H) and projectile (Li3+) using PySCF.
    """
    # Target: H (1s)
    target = gto.Mole()
    target.atom = [['He', (0.0, 0.0, 0.0)]]
    target.basis = {'He': 'sto-3g'}  # Minimal basis for testing
    target.build()

    # Projectile: Li3+ (no electrons, but we define a basis for the electron)
    projectile = gto.Mole()
    projectile.atom = [['He', (0.0, 0.0, 5.0)]]  # 5.0 Bohr separation (adjustable)
    projectile.basis = {'He': 'sto-3g'}  # Minimal basis for testing
    projectile.build()

    return target, projectile

# =============================================================================
# Main Workflow
# =============================================================================

def main():
    # Step 1: Set up the collision system
    target, projectile = setup_collision_system()
    print("Target (H) basis functions:", target.nao)
    print("Projectile (Li3+) basis functions:", projectile.nao)

    # Step 2: Compute coupling matrices for CGTOs
    S_matrix, M_matrix = compute_coupling_matrices_cgto(target, projectile, vx=0.1)
    print("\nOverlap Matrix (S) for CGTOs:")
    print(S_matrix)
    print("\nCoupling Matrix (M) for CGTOs:")
    print(M_matrix)

    return S_matrix, M_matrix

if __name__ == "__main__":
    S, M = main()
