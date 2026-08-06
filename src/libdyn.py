import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Define the RHS function with hmat_interp as an argument
def rhs(t, psi, hmat_interp):
     """Right-hand side of the TDSE with hmat_interp as an argument."""
     hmat = hmat_interp(t)  # Interpolate H(t) at time t
     #print(t,hmat[0,0].real)
     return -1j * np.dot(hmat, psi)

def solve_tdse(hmat_interp, psi0, t_grid):

 sol = solve_ivp(
    lambda t, psi: rhs(t, psi, hmat_interp),  # Pass hmat_interp as an argument
    (t_grid[0], t_grid[-1]),
    psi0,
    t_eval=t_grid,
    method='DOP853',
    rtol=1e-6,
    atol=1e-8,
    vectorized=False  # Ensure the function is not treated as vectorized
    )

 # Extract the solution
 return sol.y.T

def solve_tdse_sequential(hmat_interp, psi0, t_grid, dt_max=0.1):
    """
    Solve the TDSE in smaller time steps and renormalize after each segment.

    Args:
        hmat_interp: Interpolated Hamiltonian function.
        psi0: Initial wavefunction (complex array).
        t_grid: Time grid for the solution.
        dt_max: Maximum time step for each segment.

    Returns:
        psi_t: Wavefunction at each time in `t_grid`, normalized to 1.
    """
    psi_t = [psi0.copy()]
    for i in range(1, len(t_grid)):
        t_start = t_grid[i-1]
        t_end = t_grid[i]

        # Solve for this segment
        sol = solve_ivp(
            lambda t, psi: rhs(t, psi, hmat_interp),
            (t_start, t_end),
            psi_t[-1],
            t_eval=[t_start, t_end],
            method='DOP853',
            rtol=1e-6,
            atol=1e-8,
            vectorized=False
        )

        # Append and renormalize
        psi_segment = sol.y.T
        psi_segment = psi_segment / np.linalg.norm(psi_segment, axis=1)[:, np.newaxis]
        psi_t.append(psi_segment[-1])

    return np.array(psi_t)

