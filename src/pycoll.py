#!/usr/bin/env python3
import numpy as np
import sys
import os
import pathlib
import logging
from pyscf import gto
from scipy.interpolate import interp1d
import importlib
from libpycoll import *
from cimat import *
from datetime import datetime
import shutil
from tee import *
import importlib.util


if __name__ == "__main__":

    # Get the current working directory
    working_dir = pathlib.Path.cwd()

    # Get the input script name from command-line arguments
    input_script = sys.argv[1]
    module_path = working_dir / input_script

    # Construct the full path to the input script
    input_script_path = working_dir / input_script
    copied_script_path = input_script_path.with_suffix('.py')
    # Copy the input script to a .py file in the working directory
    shutil.copy(input_script_path, copied_script_path)

    # Import the module from the working directory
    module_name = input_script_path.stem
    inp = importlib.import_module(module_name)
    # Copy all names from the module into the global namespace
    globals().update(vars(inp))

    # Create a directory name with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"{sys.argv[1]}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    input_script_copy = os.path.join(run_dir, input_script+'.py')
    shutil.copy(input_script+'.py', input_script_copy)

    # Define the output file for printed messages
    output_file = os.path.join(run_dir, "output.log")

    # Redirect stdout to the Tee object
    sys.stdout = Tee(output_file)

    print(f"Created directory: {run_dir}")

    # --- Input Validation ---
    required_vars = [
        "tgeom", "tbasis", "tcharge", "tspin",
        "pgeom", "pbasis", "pcharge", "pspin",
        "orb", "bmin", "bmax", "nbb", "zmax", "ngrid", "gridtype",
        "vproj", "ne", "nodiag", "xmlfile", "i_init", "dtime",
        "elp", "talp", "tcoef", "tcenter", "tpower",
        "palp", "pcoef", "pcenter", "ppower"
    ]
    validate_input(inp, required_vars)

    # --- System Setup ---
    tmol, pmol, mol, sgeom, sbasis, scharge, sspin = setup_molecules(
        tgeom, tbasis, tcharge, tspin,
        pgeom, pbasis, pcharge, pspin,
        orb, debug
    )

    # --- Orbitals ---
    tmo, tmo_e = compute_orbitals(tmol, orb, alp=talp, coef=tcoef, center=tcenter, power=tpower, debug=debug)
    pmo, pmo_e = compute_orbitals(pmol, orb, alp=palp, coef=pcoef, center=pcenter, power=ppower, debug=debug)
    ntmo, npmo = len(tmo), len(pmo)
    nmo = ntmo + npmo
    smo = np.block([[tmo, np.zeros((ntmo, npmo))], [np.zeros((npmo, ntmo)), pmo]])

    # --- Franck-Condon Factors ---
    # Get atomic masses
    masses = tmol.atom_mass_list()
    if len(masses) > 2:
        print('Reduced mass only implemented for diatomics')
    elif len(masses) > 1:
        mass1 = masses[0]*isotop1
        mass2 = masses[1]*isotop2
        mass = mass1*mass2/(mass1+mass2)*1822.888
    else:
        mass = 0.0 # atomic case No vibration
    print()
    print('Target reduced mass : ',mass)

    csfs = process_xml_csf(xmlfile)
    ncsfs = len(csfs)

    nep_csf = []
    for csf in csfs:
       _, alpe, betae = csf.terms[0]
       nte = int(np.count_nonzero(np.array(alpe)<ntmo) + np.count_nonzero(np.array(betae)<ntmo))
       npe = len(alpe)+len(betae)-nte
       nep_csf.append(npe)

    # --- Asymptotic Energies ---
    if orb == "modpot":
        ovmo, kin, pot, deriv_z = hcore_modpot(np.concatenate((talp, palp)), np.concatenate((tcoef, pcoef)), np.concatenate((tcenter, [[0, 0, -10000.00]] * len(pcenter))), np.concatenate((tpower, ppower)), mol, smo)
    elif orb == "modpot_erf":
        ovmo, kin, pot, deriv_z = hcore_modpot_erf(np.concatenate((talp, palp)), np.concatenate((tcoef, pcoef)), np.concatenate((tcenter, [[0, 0, -10000.00]] * len(pcenter))), mol, smo)
    elif orb == "HF":
        ovmo, kin, pot, deriv_z = hcore(mol, smo)
    else:
        raise NotImplementedError("Only HF or modpot orbitals are implemented.")

    r12mo = twoeints(mol, smo)
    r12mo_antisym = r12mo - r12mo.transpose(0, 2, 1, 3)
    h1emo = kin + pot
    hmat, smat = cimat(ovmo, h1emo, r12mo, r12mo_antisym, ne, nmo, csfs)

    if nodiag:
        diagonal_mask = np.eye(hmat.shape[0], dtype=bool)
        hmat = hmat * diagonal_mask

    fulleig, fulleigv = np.linalg.eig(hmat)
    idx = fulleig.argsort()[::-1]
    fulleig = fulleig[idx]
    fulleigv = fulleigv[:,idx]
    print_asymp_elec_eig(fulleig,fulleigv,csfs)

    if len(sys.argv)>2 and sys.argv[2]=='-1' :
     sys.exit()

    n_elec_sta, eig, eigv, morse_p = select_sta(fulleig,fulleigv,sta_list)
    if any(len(t) == 0 for t in sta_list.values()) or morse_p == None: #fixed nuclei approximation
      print('Vibration DOF disabled, Fixed Nuclei Approximation.')
      evib = np.zeros((n_elec_sta,1))
      mat_fcf = [[None for _ in range(n_elec_sta)] for _ in range(n_elec_sta)]
      morse_p = []
      for i in range(n_elec_sta):
        morse_p.append([0.,0.,0.,1])
        for j in range(n_elec_sta):
          mat_fcf[j][i] = [[1.0]]
      mat_fcf = np.array(mat_fcf)
    else:
      evib, mat_fcf = compute_fcf(morse_p, mass)

    if len(sys.argv)>2 and sys.argv[2]=='0' :
      # Create a directory to store CSV files
      os.makedirs("fcf_csv_files", exist_ok=True)
      for i in range(len(mat_fcf)):
        for j in range(len(mat_fcf[i])):
          fcf = mat_fcf[j][i]
          filename = f"fcf_csv_files/fcf_j{j}_i{i}.csv"
          np.savetxt(filename, fcf, delimiter=" ", fmt="%.6f", header=f"FCF for (j={j}, i={i})")

    print_asymp_eig(eig,eigv,morse_p,evib)

    sta_file = os.path.join(run_dir, "sta")
    fsta = open(sta_file,"w")
    nsta = len(eig)
    for i in range(n_elec_sta):
      n_vib_i = morse_p[i][3]
      for j in range(n_vib_i):
          print(i, j, eig[i].real+evib[i][j], eig[i].real, evib[i][j],file=fsta)
    fsta.close()

    print("Asymptotic energies computed.")
    print()

    if(len(sys.argv)>2 and sys.argv[2]=='0'):
     sys.exit()

    # --- Collision Dynamics ---
    blist, zlist, tlist = setup_collision_grid(bmin, bmax, nbb, zmax, ngrid, gridtype, vproj)
    n_vib_total = sum(p[3] for p in morse_p)
    vib_indices = []
    start = 0
    for i in range(n_elec_sta):
        n_vib_i = morse_p[i][3]
        vib_indices.append((start, start + n_vib_i))
        start += n_vib_i

    # --- TDSE Propagation for Each Impact Parameter ---
    prob_file = os.path.join(run_dir, "prob")
    fprob = open(prob_file,"w")
    for b in blist:
        tmat = []
        for zproj in zlist:
            time = zproj / vproj

            xp, yp, zp = b, 0, zproj
            pgeom = elp + f" {xp} {yp} {zp}"
            sgeom = tgeom + pgeom
            mol = gto.M(atom=sgeom, basis=sbasis, charge=scharge, spin=sspin)
            mol.build(unit="Bohr")

            zpcenter = [np.repeat([xp,yp,zp], len(pcenter))]
            salp = np.concatenate((talp, palp), axis=0)
            scoef = np.concatenate((tcoef, pcoef), axis=0)
            scenter = np.concatenate((tcenter, zpcenter), axis=0)
            spower = np.concatenate((tpower, ppower), axis=0)

            if orb == "modpot":
                ovmo, kin, pot, deriv_z = hcore_modpot(salp, scoef, scenter, spower, mol, smo)
            elif orb == "modpot_erf":
                ovmo, kin, pot, deriv_z = hcore_modpot_erf(salp, scoef, scenter, mol, smo)
            elif orb == "HF":
                ovmo, kin, pot, deriv_z = hcore(mol, smo)
            else:
                raise NotImplementedError("Only HF or modpot orbitals are implemented.")

            r12mo = twoeints(mol, smo)
            r12mo = r12mo.astype(np.complex128)  # Convert to complex

            #dipole_z = mol.intor('int1e_r')[2]  # Dipole integrals (z-component)
            #dipole_z_mo = np.dot(smo.T.conj(), np.dot(dipole_z, smo))
            #z_mo = np.abs(dipole_z_mo)*np.sign(zproj)#/(np.abs(ovmo)+1e-12)**0.5

            phase_matrix = np.zeros((nmo, nmo), dtype=np.complex128)
            phase_matrix[0:ntmo, 0:ntmo] = 1.0  # tmo-tmo: phase = 1
            phase_matrix[ntmo:nmo, ntmo:nmo] = 1.0  # pmo-pmo: phase = 1

            # Fill off-diagonal blocks (tmo-pmo pairs)
            for i in range(ntmo):
              for j in range(ntmo, nmo):
                 phase_ij = np.exp(+vproj * zproj * 1.0j) * np.exp(-0.5 * vproj**2 * time * 1.0j)
                 #phase_ij = np.exp(+vproj * z_mo[j,i] * 1.0j) * np.exp(-0.5 * vproj**2 * time * 1.0j)
                 phase_matrix[i, j] = phase_ij
                 phase_matrix[j, i] = np.conj(phase_ij)  # Ensure Hermitian symmetry

            #phase = np.exp(+vproj*zproj*1.0j)*np.exp(-0.5*vproj**2*time*1.0j)
            #r12mo[:, :, :, ntmo:nmo] *= phase  # σ is pmo
            #r12mo[:, :, ntmo:nmo, :] *= np.conj(phase)  # λ is pmo
            #r12mo[:, ntmo:nmo, :, :] *= phase  # ν is pmo
            #r12mo[ntmo:nmo, :, :, :] *= np.conj(phase)  # μ is pmo

            # Reshape phase_matrix for (μ, ν) and (λ, σ)
            phase_mu_nu = phase_matrix.reshape(nmo, nmo, 1, 1)  # For (μ, ν)
            phase_la_sigma = phase_matrix.reshape(1, 1, nmo, nmo)  # For (λ, σ)

            # Apply phase factors to r12mo
            r12mo *= phase_mu_nu * phase_la_sigma

            r12mo_antisym = r12mo - r12mo.transpose(0, 2, 1, 3)

            h1emo = h1emo.astype(np.complex128)  # Convert to complex
            ovmo = ovmo.astype(np.complex128)  # Convert to complex
            kin = kin.astype(np.complex128)  # Convert to complex

            kin[0:ntmo,ntmo:nmo] -= vproj**2*ovmo[0:ntmo,ntmo:nmo]
            kin[0:ntmo,ntmo:nmo] += 2.0*1j*vproj*deriv_z[0:ntmo,ntmo:nmo]

            h1emo = kin + pot
            # --- Apply phase to 1-electron matrices (h1emo and ovmo) ---
            h1emo[:,:] *= phase_matrix[:,:]
            ovmo[:,:] *= phase_matrix[:,:]

            matH, matS = cimat(ovmo, h1emo, r12mo, r12mo_antisym, ne, nmo, csfs)

            # Transform to adiabatic basis
            hmat = np.transpose(eigv) @ matH @ eigv
            smat = np.transpose(eigv) @ matS @ eigv

            # Build vibronic matrices
            hmatvib, smatvib = build_vibronic_matrices(hmat, smat, mat_fcf, evib, morse_p, vib_indices)

            # Solve for this zproj
            inv_smatvib = np.linalg.inv(smatvib)
            mat = inv_smatvib @ hmatvib
            tmat.append(mat)

        # Interpolate and propagate
        hmat_interp = interp1d(tlist, tmat, axis=0)
        psi0 = np.zeros(n_vib_total, dtype=complex)
        psi0[i_init] = 1.0
        ntime = int(2.0 * zmax / (vproj * dtime))
        t_grid = np.linspace(zlist[0] / vproj, zlist[-1] / vproj, ntime)
        prob = run_tdse(hmat_interp, psi0, t_grid)

        # Print probabilities
        formatted_prob = "  ".join([f"{p:.6f}" for p in prob])
        print()
        print("Probabilities:")
        print()
        print(f"{b} {formatted_prob} {np.sum(prob):.6f}")
        print(f"{b} {formatted_prob} {np.sum(prob):.6f}",file=fprob)
        # Define the output file path

    fprob.close()
