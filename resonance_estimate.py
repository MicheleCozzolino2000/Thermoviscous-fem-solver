"""
Estimate the resonance frequency of a Helmholtz resonator
using thermoviscous equivalent properties for cylindrical ducts.
"""

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import jv

# Resonator geometry
Rn = 0.0056   # Neck radius [m]
Ln = 0.025    # Neck length [m]
Rc = 0.101    # Cavity radius [m]
Lc = 0.150    # Cavity length [m]

# Physical parameters
rho0 = 1.204      # Density [kg/m^3]
gamma = 1.3978    # Ratio of specific heats [-]
p0 = 101325       # Pressure [Pa]
nu = 1.5086e-5    # Kinematic viscosity [m^2/s]
alpha = 2.101e-5  # Thermal diffusivity [m^2/s]


def rho_eq_cyl(freq, R, rho0, nu):
    """Compute the equivalent density for a cylindrical duct."""
    omega = 2 * np.pi * freq
    delta_v = np.sqrt(2 * nu / omega)
    z_v = (1j - 1) * R / delta_v
    fv = 2 * jv(1, z_v) / (jv(0, z_v) * z_v)
    return rho0 / (1 - fv)


def K_eq_cyl(freq, R, gamma, p0, alpha):
    """Compute the equivalent bulk modulus for a cylindrical cavity."""
    omega = 2 * np.pi * freq
    delta_t = np.sqrt(2 * alpha / omega)
    z_t = (1j - 1) * R / delta_t
    fk = 2 * jv(1, z_t) / (jv(0, z_t) * z_t)
    return gamma * p0 / (1 + (gamma - 1) * fk)


def helmholtz_resonance_residual(freq, Rn, Rc, Ln, Lc, rho0, gamma, p0, nu, alpha):
    """Residual used to estimate the Helmholtz resonance frequency."""
    An = np.pi * Rn**2
    Ac = np.pi * Rc**2

    # Neck end correction based on Ingard (1953)
    delta = 1.7 * Rn
    Ln_eff = Ln + delta

    rho_n_eq = rho_eq_cyl(freq, Rn, rho0, nu)
    K_c_eq = K_eq_cyl(freq, Rc, gamma, p0, alpha)

    # Use the real parts of the equivalent properties in the resonance condition
    omega_res_sq = (
        np.real(K_c_eq) / np.real(rho_n_eq)
    ) * (An / Ac) * (1.0 / (Lc * Ln_eff))

    omega = 2 * np.pi * freq
    return omega**2 - omega_res_sq


sol = root_scalar(
    helmholtz_resonance_residual,
    args=(Rn, Rc, Ln, Lc, rho0, gamma, p0, nu, alpha),
    bracket=[35.0, 70.0],
    method="brentq",
)

if not sol.converged:
    raise RuntimeError("Root finding did not converge.")
print(f"Iterations: {sol.iterations}")

f_res = sol.root
print(f"Resonance frequency = {f_res:.6f} Hz")