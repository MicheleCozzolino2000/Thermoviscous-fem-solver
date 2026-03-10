"""
Frequency sweep for the thermoviscous tube solver.

This script loads the cylindrical tube mesh, computes the thermophysical
properties of air, solves the thermoviscous FEM problem over a frequency
range, evaluates the inlet surface impedance, and compares the numerical
results with the analytical thermoviscous solution for a cylindrical tube.
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import ufl
from mpi4py import MPI
from scipy.special import jv

from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, fem, geometry, io
from dolfinx.fem import Function, dirichletbc
from dolfinx.fem.petsc import LinearProblem


class MicrophonePressure:
    def __init__(self, domain, microphone_position):
        self._domain = domain
        self._position = np.asarray(
            microphone_position, dtype=domain.geometry.x.dtype
        ).reshape(3, -1)
        self._local_cells, self._local_position = self.compute_local_microphones()

    def compute_local_microphones(
        self,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]]:
        points = self._position.T
        bb_tree = geometry.bb_tree(self._domain, self._domain.topology.dim)

        cells = []
        points_on_proc = []

        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(
            self._domain, cell_candidates, points
        )

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        return (
            np.asarray(cells, dtype=np.int32),
            np.asarray(points_on_proc, dtype=self._domain.geometry.x.dtype),
        )

    def listen(
        self,
        p_a: fem.Function,
        recompute_collisions: bool = False,
    ) -> npt.NDArray[np.complexfloating]:
        if recompute_collisions:
            self._local_cells, self._local_position = self.compute_local_microphones()

        if len(self._local_cells) > 0:
            return p_a.eval(self._local_position, self._local_cells)

        return np.zeros(0, dtype=default_scalar_type)


class ThermoviscousSolver:
    """
    Finite element solver for the linearized thermoviscous acoustic problem
    in the frequency domain.
    """

    def __init__(
        self,
        domain,
        facet_tags,
        inlet_tag,
        outlet_tag,
        wall_tag,
        freq,
        rho,
        mu,
        lambda_,
        cp,
        ka,
        T0,
        p0,
        c,
    ):
        self.domain = domain
        self.facet_tags = facet_tags

        self.inlet_tag = inlet_tag
        self.outlet_tag = outlet_tag
        self.wall_tag = wall_tag
        self.freq = freq

        self.rho = rho
        self.mu = mu
        self.lambda_ = lambda_
        self.cp = cp
        self.ka = ka
        self.T0 = T0
        self.p0 = p0
        self.c = c

        cell = domain.basix_cell()

        # P2 for velocity and temperature, P1 for pressure
        P2_vec = element("Lagrange", cell, degree=2, shape=(domain.geometry.dim,))
        P2_sca = element("Lagrange", cell, degree=2)
        P1 = element("Lagrange", cell, degree=1)

        mixed = mixed_element([P2_vec, P2_sca, P1])

        self.Me = fem.functionspace(domain, mixed)
        self.Ve = self.Me.sub(0)
        self.Te = self.Me.sub(1)
        self.Pe = self.Me.sub(2)

    def solve(self):
        omega = 2 * np.pi * self.freq
        T0 = self.T0
        p0 = self.p0
        rho = self.rho
        mu = self.mu
        lam = self.lambda_
        ka = self.ka
        cp = self.cp

        dx = ufl.dx(self.domain)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(self.domain)

        # Trial and test functions
        U = ufl.TrialFunction(self.Me)
        v, T, p = ufl.split(U)

        V = ufl.TestFunction(self.Me)
        w, s, q = ufl.split(V)

        Ve = self.Ve
        Te = self.Te
        Pe = self.Pe

        Ve_collapsed, _ = Ve.collapse()
        Te_collapsed, _ = Te.collapse()
        Pe_collapsed, _ = Pe.collapse()

        # Bilinear form
        a = (
            1j * omega * rho * ufl.inner(v, w) * dx
            + 2 * mu * ufl.inner(ufl.sym(ufl.grad(v)), ufl.sym(ufl.grad(w))) * dx
            + lam * ufl.inner(ufl.div(v), ufl.div(w)) * dx
            - ufl.inner(p, ufl.div(w)) * dx
            - 1j * omega * cp * rho * ufl.inner(T, s) * dx
            - ka * ufl.inner(ufl.grad(T), ufl.grad(s)) * dx
            + 1j * omega * ufl.inner(p, s) * dx
            - ufl.inner(ufl.div(v), q) * dx
            + 1j * omega * (1 / T0) * ufl.inner(T, q) * dx
            - 1j * omega * (1 / p0) * ufl.inner(p, q) * dx
        )

        # Boundary excitation
        p_in = Function(Pe_collapsed)
        p_in.x.array[:] = 1.0
        g = p_in

        # Source terms
        f = Function(Ve_collapsed)
        Q_source = Function(Te_collapsed)

        # Linear form
        L = (
            ufl.inner(f, w) * dx
            - ufl.inner(Q_source, s) * dx
            - ufl.inner(g * n, w) * ds(self.inlet_tag)
        )

        # Boundary conditions
        v_wall = Function(Ve_collapsed)
        v_wall.x.array[:] = 0.0 + 0.0j

        T_zero = Function(Te_collapsed)
        T_zero.x.array[:] = 0.0

        outlet_facets = self.facet_tags.find(self.outlet_tag)
        wall_facets = self.facet_tags.find(self.wall_tag)

        dofs_v_outlet = fem.locate_dofs_topological(
            (Ve, Ve_collapsed), self.domain.topology.dim - 1, outlet_facets
        )
        dofs_v_side = fem.locate_dofs_topological(
            (Ve, Ve_collapsed), self.domain.topology.dim - 1, wall_facets
        )
        bc_v_outlet = dirichletbc(v_wall, dofs_v_outlet, Ve)
        bc_v_side = dirichletbc(v_wall, dofs_v_side, Ve)

        dofs_T_outlet = fem.locate_dofs_topological(
            (Te, Te_collapsed), self.domain.topology.dim - 1, outlet_facets
        )
        dofs_T_side = fem.locate_dofs_topological(
            (Te, Te_collapsed), self.domain.topology.dim - 1, wall_facets
        )
        bc_T_outlet = dirichletbc(T_zero, dofs_T_outlet, Te)
        bc_T_side = dirichletbc(T_zero, dofs_T_side, Te)

        bcs = [bc_v_outlet, bc_v_side, bc_T_side, bc_T_outlet]

        U_sol = Function(self.Me)

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            u=U_sol,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

        problem.solve()
        v_sol, T_sol, p_sol = U_sol.split()

        return v_sol, T_sol, p_sol


def compute_air_properties(Tc, p_0, hum, xc):
    T = 273.15 + Tc
    Ra = 287.05

    psv = np.exp(
        1.2378847e-5 * T**2
        - 1.9121316e-2 * T
        + 33.93711047
        - 6.3431645e3 * T**(-1)
    )
    f = 1.00062 + 3.14e-8 * p_0 + 5.6e-7 * Tc**2
    xw = (hum / 100) * (psv / p_0) * f

    Z = (
        1
        - (p_0 / T)
        * (
            1.58123e-6
            - 2.9331e-8 * Tc
            + 1.1043e-10 * Tc**2
            + (5.707e-6 - 2.051e-8 * Tc) * xw
            + (1.9898e-4 - 2.376e-6 * Tc) * xw**2
        )
        + (p_0 / T) ** 2 * (1.83e-11 - 0.765e-8 * xw**2)
    )

    rho0 = (
        (3.48349 + 1.44 * (xc - 0.0004))
        * 10**(-3)
        * (p_0 / (Z * T))
        * (1 - 0.3780 * xw)
    )

    c = (
        331.5024
        + 0.603055 * Tc
        - 0.000528 * Tc**2
        + (51.471935 + 0.1495874 * Tc - 0.000782 * Tc**2) * xw
        + (-1.82e-7 + 3.73e-8 * Tc - 2.93e-10 * Tc**2) * p_0
        + (-85.20931 - 0.228525 * Tc + 5.91e-5 * Tc**2) * xc
        - 2.835149 * xw**2
        - 2.5e-13 * p_0**2
        + 29.179762 * xc**2
        + 0.000486 * xw * p_0 * xc
    )

    mu_air = (
        84.986
        + 7 * T
        + (113.157 - T) * xw
        - 3.7501e-3 * T**2
        - 100.015 * xw**2
    ) * 10**(-8)

    cp1 = (
        0.251625
        - 9.2525e-5 * T
        + 2.1334e-7 * T**2
        - 1.0043e-10 * T**3
        + (0.12477 - 2.283e-5 * T + 1.267e-7 * T**2) * xw
        + (0.01116 + 4.61e-6 * T + 1.74e-8 * T**2) * xw**2
    )

    cp = cp1 * 4.1868 * 10**3
    eta = 0.60 * mu_air
    lambda_air = eta - (2 / 3) * mu_air
    ni_air = mu_air / rho0

    k_air = (
        1.30030e-3
        + 9.36766e-5 * T
        - 4.44247e-8 * T**2
        + 2.31716e-11 * T**3
        - 6.59976e-15 * T**4
    )

    cv = cp - Ra
    Pr = mu_air * cp / k_air
    alpha = k_air / (rho0 * cp)
    gamma = cp / (cp - Ra)

    return {
        "T": T,
        "Ra": Ra,
        "xw": xw,
        "rho0": rho0,
        "c": c,
        "mu_air": mu_air,
        "cp": cp,
        "eta": eta,
        "lambda_air": lambda_air,
        "ni_air": ni_air,
        "k_air": k_air,
        "cv": cv,
        "Pr": Pr,
        "alpha": alpha,
        "gamma": gamma,
    }


def compute_surface_impedance(domain, facet_tags, inlet_tag, v_sol, p_sol):
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    n = ufl.FacetNormal(domain)
    v_n = -ufl.dot(v_sol, n)

    area = fem.assemble_scalar(fem.form(1.0 * ds(inlet_tag)))
    avg_velocity = fem.assemble_scalar(fem.form(v_n * ds(inlet_tag))) / area
    avg_pressure = fem.assemble_scalar(fem.form(p_sol * ds(inlet_tag))) / area

    Zs = avg_pressure / avg_velocity
    return Zs


def solve_and_postprocess(
    freq,
    domain,
    facet_tags,
    inlet_tag,
    outlet_tag,
    wall_tag,
    rho0,
    mu_air,
    lambda_air,
    cp,
    k_air,
    T,
    p_0,
    c,
):
    solver = ThermoviscousSolver(
        domain,
        facet_tags,
        inlet_tag,
        outlet_tag,
        wall_tag,
        freq,
        rho0,
        mu_air,
        lambda_air,
        cp,
        k_air,
        T,
        p_0,
        c,
    )

    v_sol, T_sol, p_sol = solver.solve()
    Zs = compute_surface_impedance(domain, facet_tags, inlet_tag, v_sol, p_sol)

    return Zs


def analytic_Zs(freqs, R_tube, L_tube, rho0, gamma, p_0, ni_air, alpha):
    Zs_stinson = np.zeros(len(freqs), dtype=np.complex128)

    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        delta_v = np.sqrt(2 * ni_air / omega)
        delta_t = np.sqrt(2 * alpha / omega)

        z_v = (1j - 1) * R_tube / delta_v
        z_t = (1j - 1) * R_tube / delta_t

        fv = 2 * jv(1, z_v) / (jv(0, z_v) * z_v)
        fk = 2 * jv(1, z_t) / (jv(0, z_t) * z_t)

        rho_eq = rho0 / (1 - fv)
        K_eq = gamma * p_0 / (1 + (gamma - 1) * fk)
        Zc = np.sqrt(rho_eq * K_eq)
        k_c = omega * np.sqrt(rho_eq / K_eq)

        Z_s_tv = -1j * Zc / np.tan(k_c * L_tube)
        Zs_stinson[i] = Z_s_tv

    return Zs_stinson


def main():
    os.makedirs("ThermoViscous_tube", exist_ok=True)
    os.makedirs("ThermoViscous_tube/mesh_data", exist_ok=True)

    # Mesh
    domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
        f"mesh_impedance.msh",
        MPI.COMM_WORLD,
        0,
        gdim=3,
    )

    # Physical parameters
    Tc = 21
    p_0 = 101325
    hum = 20
    xc = 0.0004

    air = compute_air_properties(Tc, p_0, hum, xc)

    T = air["T"]
    rho0 = air["rho0"]
    c = air["c"]
    mu_air = air["mu_air"]
    cp = air["cp"]
    lambda_air = air["lambda_air"]
    ni_air = air["ni_air"]
    k_air = air["k_air"]
    alpha = air["alpha"]
    gamma = air["gamma"]

    # Reference boundary-layer thickness
    freq_test = 100
    omega = 2 * np.pi * freq_test
    delta_v = np.sqrt(2 * ni_air / omega)
    delta_t = np.sqrt(2 * alpha / omega)
    delta_ref = min(delta_v, delta_t)

    properties = {
        "Humidity [%]": hum,
        "Quiescent Temperature [K]": T,
        "Quiescent Pressure [Pa]": p_0,
        "Speed of sound [m/s]": c,
        "Quiescent air density [kg/m^3]": rho0,
        "Thermal conductivity [W/m*K]": k_air,
        "Dynamic viscosity [Pa*s]": mu_air,
        "Bulk viscosity [Pa*s]": air["eta"],
        "Second viscosity [Pa*s]": lambda_air,
        "Kinematic viscosity [m^2/s]": ni_air,
        "Ratio of specific heats [-]": gamma,
        "Specific heat cp [J/kg*K]": cp,
        "Specific heat cv [J/kg*K]": air["cv"],
        "Prandtl number [-]": air["Pr"],
        "Thermal diffusivity [m^2/s]": alpha,
        "Thermal Boundary Layer [mm]": delta_t * 1e3,
        "Viscous Boundary Layer [mm]": delta_v * 1e3,
    }

    # Geometry
    L_tube = 0.05
    R_tube = 0.0005

    # Optional point microphone
    mic_pos = np.array([0.001, 0.0, 0.0])

    # Boundary tags
    inlet_tag = 2
    outlet_tag = 3
    wall_tag = 4

    # Frequency sweep
    freqs = np.arange(50, 1001, 50)
    Zs_vec = np.zeros(len(freqs), dtype=np.complex128)

    for i, freq in enumerate(freqs):
        Zs_vec[i] = solve_and_postprocess(
            float(freq),
            domain,
            facet_tags,
            inlet_tag,
            outlet_tag,
            wall_tag,
            rho0,
            mu_air,
            lambda_air,
            cp,
            k_air,
            T,
            p_0,
            c,
        )
        progress = (i + 1) / len(freqs) * 100
        print(f"Calcolo in corso: {progress:.0f}%")

    Zs_stinson = analytic_Zs(freqs, R_tube, L_tube, rho0, gamma, p_0, ni_air, alpha)

    # Plot: real and imaginary parts
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        freqs,
        np.real(Zs_vec),
        linestyle="-",
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2.0,
        linewidth=2.2,
        label=r"$Z_s$ FEM",
    )
    plt.plot(
        freqs,
        np.real(Zs_stinson),
        linestyle="--",
        marker="X",
        markersize=9,
        linewidth=2,
        label=r"$Z_s$ analytical",
    )
    plt.xlabel(r"Frequency [$\mathrm{Hz}$]")
    plt.ylabel(r"Re($Z_s$) [$\mathrm{Pa\,s/m}$]")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        freqs,
        np.imag(Zs_vec),
        linestyle="-",
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2.0,
        linewidth=2.2,
        label=r"$Z_s$ FEM",
    )
    plt.plot(
        freqs,
        np.imag(Zs_stinson),
        linestyle="--",
        marker="X",
        markersize=9,
        linewidth=2,
        label=r"$Z_s$ analytical",
    )
    plt.xlabel(r"Frequency [$\mathrm{Hz}$]")
    plt.ylabel(r"Im($Z_s$) [$\mathrm{Pa\,s/m}$]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Zs_Re_Im.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot: magnitude and phase
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        freqs,
        np.abs(Zs_vec),
        linestyle="-",
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2.0,
        linewidth=2.2,
        label=r"$Z_s$ FEM",
    )
    plt.plot(
        freqs,
        np.abs(Zs_stinson),
        linestyle="--",
        marker="X",
        markersize=9,
        linewidth=2,
        label=r"$Z_s$ analytical",
    )
    plt.xlabel(r"Frequency [$\mathrm{Hz}$]")
    plt.ylabel(r"$|Z_s|$ [$\mathrm{Pa\,s/m}$]")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        freqs,
        np.angle(Zs_vec),
        linestyle="-",
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2.0,
        linewidth=2.2,
        label=r"$Z_s$ FEM",
    )
    plt.plot(
        freqs,
        np.angle(Zs_stinson),
        linestyle="--",
        marker="X",
        markersize=9,
        linewidth=2,
        label=r"$Z_s$ analytical",
    )
    plt.xlabel(r"Frequency [$\mathrm{Hz}$]")
    plt.ylabel(r"Phase [$\mathrm{rad}$]")
    plt.ylim(-np.pi, 0)
    plt.yticks(
        [-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0],
        [
            r"$-\pi$",
            r"$-\frac{3\pi}{4}$",
            r"$-\frac{\pi}{2}$",
            r"$-\frac{\pi}{4}$",
            r"$0$",
        ],
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Zs_Magnitude_Phase.png", dpi=300, bbox_inches="tight")
    plt.show()

    np.savez_compressed(
        "impedance_results.npz",
        freqs=freqs,
        Zs_vec=Zs_vec,
        Zs_stinson=Zs_stinson,
    )


if __name__ == "__main__":
    main()