import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import ufl
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx import fem, geometry, io, default_scalar_type
from dolfinx.fem import Function, dirichletbc
from dolfinx_mpc import MultiPointConstraint
from dolfinx_mpc.problem import LinearProblem as MPCLinearProblem

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
        inlet_neck_tag,
        wall_tag,
        sym_yx_tag,
        sym_theta_tag,
        wall_guide_tag,
        theta_rad,
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
        self.inlet_neck_tag = inlet_neck_tag
        self.wall_tag = wall_tag
        self.sym_yx_tag = sym_yx_tag
        self.sym_theta_tag = sym_theta_tag
        self.wall_guide_tag = wall_guide_tag

        self.theta = theta_rad
        self.freq = freq

        self.rho = rho
        self.mu = mu
        self.lambda_ = lambda_
        self.cp = cp
        self.ka = ka
        self.T0 = T0
        self.p0 = p0
        self.c = c

        # Mixed finite element space:
        # P2 vector for velocity, P2 scalar for temperature, P1 scalar for pressure
        cell = domain.basix_cell()
        P2_vec = element("Lagrange", cell, degree=2, shape=(domain.geometry.dim,))
        P2_sca = element("Lagrange", cell, degree=2)
        P1 = element("Lagrange", cell, degree=1)

        mixed = mixed_element([P2_vec, P2_sca, P1])

        self.Me = fem.functionspace(domain, mixed)
        self.Ve = self.Me.sub(0)
        self.Te = self.Me.sub(1)
        self.Pe = self.Me.sub(2)

    def solve(self):
        """
        Solve the coupled thermoviscous problem and return velocity,
        temperature, and pressure fields.
        """
        omega = 2 * np.pi * self.freq
        T0 = self.T0
        p0 = self.p0
        rho = self.rho
        mu = self.mu
        lam = self.lambda_
        ka = self.ka
        cp = self.cp
        theta = self.theta

        dx = ufl.dx(self.domain)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(self.domain)

        # Trial and test functions
        U = ufl.TrialFunction(self.Me)
        v, T, p = ufl.split(U)

        V = ufl.TestFunction(self.Me)
        w, s, q = ufl.split(V)

        Ve_collapsed, _ = self.Ve.collapse()
        Te_collapsed, _ = self.Te.collapse()
        Pe_collapsed, _ = self.Pe.collapse()

        # Bilinear form
        a = (
            1j * omega * rho * ufl.inner(v, w) * dx
            + 2 * mu * ufl.inner(ufl.sym(ufl.grad(v)), ufl.sym(ufl.grad(w))) * dx
            + lam * ufl.inner(ufl.div(v), ufl.div(w)) * dx
            - ufl.inner(p, ufl.div(w)) * dx
            - 1j * omega * cp * rho * (1 / T0) * ufl.inner(T, s) * dx
            - ka * (1 / T0) * ufl.inner(ufl.grad(T), ufl.grad(s)) * dx
            + 1j * omega * (1 / T0) * ufl.inner(p, s) * dx
            - ufl.inner(ufl.div(v), q) * dx
            + 1j * omega * (1 / T0) * ufl.inner(T, q) * dx
            - 1j * omega * (1 / p0) * ufl.inner(p, q) * dx
        )

        # Boundary excitation
        p_in = Function(Pe_collapsed)
        p_in.x.array[:] = 1.0

        # Zero volumetric forcing and thermal source
        f = Function(Ve_collapsed)
        Q_source = Function(Te_collapsed)

        # Linear form
        L = (
            ufl.inner(f, w) * dx
            - ufl.inner(Q_source, s) * dx
            - ufl.inner(p_in * n, w) * ds(self.inlet_tag)
        )

        # Boundary values
        v_wall = Function(Ve_collapsed)
        v_wall.x.array[:] = 0.0 + 0.0j

        T_zero = Function(Te_collapsed)
        T_zero.x.array[:] = 0.0

        inlet_facets = self.facet_tags.find(self.inlet_tag)
        inlet_neck_facets = self.facet_tags.find(self.inlet_neck_tag)
        wall_facets = self.facet_tags.find(self.wall_tag)
        wall_guide_facets = self.facet_tags.find(self.wall_guide_tag)
        sym_yx_facets = self.facet_tags.find(self.sym_yx_tag)
        sym_theta_facets = self.facet_tags.find(self.sym_theta_tag)

        # No-slip condition on solid walls
        dofs_v_wall = fem.locate_dofs_topological(
            (self.Ve, Ve_collapsed), self.domain.topology.dim - 1, wall_facets
        )
        bc_v_wall = dirichletbc(v_wall, dofs_v_wall, self.Ve)

        dofs_v_wallguide = fem.locate_dofs_topological(
            (self.Ve, Ve_collapsed), self.domain.topology.dim - 1, wall_guide_facets
        )
        bc_v_wallguide = dirichletbc(v_wall, dofs_v_wallguide, self.Ve)

        # Isothermal walls
        dofs_T_wall = fem.locate_dofs_topological(
            (self.Te, Te_collapsed), self.domain.topology.dim - 1, wall_facets
        )
        bc_T_wall = dirichletbc(T_zero, dofs_T_wall, self.Te)

        dofs_T_wallguide = fem.locate_dofs_topological(
            (self.Te, Te_collapsed), self.domain.topology.dim - 1, wall_guide_facets
        )
        bc_T_wallguide = dirichletbc(T_zero, dofs_T_wallguide, self.Te)

        # Symmetry conditions on velocity components
        Vy = self.Me.sub(0).sub(1)
        Vz = self.Me.sub(0).sub(2)

        Vy_c, _ = Vy.collapse()
        Vz_c, _ = Vz.collapse()

        zero_y = fem.Function(Vy_c)
        zero_y.x.array[:] = 0.0 + 0.0j

        zero_z = fem.Function(Vz_c)
        zero_z.x.array[:] = 0.0 + 0.0j

        dofs_vz_sym = fem.locate_dofs_topological(
            (Vz, Vz_c), self.domain.topology.dim - 1, sym_yx_facets
        )
        bc_vz_sym = fem.dirichletbc(zero_z, dofs_vz_sym, Vz)

        def centerline_axis(x):
            return np.sqrt(x[1] ** 2 + x[2] ** 2) < 1e-10

        dofs_vy = fem.locate_dofs_geometrical((Vy, Vy_c), centerline_axis)
        dofs_vz = fem.locate_dofs_geometrical((Vz, Vz_c), centerline_axis)

        bc_axis_y = dirichletbc(zero_y, dofs_vy, Vy)
        bc_axis_z = dirichletbc(zero_z, dofs_vz, Vz)

        bcs = [
            bc_v_wall,
            bc_v_wallguide,
            bc_T_wall,
            bc_T_wallguide,
            bc_vz_sym,
            bc_axis_y,
            bc_axis_z,
        ]

        # Multi-point constraint for the sector symmetry plane
        mpc = MultiPointConstraint(self.Me)

        nh = fem.Function(Ve_collapsed)
        arr = nh.x.array.reshape((-1, 3))
        arr[:] = (0.0, -np.sin(theta), np.cos(theta))
        nh.x.scatter_forward()

        mpc.create_slip_constraint(
            self.Ve,
            (self.facet_tags, self.sym_theta_tag),
            nh,
            bcs,
        )
        mpc.finalize()

        U_sol = Function(mpc.function_space)

        problem = MPCLinearProblem(
            a,
            L,
            mpc,
            bcs=bcs,
            u=U_sol,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

        problem.solve()
        mpc.backsubstitution(U_sol)

        v_sol, T_sol, p_sol = U_sol.split()
        return v_sol, T_sol, p_sol


def compute_air_properties(Tc, p_0, hum, xc):
    """
    Compute air properties from empirical correlations.
    """
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
        (3.48349 + 1.44 * (xc - 0.0004)) * 10 ** (-3) * (p_0 / (Z * T)) * (1 - 0.3780 * xw)
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
        + (113.157 - 1 * T) * xw
        - 3.7501e-3 * T**2
        - 100.015 * xw**2
    ) * 10 ** (-8)
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


def compute_surface_impedance(
    domain, facet_tags, inlet_neck_tag, wall_guide_tag, v_sol, p_sol
):
    """
    Compute surface impedance using area-averaged pressure and normal velocity.
    """
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags)
    n = ufl.FacetNormal(domain)

    v_n = -ufl.dot(v_sol("+"), n("+"))

    S1 = fem.assemble_scalar(fem.form(1.0 * dS(inlet_neck_tag)))
    S2 = fem.assemble_scalar(fem.form(1.0 * ds(wall_guide_tag)))
    S = S1 + S2

    avg_velocity = fem.assemble_scalar(fem.form(v_n * dS(inlet_neck_tag))) / S
    avg_pressure1 = fem.assemble_scalar(fem.form(p_sol("+") * dS(inlet_neck_tag))) / S1
    avg_pressure2 = fem.assemble_scalar(fem.form(p_sol * ds(wall_guide_tag))) / S2
    avg_pressure = (avg_pressure1 * S1 + avg_pressure2 * S2) / S

    Zs = avg_pressure / avg_velocity
    return Zs, avg_pressure, avg_velocity


def solve_and_postprocess(
    freq_test,
    domain,
    facet_tags,
    inlet_tag,
    inlet_neck_tag,
    wall_tag,
    sym_yx_tag,
    sym_theta_tag,
    wall_guide_tag,
    theta_rad,
    rho0,
    mu_air,
    lambda_air,
    cp,
    k_air,
    T,
    p_0,
    c,
):
    """
    Solve the thermoviscous problem at a given frequency and compute
    impedance-related quantities.
    """
    solver = ThermoviscousSolver(
        domain,
        facet_tags,
        inlet_tag,
        inlet_neck_tag,
        wall_tag,
        sym_yx_tag,
        sym_theta_tag,
        wall_guide_tag,
        theta_rad,
        freq_test,
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

    Zs, avg_p, avg_v = compute_surface_impedance(
        domain, facet_tags, inlet_neck_tag, wall_guide_tag, v_sol, p_sol
    )

    Z0 = rho0 * c
    r = (Zs - Z0) / (Zs + Z0)
    alpha_ = 1.0 - np.abs(r) ** 2

    if domain.comm.rank == 0:
        print(f"\nfreq_test = {freq_test}")
        print(f"Zs    = {Zs}  Pa*s/m")
        print(f"|Zs|  = {np.abs(Zs):.6e}  Pa*s/m")
        print(f"argZs = {np.angle(Zs):.6e}  rad")
        print(f"alpha = {alpha_:.6f}")
        print(f"r     = {r}")

    return Zs, alpha_, r, avg_p, avg_v


def plot_complex_quantity(freqs, values, ylabel, label_real, label_imag):
    plt.figure(figsize=(7, 5))
    plt.plot(freqs, values.real, "o-", linewidth=2, label=label_real)
    plt.plot(freqs, values.imag, "o-", linewidth=2, label=label_imag)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scalar_quantity(freqs, values, ylabel):
    plt.figure(figsize=(7, 5))
    plt.plot(freqs, values, "-", linewidth=2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # -----------------------------
    # Mesh and output configuration
    # -----------------------------
    theta_deg = 10
    name_mesh = "Threshold"
    tag_mesh = "RH_sector"

    config_tag = "config5"
    out_dir = "data_RH"
    os.makedirs(out_dir, exist_ok=True)

    fname = os.path.join(out_dir, f"data_RH_{config_tag}.npz")

    domain, cell_tags, facet_tags = io.gmshio.read_from_msh(
        f"{tag_mesh}_{theta_deg}deg_{name_mesh}.msh",
        MPI.COMM_WORLD,
        0,
        gdim=3,
    )

    # -----------------------------
    # Geometry
    # -----------------------------
    Rc = 0.101
    Lg = Rc
    Rn = 0.0056
    Ln = 0.025
    Lc = 0.150

    theta_rad = np.deg2rad(theta_deg)

    # -----------------------------
    # Physical parameters
    # -----------------------------
    Tc = 20
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
    cv = air["cv"]
    Pr = air["Pr"]
    alpha = air["alpha"]
    gamma = air["gamma"]

    # -----------------------------
    # Reference boundary-layer thickness
    # -----------------------------
    freq_test = 100
    omega = 2 * np.pi * freq_test
    delta_v = np.sqrt(2 * ni_air / omega)
    delta_t = np.sqrt(2 * alpha / omega)
    delta_ref = min(delta_v, delta_t)

    properties = {
        "Humidity [%]": hum,
        "Quiescent temperature [K]": T,
        "Quiescent pressure [Pa]": p_0,
        "Speed of sound [m/s]": c,
        "Quiescent air density [kg/m^3]": rho0,
        "Thermal conductivity [W/m*K]": k_air,
        "Dynamic viscosity [Pa*s]": mu_air,
        "Bulk viscosity [Pa*s]": air["eta"],
        "Second viscosity [Pa*s]": lambda_air,
        "Kinematic viscosity [m^2/s]": ni_air,
        "Ratio of specific heats [-]": gamma,
        "Specific heat cp [J/kg*K]": cp,
        "Specific heat cv [J/kg*K]": cv,
        "Prandtl number [-]": Pr,
        "Thermal diffusivity [m^2/s]": alpha,
        "Thermal boundary layer [mm]": delta_t * 1e3,
        "Viscous boundary layer [mm]": delta_v * 1e3,
    }

    # -----------------------------
    # Boundary tags
    # -----------------------------
    inlet_tag = 2
    inlet_neck_tag = 3
    neck_mouth_tag = 4
    wall_tag = 5
    sym_yx_tag = 6
    sym_theta_tag = 7
    wall_guide_tag = 8

    # -----------------------------
    # Frequency sweep
    # -----------------------------
    freqs = np.arange(25, 101, 5)

    Zs_vec = np.zeros(len(freqs), dtype=np.complex128)
    a_vec = np.zeros(len(freqs))
    r_vec = np.zeros(len(freqs), dtype=np.complex128)
    avg_p_vec = np.zeros(len(freqs), dtype=np.complex128)
    avg_v_vec = np.zeros(len(freqs), dtype=np.complex128)

    for i, f in enumerate(freqs):
        Zs_vec[i], a_vec[i], r_vec[i], avg_p_vec[i], avg_v_vec[i] = solve_and_postprocess(
            float(f),
            domain,
            facet_tags,
            inlet_tag,
            inlet_neck_tag,
            wall_tag,
            sym_yx_tag,
            sym_theta_tag,
            wall_guide_tag,
            theta_rad,
            rho0,
            mu_air,
            lambda_air,
            cp,
            k_air,
            T,
            p_0,
            c,
        )

    Z0 = rho0 * c

    if domain.comm.rank == 0:
        print(f"Z0 = {Z0}")

    # -----------------------------
    # Plots
    # -----------------------------
    plot_complex_quantity(
        freqs,
        Zs_vec,
        ylabel="Zs [Pa*s/m]",
        label_real="Re(Zs)",
        label_imag="Im(Zs)",
    )

    plot_scalar_quantity(
        freqs,
        a_vec,
        ylabel="Absorption coefficient alpha [-]",
    )

    plot_complex_quantity(
        freqs,
        avg_p_vec,
        ylabel="Average pressure [Pa]",
        label_real="Re(p)",
        label_imag="Im(p)",
    )

    plot_complex_quantity(
        freqs,
        avg_v_vec,
        ylabel="Average velocity [m/s]",
        label_real="Re(v)",
        label_imag="Im(v)",
    )

    # -----------------------------
    # Resonance frequency from Im(Zs)=0
    # -----------------------------
    ImZ = Zs_vec.imag
    sign_change = np.where(np.diff(np.sign(ImZ)) != 0)[0]

    f0_FEM = np.nan
    if len(sign_change) > 0:
        i = sign_change[0]
        f1, f2 = freqs[i], freqs[i + 1]
        z1, z2 = ImZ[i], ImZ[i + 1]
        f0_FEM = f1 - z1 * (f2 - f1) / (z2 - z1)

        if domain.comm.rank == 0:
            print(f"f0_FEM = {f0_FEM} Hz")
    else:
        if domain.comm.rank == 0:
            print("No sign change found for Im(Zs).")

    np.savez(
        fname,
        config_tag=config_tag,
        freqs=freqs.astype(float),
        a_vec=a_vec.astype(float),
        Zs_vec=Zs_vec.astype(np.complex128),
        r_vec=r_vec.astype(np.complex128),
        avgp=avg_p_vec.astype(np.complex128),
        avgv=avg_v_vec.astype(np.complex128),
        Z0=float(Z0),
        f0=float(f0_FEM),
    )

    if domain.comm.rank == 0:
        print(f"Saved: {fname}")

    plt.figure(figsize=(7, 5))
    plt.plot(freqs, Zs_vec.imag, "o-", linewidth=2, label="Im(Zs)")
    plt.axhline(0, color="k", linestyle="--")

    if len(sign_change) > 0:
        plt.axvline(f0_FEM, color="r", linestyle="--", label=f"f0 = {f0_FEM:.2f} Hz")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Im(Zs) [Pa*s/m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()