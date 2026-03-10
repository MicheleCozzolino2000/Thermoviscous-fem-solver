"""
Thermoviscous Acoustic Solver for Cylindrical Tubes
Solves the linearized thermoviscous acoustic problem in the frequency domain.
Loads either mesh_narrow.msh or mesh_wide.msh dynamically.
"""

import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element, mixed_element
from dolfinx import fem, io, geometry, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import Function, dirichletbc

class MicrophonePressure:
    """Utility class to extract fields at specific spatial points."""
    def __init__(self, domain, microphone_position):
        self._domain = domain
        self._position = np.asarray(
            microphone_position, dtype=domain.geometry.x.dtype
        ).reshape(3, -1)
        self._local_cells, self._local_position = self.compute_local_microphones()

    def compute_local_microphones(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]]:
        points = self._position.T
        bb_tree = geometry.bb_tree(self._domain, self._domain.topology.dim)
        cells = []
        points_on_proc = []
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(self._domain, cell_candidates, points)

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        return np.asarray(cells, dtype=np.int32), np.asarray(points_on_proc, dtype=self._domain.geometry.x.dtype)

    def listen(self, p_a: fem.Function, recompute_collisions: bool = False) -> npt.NDArray[np.complexfloating]:
        if recompute_collisions:
            self._local_cells, self._local_position = self.compute_local_microphones()
        if len(self._local_cells) > 0:
            return p_a.eval(self._local_position, self._local_cells)
        else:
            return np.zeros(0, dtype=default_scalar_type)


class ThermoviscousSolver:
    """Finite element solver for the coupled thermoviscous acoustic problem."""
    def __init__(self, domain, facet_tags, inlet_tag, outlet_tag, wall_tag, freq, properties):
        self.domain = domain
        self.facet_tags = facet_tags
        self.inlet_tag = inlet_tag
        self.outlet_tag = outlet_tag
        self.wall_tag = wall_tag
        self.freq = freq

        # Unpack physical properties
        self.rho = properties["rho0"]
        self.mu = properties["mu_air"]
        self.lambda_ = properties["lambda_air"]
        self.cp = properties["cp"]
        self.ka = properties["k_air"]
        self.T0 = properties["T"]
        self.p0 = properties["p0"]
        self.c = properties["c"]

        # Define finite elements (P2 for velocity/temperature, P1 for pressure)
        cell = domain.basix_cell()
        P2_vec = element("Lagrange", cell, degree=2, shape=(domain.geometry.dim,))
        P2_sca = element("Lagrange", cell, degree=2)
        P1 = element("Lagrange", cell, degree=1)

        mixed = mixed_element([P2_vec, P2_sca, P1])
        
        # Mixed function space and sub-spaces
        self.Me = fem.functionspace(domain, mixed)
        self.Ve = self.Me.sub(0)
        self.Te = self.Me.sub(1)
        self.Pe = self.Me.sub(2)

    def solve(self) -> fem.Function:
        omega = 2 * np.pi * self.freq
        
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

        # Bilinear form (weak formulation)
        a = (
            1j * omega * self.rho * ufl.inner(v, w) * dx
            + 2 * self.mu * ufl.inner(ufl.sym(ufl.grad(v)), ufl.sym(ufl.grad(w))) * dx
            + self.lambda_ * ufl.inner(ufl.div(v), ufl.div(w)) * dx
            - ufl.inner(p, ufl.div(w)) * dx
            - 1j * omega * self.cp * self.rho * (1 / self.T0) * ufl.inner(T, s) * dx
            - self.ka * (1 / self.T0) * ufl.inner(ufl.grad(T), ufl.grad(s)) * dx
            + 1j * omega * (1 / self.T0) * ufl.inner(p, s) * dx
            - ufl.inner(ufl.div(v), q) * dx
            + 1j * omega * (1 / self.T0) * ufl.inner(T, q) * dx
            - 1j * omega * (1 / self.p0) * ufl.inner(p, q) * dx
        )
        
        # Boundary excitations and linear form
        p_in = Function(Pe_collapsed)
        p_in.x.array[:] = 1.0  # 1 Pa input pressure
        p_out = Function(Pe_collapsed)
        p_out.x.array[:] = 0.0

        f = Function(Ve_collapsed)       # Volumetric force
        Q_source = Function(Te_collapsed) # Thermal source

        L = (
            ufl.inner(f, w) * dx
            - ufl.inner(Q_source, s) * dx 
            - ufl.inner(p_in * n, w) * ds(self.inlet_tag)
            - ufl.inner(p_out * n, w) * ds(self.outlet_tag)
        )

        # Boundary Conditions Setup
        v_wall = Function(Ve_collapsed)
        v_wall.x.array[:] = 0.0 + 0.0j

        T_zero = Function(Te_collapsed)
        T_zero.x.array[:] = 0.0

        wall_facets = self.facet_tags.find(self.wall_tag)

        # No-slip and isothermal walls
        dofs_v_side = fem.locate_dofs_topological((self.Ve, Ve_collapsed), self.domain.topology.dim-1, wall_facets)
        bc_v_side = dirichletbc(v_wall, dofs_v_side, self.Ve)

        dofs_T_side = fem.locate_dofs_topological((self.Te, Te_collapsed), self.domain.topology.dim-1, wall_facets)
        bc_T_side = dirichletbc(T_zero, dofs_T_side, self.Te)

        bcs = [bc_v_side, bc_T_side]

        # Solve linear problem
        U_sol = Function(self.Me)
        problem = LinearProblem(
            a, L, bcs=bcs, u=U_sol,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
        )
        problem.solve()
        
        v_sol, T_sol, p_sol = U_sol.split()
        return v_sol, T_sol, p_sol


def compute_air_properties(Tc, p_0, hum, xc):
    """Computes physical properties of air based on empirical correlations."""
    T = 273.15 + Tc
    Ra = 287.05
    psv = np.exp(1.2378847e-5*T**2 - 1.9121316e-2*T + 33.93711047 - 6.3431645e3*T**(-1))
    f = 1.00062 + 3.14e-8*p_0 + 5.6e-7*Tc**2
    xw = (hum/100) * (psv/p_0) * f
    Z = 1 - (p_0/T)*(1.58123e-6 - 2.9331e-8*Tc + 1.1043e-10*Tc**2 + (5.707e-6 - 2.051e-8*Tc)*xw + (1.9898e-4 - 2.376e-6*Tc)*xw**2) + (p_0/T)**2 * (1.83e-11 - 0.765e-8*xw**2)
    rho0 = (3.48349 + 1.44*(xc-0.0004))*10**(-3) * (p_0/(Z*T)) * (1 - 0.3780*xw)
    c = 331.5024 + 0.603055*Tc - 0.000528*Tc**2 + (51.471935 + 0.1495874*Tc - 0.000782*Tc**2)*xw + (-1.82e-7 + 3.73e-8*Tc - 2.93e-10 *Tc**2)*p_0 + (-85.20931 - 0.228525*Tc + 5.91e-5 * Tc**2)*xc - 2.835149*xw**2 - 2.5e-13*p_0**2 + 29.179762*xc**2 + 0.000486*xw*p_0*xc
    mu_air = (84.986 + 7*T + (113.157 - 1*T)*xw - 3.7501e-3 * T**2 - 100.015*xw**2)*10**(-8)
    cp1 = 0.251625 - 9.2525e-5*T + 2.1334e-7*T**2 - 1.0043e-10*T**3 + (0.12477 - 2.283e-5*T + 1.267e-7*T**2)*xw + (0.01116 + 4.61e-6*T + 1.74e-8*T**2)*xw**2
    cp = cp1*4.1868*10**3
    eta = 0.60*mu_air
    lambda_air = eta - (2/3) * mu_air 
    ni_air = mu_air/rho0
    k_air = 1.30030e-3 +9.36766e-5*T-4.44247e-8*T**2+2.31716e-11 *T**3-6.59976e-15 * T**4
    
    return {
        "T": T, "p0": p_0, "rho0": rho0, "c": c, "mu_air": mu_air, "lambda_air": lambda_air,
        "cp": cp, "k_air": k_air, "ni_air": ni_air, "eta": eta
    }


def extract_field_along_axis(mic, field_solution, scaling_factor, axis_vals=None, component=None):
    """Utility to extract data along an axis using the Microphone class."""
    data = mic.listen(field_solution)
    if component is not None:
        data = data[:, component]
    data = data / scaling_factor
    if axis_vals is not None:
        n = min(len(axis_vals), len(data))
        data = data[:n]
    return {"real": data.real, "imag": data.imag, "abs": np.abs(data), "arg": np.angle(data)}


def plot_field(axis_vals, field_dict, axis_label, field_label, lim, title_prefix, filename):
    """Utility to plot real, imag, and absolute values of a field."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Re", "Im", "Abs"]
    for ax, key, t in zip(axs, ["real", "imag", "abs"], titles):
        y = np.array(field_dict[key])
        n = min(len(axis_vals), len(y))
        ax.plot(axis_vals[:n], y[:n], "-")
        ax.set_xlabel(axis_label)
        ax.set_ylabel(field_label)
        ax.set_title(f"{title_prefix} ({t})")
        ax.grid(True)
        ax.set_ylim(bottom=0, top=lim)
    
    fig.tight_layout()
    os.makedirs("ThermoViscous_tube", exist_ok=True)
    fig.savefig(os.path.join("ThermoViscous_tube", filename))
    plt.close()


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    MESH_NAME = "mesh_wide.msh" # Change to "mesh_wide.msh" as needed
    
    # Automatically infer testing frequency and radius from mesh name
    if "wide" in MESH_NAME:
        freq_test = 10000
        R_tube = 1e-3
    elif "narrow" in MESH_NAME:
        freq_test = 100
        R_tube = 1e-4
    else:
        raise ValueError("Invalid mesh name. Must contain 'wide' or 'narrow'.")

    L_tube = 10 * R_tube

    # Physical parameters
    Tc, p_0, hum, xc = 20, 101325, 20, 0.0004
    air_props = compute_air_properties(Tc, p_0, hum, xc)

    # Output directories
    os.makedirs("ThermoViscous_tube", exist_ok=True)

    # -----------------------------
    # Load Mesh
    # -----------------------------
    domain, cell_tags, facet_tags = io.gmshio.read_from_msh(MESH_NAME, MPI.COMM_WORLD, 0, gdim=3)

    inlet_tag, outlet_tag, wall_tag = 2, 3, 4

    # -----------------------------
    # Solve Problem
    # -----------------------------
    solver = ThermoviscousSolver(
        domain, facet_tags, inlet_tag, outlet_tag, wall_tag, freq_test, air_props
    )
    v_sol, T_sol, p_sol = solver.solve()

    # -----------------------------
    # Post-Processing
    # -----------------------------
    x_pos = L_tube / 2
    y_vals = np.linspace(0, R_tube, 200)
    
    # Microphone array setup along the Y-axis at L/2
    mic_pos_y_axis = np.vstack([np.full_like(y_vals, x_pos), y_vals, np.zeros_like(y_vals)])
    mic_y_axis = MicrophonePressure(domain, mic_pos_y_axis)

    # Extract fields
    p_axis_dict = extract_field_along_axis(mic_y_axis, p_sol, air_props["p0"])
    T_axis_dict = extract_field_along_axis(mic_y_axis, T_sol, air_props["T"])

    # Compute complex density
    rho_complex = air_props["rho0"] * (
        (p_axis_dict["real"] + 1j*p_axis_dict["imag"]) -
        (T_axis_dict["real"] + 1j*T_axis_dict["imag"])
    )
    rho_axis_dict = {"real": rho_complex.real, "imag": rho_complex.imag, "abs": np.abs(rho_complex)}

    # Plot Fields
    tag = MESH_NAME.split('.')[0]
    plot_field(y_vals/R_tube, p_axis_dict, "r/R [-]", "p/$p_0$ [-]", 1.2 * np.max(p_axis_dict["abs"]), "Pressure", f"pressure_plot_{tag}.png")
    plot_field(y_vals/R_tube, T_axis_dict, "r/R [-]", "T/$T_0$ [-]", 1.2 * np.max(T_axis_dict["abs"]), "Temperature", f"temperature_plot_{tag}.png")
    plot_field(y_vals/R_tube, rho_axis_dict, "r/R [-]", r"$\rho / \rho_0$ [-]", 1.2 * np.max(rho_axis_dict["abs"]), "Density", f"density_plot_{tag}.png")

    if domain.comm.rank == 0:
        print(f"Analysis complete for {MESH_NAME}.")
        print(f"Max |p/p0| = {np.max(p_axis_dict['abs']):.6e}")
        print(f"Max |T/T0| = {np.max(T_axis_dict['abs']):.6e}")
        print(f"Max |rho/rho0| = {np.max(rho_axis_dict['abs']):.6e}")

if __name__ == "__main__":
    main()