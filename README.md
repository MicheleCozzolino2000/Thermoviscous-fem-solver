# Thermoviscous Acoustics FEM Solver

This repository contains a finite element implementation of a **thermoviscous acoustic solver** based on the **linearized Navier–Stokes equations (FLNS)** in the frequency domain, implemented using **FEniCSx**.  
The solver is designed to model acoustic wave propagation in fluids while accounting for **viscous and thermal dissipation**, which become important in confined geometries such as narrow ducts or resonant acoustic devices.

The code was developed to investigate thermoviscous effects in cylindrical ducts and resonant acoustic systems, and includes several validation and application cases.

---

# Overview

The repository includes scripts for:

- generating computational meshes using **Gmsh**
- solving the **coupled thermoviscous acoustic equations** with FEM
- validating the numerical model against **analytical thermoviscous solutions**
- simulating a **Helmholtz resonator** including viscous and thermal losses.

The scripts are organized into three main groups of applications:

1. **Validation using the classical Stinson cylindrical tube cases**
2. **Impedance validation in cylindrical ducts**
3. **Simulation of a thermoviscous Helmholtz resonator**

---

# 1. Stinson Benchmark Validation

The first validation case reproduces the classical thermoviscous wave propagation problem in cylindrical ducts described by **Stinson**. These configurations are widely used to validate numerical solvers for thermoviscous acoustics.

Two limiting cases are considered:

- **Narrow tube**: viscous and thermal boundary layers dominate the entire cross-section.
- **Wide tube**: boundary layers are thin compared to the tube radius.

Mesh generation for these two cases is handled by:

- `narrow_mesher.py`  
  Generates the mesh for the narrow cylindrical tube configuration and exports it as `mesh_narrow.msh`.

- `wide_mesher.py`  
  Generates the mesh for the wide cylindrical tube configuration with a refined **boundary layer mesh near the wall**, exported as `mesh_wide.msh`.

The thermoviscous problem is solved using:

- `Solver_stinson.py`

This script implements a **finite element solver for the coupled velocity–temperature–pressure system** derived from the linearized Navier–Stokes equations. The solver operates in the frequency domain and allows extraction of acoustic quantities such as pressure and velocity along the tube axis.

These cases are used as **benchmark problems** to verify that the numerical solver correctly reproduces the thermoviscous acoustic behavior predicted by analytical models.

---

# 2. Tube Impedance Validation

A second validation step is performed by comparing the **surface impedance obtained from the FEM simulation** with the **analytical thermoviscous impedance of a cylindrical tube** derived from Stinson’s formulation.

The mesh for this configuration is generated with:

- `mesh_impedance_validation.py`

This script creates a straight cylindrical tube mesh and defines the required **physical groups** (inlet, outlet, wall, and domain) for the FEM solver. The mesh is exported as `mesh_impedance.msh`.

The impedance validation is performed using:

- `tube_impedance_validation.py`

This script:

1. loads the cylindrical mesh
2. computes the thermophysical properties of air
3. performs a **frequency sweep**
4. solves the thermoviscous FEM problem for each frequency
5. computes the **surface impedance at the inlet**
6. compares the FEM results with the **analytical thermoviscous impedance**.

This comparison provides an additional verification that the FEM implementation correctly captures the frequency-dependent dissipative behavior of thermoviscous acoustics.

---

# 3. Helmholtz Resonator Simulation

The final part of the repository focuses on the simulation of a **coaxial Helmholtz resonator**, a classical acoustic device used for low-frequency sound absorption.

The geometry consists of three main regions:

- a **waveguide**
- a **neck**
- a **cavity**

Because the geometry is axisymmetric, the simulation is performed on a **sector of the full domain** in order to reduce the computational cost.

The mesh for this configuration is generated using:

- `mesh_RH_sector.py`

This script constructs the geometry of the waveguide, neck and cavity, fragments the volumes to obtain conforming interfaces, identifies the boundary surfaces and assigns the appropriate physical groups. A **local mesh refinement is applied near the neck**, where viscous and thermal losses are most significant.

The thermoviscous simulation of the resonator is performed using:

- `Solver_RH_MPC.py`

This solver includes:

- the coupled velocity–temperature–pressure thermoviscous formulation
- no-slip and isothermal boundary conditions at solid walls
- symmetry boundary conditions
- **multi-point constraints (MPC)** to enforce sector periodicity.

The solver computes quantities such as:

- surface impedance
- reflection coefficient
- absorption coefficient.

---

# Analytical Resonance Estimate

An analytical estimate of the resonance frequency of the Helmholtz resonator is provided by:

- `resonance_estimate.py`

This script computes the resonance frequency using an **equivalent acoustic model** that includes thermoviscous effects using an equivalent acoustic model based on::

- equivalent density
- equivalent bulk modulus
- an **Ingard end correction** for the neck length.

The estimated resonance frequency can be used as a reference when selecting the frequency range for FEM simulations.

---

# Dependencies

The scripts rely on the following Python libraries:
dolfinx
mpi4py
petsc4py
gmsh
numpy
scipy
matplotlib


The solver is intended to run in a **FEniCSx environment with PETSc support**.

---

# Author

Michele Cozzolino  
MSc Mechanical Engineering

Development of an **open-source finite element solver for thermoviscous acoustics** aimed at modeling dissipative acoustic phenomena in confined fluid domains.

## Credits

Special thanks to the following people for their support and guidance during the development of this project:

- [Elio Di Giulio](https://github.com/eliodigiulio)
- [Antonio Baiano Svizzero](https://github.com/bayswiss)

# References

The implementation and validation cases included in this repository are based on classical results in thermoviscous acoustics and on open-source finite element tools.

[1] W. R. Kampinga, *Viscothermal Acoustics Using Finite Elements: Analysis Tools for Engineers*, PhD thesis, University of Twente, 2010.

[2] M. R. Stinson, “The propagation of plane sound waves in narrow and wide circular tubes, and generalization to uniform tubes of arbitrary cross-sectional shape,” *Journal of the Acoustical Society of America*, vol. 89, pp. 550–558, 1991.

[3] G. Kirchhoff, “Ueber den Einfluss der Wärmeleitung in einem Gase auf die Schallbewegung,” *Annalen der Physik*, 1868.

[4] A. D. Pierce, *Acoustics: An Introduction to Its Physical Principles and Applications*, Acoustical Society of America, 1991.

[5] U. Ingard, “On the theory and design of acoustic resonators,” *Journal of the Acoustical Society of America*, vol. 25, pp. 1037–1061, 1953.

[6] C. Geuzaine and J.-F. Remacle, *Gmsh Reference Manual*, Version 4.15.1, 2024.  
https://gmsh.info

[7] C. Geuzaine, “tube_boundary_layer.py: Gmsh Python API example.”  
https://gitlab.onelab.info/gmsh/gmsh

[8] J. S. Dokken, *dolfinx-mpc: Multi-point constraints with FEniCSx*.  
https://jsdokken.com/dolfinx_mpc

[9] J. S. Dokken et al., “Stokes demo in dolfinx_mpc.”  
https://github.com/jorgensd/dolfinx_mpc

[10] FEniCS Project, *dolfinx documentation*.  
https://docs.fenicsproject.org

[11] A. B. Svizzero and J. S. Dokken, “Helmholtz equation implementation in FEniCSx.”  
https://jsdokken.com/dolfinx-tutorial

[12] K. Rasmussen, *Calculation methods for the physical properties of air used in the calibration of microphones*, Technical University of Denmark, 1997.

[13] R. Mastrullo, P. Mazzei, V. Naso, R. Vanoli, *Fondamenti di trasmissione del calore*, Liguori Editore, 1991.

[14] P. M. Morse and K. U. Ingard, *Theoretical Acoustics*, Princeton University Press, 1986.
