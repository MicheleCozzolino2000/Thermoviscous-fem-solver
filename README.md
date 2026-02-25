# Thermoviscous Acoustic FEM Solver (FEniCSx)

Finite element solvers for thermoviscous acoustic propagation developed in Python using FEniCSx.  
The project implements the full linearized Navier–Stokes formulation for viscous and thermal acoustic losses, with validation against analytical models and applications to waveguides and Helmholtz resonators.

## Features

- Full Linearized Navier–Stokes (FLNS) thermoviscous acoustic model  
- Mixed FEM formulation (velocity, pressure, temperature)  
- Implementation in FEniCSx / UFL  
- Support for complex-valued frequency-domain simulations  
- Boundary conditions for:
  - rigid no-slip walls  
  - isothermal/adiabatic thermal conditions  
  - acoustic inlet/outlet forcing  
  - symmetry planes  
- Mesh workflows and post-processing tools  
- Application examples including Helmholtz resonators  


## Requirements

Typical environment:

- Python ≥ 3.10  
- FEniCSx 0.9.0
- PETSc / mpi4py  
- NumPy  
- Matplotlib  
