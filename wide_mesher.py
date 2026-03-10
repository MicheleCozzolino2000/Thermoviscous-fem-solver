"""
Cylinder Boundary Layer Mesh Generator

Generates a cylindrical mesh with boundary layer using Gmsh
and exports it for FEniCSx.

Output: mesh_wide.msh
"""

import sys
import gmsh
import numpy as np


def generate_mesh():
    gmsh.initialize(sys.argv)
    gmsh.model.add("Tube boundary layer")

    # Geometric parameters
    D = 2e-3
    R = D / 2
    L = 10 * R
    order2 = False

    gmsh.option.setNumber("Mesh.MeshSizeMax", R / 5)

    # Boundary layer parameters
    N_sl = 6
    delta_sl = 1e-4
    k = 1.3

    # Compute single-layer thicknesses
    if k == 1:
        d = [delta_sl / N_sl for _ in range(N_sl)]
    else:
        delta0 = delta_sl * (k - 1) / (k**N_sl - 1)
        d = [delta0 * (k**i) for i in range(N_sl)]

    n = [1] * N_sl
    d = np.cumsum(d)
    d = [-x for x in d]
    print(d)

    # Geometry
    gmsh.model.occ.addCylinder(0, 0, 0, L, 0, 0, R)
    gmsh.model.occ.synchronize()

    gmsh.model.occ.remove(gmsh.model.occ.getEntities(3))
    gmsh.model.occ.remove([(2, 2), (2, 3)])
    gmsh.model.occ.synchronize()

    # Boundary layer extrusion
    gmsh.option.setNumber("Geometry.ExtrudeReturnLateralEntities", 0)
    e = gmsh.model.geo.extrudeBoundaryLayer(gmsh.model.getEntities(2), n, d, False)

    top_ent = [s for s in e if s[0] == 2]
    top_surf = [s[1] for s in top_ent]
    gmsh.model.geo.synchronize()

    # Rebuild top surfaces
    bnd_ent = gmsh.model.getBoundary(top_ent)
    bnd_curv = [c[1] for c in bnd_ent]
    loops = gmsh.model.geo.addCurveLoops(bnd_curv)

    for l in loops:
        top_surf.append(gmsh.model.geo.addPlaneSurface([l]))

    gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(top_surf)])
    gmsh.model.geo.synchronize()

    surfaces = gmsh.model.getEntities(2)
    print(surfaces)
    print(gmsh.model.getEntities(2))

    volumes = [v[1] for v in gmsh.model.getEntities(3)]
    print("Volumi trovati:", volumes)

    # Physical groups
    gmsh.model.addPhysicalGroup(3, volumes, 1, "domain")
    gmsh.model.addPhysicalGroup(2, [20, 27], 2, "inlet")
    gmsh.model.addPhysicalGroup(2, [1], 4, "wall")
    gmsh.model.addPhysicalGroup(2, [12, 26], 3, "outlet")

    # Mesh generation
    if order2:
        gmsh.model.mesh.setOrder(2)

    gmsh.model.mesh.generate(3)
    gmsh.model.occ.synchronize()

    # Optional GUI visualization
    # gmsh.fltk.run()

    # Save mesh
    filename = "mesh_wide.msh"
    gmsh.write(filename)
    print(f"Salvata: {filename}")

    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()