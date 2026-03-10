"""
Straight cylindrical tube mesh generator for FEniCSx.

Creates a cylindrical 3D mesh with inlet, outlet, wall, and domain
physical groups, then exports it in .msh format.
"""

import sys
import gmsh


def generate_mesh():
    # os.makedirs("Mesh_tube_BL", exist_ok=True)

    # Cylinder geometry
    D = 0.001
    R = D / 2
    L = 5e-2

    name = "A1"
    order2 = False

    gmsh.initialize(sys.argv)
    gmsh.model.add("Tube boundary layer")

    # Global mesh parameters
    gmsh.option.setNumber("Mesh.MeshSizeMax", R / 3)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    # Create cylinder with axis along x
    c1 = gmsh.model.occ.addCylinder(0, 0, 0, L, 0, 0, R)
    gmsh.model.occ.synchronize()

    # Retrieve cylinder boundary surfaces
    surfs = gmsh.model.getBoundary([(3, c1)], oriented=False, recursive=False)
    print("Superfici del cilindro:", surfs)

    # Surface coloring for debug
    # lateral: red
    # outlet: green
    # inlet: blue
    gmsh.model.setColor([(2, 1)], 255, 0, 0)
    gmsh.model.setColor([(2, 2)], 0, 255, 0)
    gmsh.model.setColor([(2, 3)], 0, 0, 255)

    for s in gmsh.model.occ.getEntities(2):
        cm = gmsh.model.occ.getCenterOfMass(s[0], s[1])
        print(f"Surface id={s[1]}, dim={s[0]}, center={cm}")

    inlet_surf = [
        s[1]
        for s in gmsh.model.occ.getEntities(2)
        if gmsh.model.occ.getCenterOfMass(s[0], s[1])[0] < 1e-12
    ]
    outlet_surf = [
        s[1]
        for s in gmsh.model.occ.getEntities(2)
        if abs(gmsh.model.occ.getCenterOfMass(s[0], s[1])[0] - L) < 1e-12
    ]
    wall_surf = [
        s[1]
        for s in gmsh.model.occ.getEntities(2)
        if s[1] not in inlet_surf + outlet_surf
    ]

    print("inlet_surf:", inlet_surf)
    print("outlet_surf:", outlet_surf)
    print("wall_surf:", wall_surf)

    # Physical groups: boundary surfaces
    gmsh.model.addPhysicalGroup(2, inlet_surf, 2)
    gmsh.model.setPhysicalName(2, 2, "inlet")

    gmsh.model.addPhysicalGroup(2, outlet_surf, 3)
    gmsh.model.setPhysicalName(2, 3, "outlet")

    gmsh.model.addPhysicalGroup(2, wall_surf, 4)
    gmsh.model.setPhysicalName(2, 4, "wall")

    gmsh.model.geo.synchronize()

    # Mesh optimization options
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    if order2:
        gmsh.model.mesh.setOrder(2)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    # Physical group: volume
    gmsh.model.addPhysicalGroup(3, [c1], 1)
    gmsh.model.setPhysicalName(3, 1, "domain")

    print("Physical groups:", gmsh.model.getPhysicalGroups())
    for dim, tag in gmsh.model.getPhysicalGroups():
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        print(f"PG dim={dim}, tag={tag}, entities={entities}")


    filename = f"mesh_impedance.msh"
    gmsh.write(filename)
    print(f"Salvata: {filename}")

    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()