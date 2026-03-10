"""
Narrow Cylinder Mesh Generator
Generates a narrow cylindrical mesh using Gmsh and exports it for FEniCSx.
Output: mesh_narrow.msh
"""

import sys
import gmsh

def generate_narrow_mesh():
    # Geometric parameters
    D = 2e-4            # Diameter [m]
    R = D / 2           # Radius [m]
    L = 10 * R          # Length [m]

    gmsh.initialize(sys.argv)
    gmsh.model.add("Tube_Narrow")

    # Global mesh parameters
    gmsh.option.setNumber("Mesh.MeshSizeMax", R / 5)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)

    # Create cylinder along the x-axis
    c1 = gmsh.model.occ.addCylinder(0, 0, 0, L, 0, 0, R)
    gmsh.model.occ.synchronize()

    # Identify surfaces based on their center of mass
    surfaces = gmsh.model.occ.getEntities(2)
    inlet_surf = [s[1] for s in surfaces if gmsh.model.occ.getCenterOfMass(s[0], s[1])[0] < 1e-12]
    outlet_surf = [s[1] for s in surfaces if abs(gmsh.model.occ.getCenterOfMass(s[0], s[1])[0] - L) < 1e-12]
    wall_surf = [s[1] for s in surfaces if s[1] not in inlet_surf + outlet_surf]

    # Define Physical Groups (Tags consistent with the solver: Inlet=2, Outlet=3, Wall=4)
    gmsh.model.addPhysicalGroup(2, inlet_surf, 2, "inlet")
    gmsh.model.addPhysicalGroup(2, outlet_surf, 3, "outlet")
    gmsh.model.addPhysicalGroup(2, wall_surf, 4, "wall")
    
    # Volume domain (Tag=1)
    gmsh.model.addPhysicalGroup(3, [c1], 1, "domain")

    gmsh.model.geo.synchronize()

    # Mesh optimization options
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    gmsh.model.occ.synchronize()

    # Save to file
    output_filename = "mesh_narrow.msh"
    gmsh.write(output_filename)
    print(f"Mesh successfully generated and saved to: {output_filename}")

    gmsh.finalize()

if __name__ == "__main__":
    generate_narrow_mesh()