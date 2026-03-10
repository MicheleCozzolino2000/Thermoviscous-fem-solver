"""
Generate a 3D sector mesh for a coaxial Helmholtz resonator.

The geometry consists of three aligned volumes:
- guide
- neck
- cavity

The script:
1. builds the three sector volumes,
2. fragments them to obtain conforming interfaces,
3. identifies boundary surfaces,
4. assigns physical groups,
5. applies a Distance + Threshold mesh size field around the neck,
6. generates and saves the mesh.
"""

import sys

import gmsh
import numpy as np


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    name = "RH_sector"
    subname = "Threshold"
    theta_deg = 10
    theta = np.deg2rad(theta_deg)

    # -----------------------------
    # Geometry
    # -----------------------------
    Rc = 0.101   # guide and cavity radius
    Lg = Rc      # guide length along the x-axis
    Rn = 0.0056  # neck radius
    Ln = 0.025   # neck length along the x-axis
    Lc = 0.150   # cavity length along the x-axis

    x0, y0, z0 = 0.0, 0.0, 0.0

    # Axial reference positions
    x_in = x0
    x_gn = x0 + Lg
    x_nc = x0 + Lg + Ln
    x_out = x0 + Lg + Ln + Lc

    # -----------------------------
    # Geometric tolerances
    # -----------------------------
    tol_x = 1e-6
    tol_z = 1e-6
    tol_pl = 5e-6
    tol_r = 1e-5

    tan_theta = np.tan(theta)

    gmsh.initialize(sys.argv)
    gmsh.model.add("Helmholtz_sector_coaxial")
    occ = gmsh.model.occ

    # -----------------------------
    # Mesh options
    # -----------------------------
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

    def extruded_sector_volume(x_base: float, R: float, Lx: float, theta: float):
        """
        Create a 3D sector volume by extruding a circular sector along the x-axis.
        """
        # Sector center
        pc = occ.addPoint(x_base, y0, z0)

        # Sector boundary points at angles 0 and theta
        p1 = occ.addPoint(x_base, y0 + R, z0)
        p2 = occ.addPoint(
            x_base,
            y0 + R * np.cos(theta),
            z0 + R * np.sin(theta),
        )

        l1 = occ.addLine(pc, p1)
        arc = occ.addCircleArc(p1, pc, p2)
        l2 = occ.addLine(p2, pc)

        cl = occ.addCurveLoop([l1, arc, l2])
        s0 = occ.addPlaneSurface([cl])

        ext = occ.extrude([(2, s0)], Lx, 0.0, 0.0)
        vol_tags = [tag for (dim, tag) in ext if dim == 3]

        if len(vol_tags) != 1:
            raise RuntimeError(f"Sector extrusion failed: vol_tags={vol_tags}")

        return vol_tags[0], ext

    def vol_props(vtag: int):
        """
        Return xmin, xmax, and an approximate radial extent from the volume bounding box.
        """
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, vtag)
        rbb = max(abs(ymin - y0), abs(ymax - y0), abs(zmin - z0), abs(zmax - z0))
        return xmin, xmax, rbb

    def bbox(dim, tag):
        return gmsh.model.getBoundingBox(dim, tag)

    def r_bbox_yz_from_bbox(bb):
        _, ymin, zmin, _, ymax, zmax = bb
        return max(abs(ymin - y0), abs(ymax - y0), abs(zmin - z0), abs(zmax - z0))

    def is_plane_x(bb, xref, tol):
        xmin, _, _, xmax, _, _ = bb
        return (abs(xmin - xref) < tol) and (abs(xmax - xref) < tol)

    def is_plane_z(bb, zref, tol):
        _, _, zmin, _, _, zmax = bb
        return (abs(zmin - zref) < tol) and (abs(zmax - zref) < tol)

    def is_plane_theta(bb, tan_theta, tol):
        """
        Check whether the surface lies on the plane z = y * tan(theta).
        """
        _, ymin, zmin, _, ymax, zmax = bb
        return (abs(zmin - ymin * tan_theta) < tol) and (abs(zmax - ymax * tan_theta) < tol)

    # -----------------------------
    # Build the three main volumes
    # -----------------------------
    guide_vol, _ = extruded_sector_volume(x0, Rc, Lg, theta)
    neck_vol, _ = extruded_sector_volume(x0 + Lg, Rn, Ln, theta)
    cav_vol, _ = extruded_sector_volume(x0 + Lg + Ln, Rc, Lc, theta)

    occ.synchronize()

    # Fragment the volumes to obtain conforming interfaces and separate surfaces
    occ.fragment([(3, guide_vol), (3, neck_vol), (3, cav_vol)], [])
    occ.synchronize()

    # -----------------------------
    # Identify final volumes
    # -----------------------------
    vols = gmsh.model.getEntities(3)
    vol_tags = [v[1] for v in vols]

    guide_candidates = []
    neck_candidates = []
    cav_candidates = []

    for vt in vol_tags:
        xmin, xmax, rbb = vol_props(vt)

        if abs(rbb - Rn) < 1e-6:
            neck_candidates.append(vt)
            continue

        if abs(rbb - Rc) < 1e-6:
            if xmax <= x_gn + 1e-6:
                guide_candidates.append(vt)
            elif xmin >= x_nc - 1e-6:
                cav_candidates.append(vt)

    if len(guide_candidates) != 1 or len(neck_candidates) != 1 or len(cav_candidates) != 1:
        print("Volume tags:", vol_tags)
        for vt in vol_tags:
            xmin, xmax, rbb = vol_props(vt)
            print(f"vt={vt}  xmin={xmin:.6e}  xmax={xmax:.6e}  rbb={rbb:.6e}")
        raise RuntimeError("Unable to identify guide, neck, and cavity volumes.")

    guide_vol = guide_candidates[0]
    neck_vol = neck_candidates[0]
    cav_vol = cav_candidates[0]

    print(f"guide_vol = {guide_vol}, neck_vol = {neck_vol}, cav_vol = {cav_vol}")

    vols = gmsh.model.getEntities(3)
    vol_tags = [v[1] for v in vols]

    if not vol_tags:
        raise RuntimeError("No final volumes were found after fragmentation.")

    # -----------------------------
    # Classify boundary surfaces
    # -----------------------------
    bnd_all = gmsh.model.getBoundary(
        [(3, vt) for vt in vol_tags], oriented=False, recursive=False
    )
    surf_tags = sorted({tag for (dim, tag) in bnd_all if dim == 2})

    bnd_neck = gmsh.model.getBoundary([(3, neck_vol)], oriented=False, recursive=False)
    surf_tags_neck = sorted({tag for (dim, tag) in bnd_neck if dim == 2})

    inlet_surf = []
    inlet_neck_surf = []
    neck_mouth_surf = []
    sym_yx_surf = []
    sym_th_surf = []
    wall_surf = []
    wall_guide_surf = []

    # Identify inlet_neck and neck_mouth from the neck boundary
    for sid in surf_tags_neck:
        bb = bbox(2, sid)
        rbb = r_bbox_yz_from_bbox(bb)

        if is_plane_x(bb, x_gn, tol_x) and abs(rbb - Rn) < tol_r:
            inlet_neck_surf.append(sid)

        if is_plane_x(bb, x_nc, tol_x) and abs(rbb - Rn) < tol_r:
            neck_mouth_surf.append(sid)

    print("inlet_neck_surf:", inlet_neck_surf)
    print("neck_mouth_surf:", neck_mouth_surf)

    # Classify the full external boundary
    for sid in surf_tags:
        bb = bbox(2, sid)
        rbb = r_bbox_yz_from_bbox(bb)

        # Guide inlet at x = x_in
        if is_plane_x(bb, x_in, tol_x) and abs(rbb - Rc) < tol_r:
            inlet_surf.append(sid)
            continue

        # Guide section just upstream of the neck
        if is_plane_x(bb, x_gn, tol_x) and abs(rbb - Rc) < tol_r:
            wall_guide_surf.append(sid)
            continue

        # Symmetry planes
        if is_plane_z(bb, 0.0, tol_z):
            sym_yx_surf.append(sid)
            continue

        if is_plane_theta(bb, tan_theta, tol_pl):
            sym_th_surf.append(sid)
            continue

        # Remaining external surfaces
        wall_surf.append(sid)

    print("Boundary surfaces:", surf_tags)
    print("inlet_surf      :", inlet_surf)
    print("inlet_neck_surf :", inlet_neck_surf)
    print("neck_mouth_surf :", neck_mouth_surf)
    print("sym_yx_surf     :", sym_yx_surf)
    print("sym_th_surf     :", sym_th_surf)
    print("wall_surf       :", wall_surf)
    print("wall_guide_surf :", wall_guide_surf)

    if (
        not inlet_surf
        or not inlet_neck_surf
        or not neck_mouth_surf
        or not wall_surf
        or not sym_yx_surf
        or not sym_th_surf
    ):
        print("\nDebug output near axial planes:")
        for sid in surf_tags:
            bb = bbox(2, sid)
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, sid)
            if (
                abs(cx - x_in) < 1e-4
                or abs(cx - x_gn) < 1e-4
                or abs(cx - x_nc) < 1e-4
                or abs(cx - x_out) < 1e-4
            ):
                print(
                    f"sid={sid}  cx={cx:.6e}  "
                    f"bb_x=[{bb[0]:.6e}, {bb[3]:.6e}]  "
                    f"rbb={r_bbox_yz_from_bbox(bb):.6e}"
                )
        raise RuntimeError("Surface classification failed.")

    # -----------------------------
    # Physical groups
    # -----------------------------
    gmsh.model.addPhysicalGroup(3, vol_tags, 1, "domain")
    gmsh.model.addPhysicalGroup(2, inlet_surf, 2, "inlet")
    gmsh.model.addPhysicalGroup(2, inlet_neck_surf, 3, "inlet_neck")
    gmsh.model.addPhysicalGroup(2, neck_mouth_surf, 4, "neck_mouth")
    gmsh.model.addPhysicalGroup(2, wall_surf, 5, "wall")
    gmsh.model.addPhysicalGroup(2, sym_yx_surf, 6, "sym_yx")
    gmsh.model.addPhysicalGroup(2, sym_th_surf, 7, "sym_theta")

    if wall_guide_surf:
        gmsh.model.addPhysicalGroup(2, wall_guide_surf, 8, "wall_guide")
    else:
        print("Warning: wall_guide_surf is empty.")

    # -----------------------------
    # Distance + Threshold mesh field
    # -----------------------------
    neck_bnd = gmsh.model.getBoundary([(3, neck_vol)], oriented=False, recursive=False)
    neck_surf_all = sorted({tag for (dim, tag) in neck_bnd if dim == 2})

    if not neck_surf_all:
        raise RuntimeError("No boundary surfaces were found for the neck volume.")

    # Mesh size parameters
    h_min = Rn / 6
    h_max = Rc / 6

    # Transition distances
    d_min = 2.0 * Rn
    d_max = 15.0 * Rn

    # Distance field from all neck boundary surfaces
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", neck_surf_all)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

    # Threshold field: map distance to mesh size
    f_th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMin", h_min)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMax", h_max)
    gmsh.model.mesh.field.setNumber(f_th, "DistMin", d_min)
    gmsh.model.mesh.field.setNumber(f_th, "DistMax", d_max)

    gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)

    # -----------------------------
    # Mesh generation
    # -----------------------------
    gmsh.model.mesh.generate(3)

    msh_file = f"{name}_{int(theta_deg)}deg_{subname}.msh"
    gmsh.write(msh_file)
    print(f"Saved mesh: {msh_file}")

    gmsh.finalize()


if __name__ == "__main__":
    main()