"""Write the egg and carton-cup meshes for pack-eggs-small-end-down (metres, origin at each part's bounding-box centre):
egg.obj: Hügelschäffer egg, L = 57 mm, B = 44 mm, w = 3 mm, revolved about local z with the small end at local +z;
  bounding box 44 x 44 x 57 mm, strictly convex.
carton.obj: 60 x 60 x 35 mm moulded-pulp block with one conical cup cut from the top: top opening diameter 46 mm,
  bottom seat diameter 24 mm, depth 25 mm (floor 10 mm thick).
"""

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

EGG_L_MM = 57.0
EGG_B_MM = 44.0
EGG_W_MM = 3.0


def egg_radius_mm(s_mm: float) -> float:
    """Hügelschäffer egg radius (mm) at axial coordinate s_mm in [-L/2, +L/2]; w = +3 mm puts the small end at +z."""
    num = max(0.0, EGG_L_MM**2 - 4.0 * s_mm**2)
    den = EGG_L_MM**2 + 8.0 * EGG_W_MM * s_mm + 4.0 * EGG_W_MM**2
    return 0.5 * EGG_B_MM * math.sqrt(num / den)


def egg(n_theta: int = 32) -> None:
    # 18 axial rings including s = -w = -3.0 mm explicitly so the widest ring hits r = B/2 = 22.0 mm exactly.
    s_rings = sorted({
        -27.5, -25.5, -23.0, -20.0, -16.0, -12.0, -8.0, -5.0, -3.0,
        0.0, 3.0, 7.0, 11.0, 15.0, 20.0, 23.0, 25.5, 27.5,
    })
    angles = [2.0 * math.pi * i / n_theta for i in range(n_theta)]
    pts: list[tuple[float, float, float]] = [(0.0, 0.0, -EGG_L_MM / 2.0)]  # index 0: bottom pole (-z, blunt end)
    ring_starts: list[int] = []
    for s in s_rings:
        r = egg_radius_mm(s)
        ring_starts.append(len(pts))
        for a in angles:
            pts.append((r * math.cos(a), r * math.sin(a), s))
    top_pole = len(pts)
    pts.append((0.0, 0.0, +EGG_L_MM / 2.0))  # small-end pole (+z)

    faces: list[tuple[int, int, int]] = []
    # Bottom cap (blunt end at -z, outward normal -z)
    r0 = ring_starts[0]
    for i in range(n_theta):
        j = (i + 1) % n_theta
        faces.append((0, r0 + j, r0 + i))
    # Side quads between consecutive rings
    for k in range(len(ring_starts) - 1):
        a0, b0 = ring_starts[k], ring_starts[k + 1]
        for i in range(n_theta):
            j = (i + 1) % n_theta
            faces.append((a0 + i, a0 + j, b0 + j))
            faces.append((a0 + i, b0 + j, b0 + i))
    # Top cap (small end at +z, outward normal +z)
    rn = ring_starts[-1]
    for i in range(n_theta):
        j = (i + 1) % n_theta
        faces.append((rn + i, rn + j, top_pole))

    with open(os.path.join(HERE, "egg.obj"), "w") as f:
        print("o egg", file=f)
        for x, y, z in pts:
            print(f"v {x / 1000.0:.6f} {y / 1000.0:.6f} {z / 1000.0:.6f}", file=f)
        for t in faces:
            print("f " + " ".join(str(idx + 1) for idx in t), file=f)


def carton(n: int = 64) -> None:
    """60 x 60 x 35 mm block with one conical cup cut from the top: top opening D=46 mm, bottom seat D=24 mm, depth 25 mm."""
    half = 30.0
    h = 35.0
    cz = h / 2.0  # 17.5 mm
    z_floor = 10.0  # 35 - 25 = 10 mm floor thickness
    r_bot = 12.0  # 24 mm bottom seat diameter
    r_top = 23.0  # 46 mm top opening diameter

    # n is a multiple of 8 so theta = pi/4, 3*pi/4, 5*pi/4, 7*pi/4 hit the 4 square corners (+-30, +-30) exactly.
    angles = [2.0 * math.pi * i / n for i in range(n)]
    pts: list[tuple[float, float, float]] = [(0.0, 0.0, z_floor)]  # 0: cup floor centre
    c_bot = len(pts)
    pts += [(r_bot * math.cos(a), r_bot * math.sin(a), z_floor) for a in angles]
    c_top = len(pts)
    pts += [(r_top * math.cos(a), r_top * math.sin(a), h) for a in angles]
    sq_top = len(pts)
    for a in angles:
        ca, sa = math.cos(a), math.sin(a)
        scale = half / max(abs(ca), abs(sa))
        pts.append((scale * ca, scale * sa, h))
    sq_bot = len(pts)
    for a in angles:
        ca, sa = math.cos(a), math.sin(a)
        scale = half / max(abs(ca), abs(sa))
        pts.append((scale * ca, scale * sa, 0.0))
    bot_centre = len(pts)
    pts.append((0.0, 0.0, 0.0))

    faces: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        # Cup floor disk (outward normal +z into the cup cavity)
        faces.append((0, c_bot + i, c_bot + j))
        # Conical cup wall (outward normal inward toward cavity axis / +z)
        faces.append((c_bot + i, c_top + i, c_top + j))
        faces.append((c_bot + i, c_top + j, c_bot + j))
        # Top surface outside the cup opening (outward normal +z)
        faces.append((c_top + i, sq_top + i, sq_top + j))
        faces.append((c_top + i, sq_top + j, c_top + j))
        # Outer vertical walls of the 60 x 60 mm block
        faces.append((sq_bot + i, sq_bot + j, sq_top + j))
        faces.append((sq_bot + i, sq_top + j, sq_top + i))
        # Bottom face of the 60 x 60 mm block (outward normal -z)
        faces.append((bot_centre, sq_bot + j, sq_bot + i))

    with open(os.path.join(HERE, "carton.obj"), "w") as f:
        print("o carton", file=f)
        for x, y, z in pts:
            print(f"v {x / 1000.0:.6f} {y / 1000.0:.6f} {(z - cz) / 1000.0:.6f}", file=f)
        for t in faces:
            print("f " + " ".join(str(idx + 1) for idx in t), file=f)


if __name__ == "__main__":
    egg()
    carton()
