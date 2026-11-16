"""Writes extinguisher.obj: the 2.5 lb extinguisher as four convex parts (body, shoulder, neck, head block), metres,
origin at the bounding-box centre (175 x 410 x 167 mm), axis along +y (bottom end at -y), head block toward +y.
Also writes hook_peg.obj: the bracket's round hook peg (6 mm diameter, 15 mm long, along y) in the bracket's frame.
Run once: python make_mesh.py"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEN, WIDTH, HEIGHT = 0.410, 0.175, 0.167
R_BODY, R_NECK = 0.050, 0.035  # apothems; the 24-gon has a flat facet down and to each side
S_BODY, S_SHOULDER, S_NECK = 0.200, 0.230, 0.290  # end of the body, shoulder cone, neck (m from the bottom end)
HEAD_DZ = 0.030  # head block reaches this far below the axis
AXIS_Z = -(HEIGHT / 2 - R_BODY)  # axis height in the bbox frame


def _ring(s, r):
    a = np.radians(7.5 + 15.0 * np.arange(24))
    rr = r / math.cos(math.radians(7.5))
    return [(rr * math.cos(t), s - LEN / 2, AXIS_Z + rr * math.sin(t)) for t in a]


def _frustum(s0, r0, s1, r1):
    return _ring(s0, r0) + _ring(s1, r1)


def _box(x0, x1, s0, s1, z0, z1):
    return [(x, s - LEN / 2, z) for x in (x0, x1) for s in (s0, s1) for z in (z0, z1)]


def _hull_faces(v):
    from itertools import combinations

    v = np.asarray(v, float)
    c = v.mean(0)
    faces = set()
    for i, j, k in combinations(range(len(v)), 3):  # all coplanar-with-nothing-outside triples (small parts only)
        n = np.cross(v[j] - v[i], v[k] - v[i])
        if np.linalg.norm(n) < 1e-12:
            continue
        d = (v - v[i]) @ n
        if np.all(d <= 1e-12) or np.all(d >= -1e-12):
            faces.add((i, j, k) if (v[i] - c) @ n > 0 else (i, k, j))
    return sorted(faces)


def main():
    parts = {
        "body": _frustum(0.0, R_BODY, S_BODY, R_BODY),
        "shoulder": _frustum(S_BODY, R_BODY, S_SHOULDER, R_NECK),
        "neck": _frustum(S_SHOULDER, R_NECK, S_NECK, R_NECK),
        "head": _box(-WIDTH / 2, WIDTH / 2, S_NECK, LEN, AXIS_Z - HEAD_DZ, HEIGHT / 2),
    }
    lines, base = [], 0
    for name, v in parts.items():
        lines.append(f"o {name}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
        # hull of the convex frustum / box by brute force would be slow for 48 points: use the structured faces
        if name == "head":
            faces = _hull_faces(v)
        else:
            faces = _frustum_faces(v)
        lines += [f"f {a + base + 1} {b + base + 1} {c + base + 1}" for a, b, c in faces]
        base += len(v)
    open(os.path.join(HERE, "extinguisher.obj"), "w").write("\n".join(lines) + "\n")
    # peg centre in the bracket's frame (x, y along the extinguisher axis, z), radius 3 mm, 16-gon
    cx, cy, cz, r, half = 0.052, 0.099, 0.0575, 0.003, 0.0075
    ring = lambda y: [(cx + r * math.cos(a), y, cz + r * math.sin(a)) for a in np.radians(22.5 * np.arange(16))]  # noqa: E731
    v = ring(cy - half) + ring(cy + half)
    f = []
    for k in range(16):
        a, b = k, (k + 1) % 16
        f += [(a, b, 16 + b), (a, 16 + b, 16 + a)]
    f += [(0, k, k + 1) for k in range(1, 15)] + [(16, 16 + k + 1, 16 + k) for k in range(1, 15)]
    c = np.asarray(v).mean(0)
    out = ["o hook_peg"] + [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    for a, b, d in f:
        n = np.cross(np.asarray(v[b]) - v[a], np.asarray(v[d]) - v[a])
        t = (a, b, d) if (np.asarray(v[a]) - c) @ n > 0 else (a, d, b)
        out.append("f " + " ".join(str(i + 1) for i in t))
    open(os.path.join(HERE, "hook_peg.obj"), "w").write("\n".join(out) + "\n")


def _frustum_faces(v):
    v = np.asarray(v, float)
    c = v.mean(0)
    f = []
    for k in range(24):
        a, b = k, (k + 1) % 24
        f += [(a, b, 24 + b), (a, 24 + b, 24 + a)]
    f += [(0, k, k + 1) for k in range(1, 23)] + [(24, 24 + k + 1, 24 + k) for k in range(1, 23)]
    out = []
    for a, b, d in f:  # orient every triangle outward
        n = np.cross(v[b] - v[a], v[d] - v[a])
        out.append((a, b, d) if (v[a] - c) @ n > 0 else (a, d, b))
    return out


if __name__ == "__main__":
    main()
