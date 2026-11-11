"""37.5 cl sparkling-wine bottle (Saverglass "Couronne Champ 29": 243.5 mm tall, 70.6 mm body), three convex parts
(body, shoulder, neck), written already tilted: its axis points neck-first along D = (cos A, 0, -sin A).
Body frame origin: on the axis, S0 from the base. Profile is (r, s) with s from the base along the axis."""
import math, os
A = math.radians(45.0)
S0 = 0.12175  # half the height: the body origin
N = 32
PARTS = {"body": [(0.0353, 0.0), (0.0353, 0.130)],
         "shoulder": [(0.0353, 0.130), (0.030, 0.160), (0.022, 0.185), (0.0155, 0.200)],
         "neck": [(0.0148, 0.200), (0.0148, 0.2435)]}
D = (math.cos(A), 0.0, -math.sin(A))
E = (math.sin(A), 0.0, math.cos(A))  # perpendicular to the axis, "up" side


def pt(r, s, a):
    c, w = r * math.cos(a), r * math.sin(a)  # c along E, w along y
    u = s - S0
    return (u * D[0] + c * E[0], w, u * D[2] + c * E[2])


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bottle.obj"), "w") as fh:
        base = 1
        for name, rs in PARTS.items():
            fh.write(f"o {name}\n")
            for r, s in rs:
                for k in range(N):
                    fh.write("v %.6f %.6f %.6f\n" % pt(r, s, 2 * math.pi * k / N))
            for i in range(len(rs) - 1):
                for k in range(N):
                    a, b = base + i * N + k, base + i * N + (k + 1) % N
                    fh.write(f"f {a} {b} {b + N} {a + N}\n")
            fh.write("f " + " ".join(str(base + k) for k in reversed(range(N))) + "\n")
            fh.write("f " + " ".join(str(base + (len(rs) - 1) * N + k) for k in range(N)) + "\n")
            base += len(rs) * N
