"""Generate the task's full-size convex collision pieces (metres)."""

from math import cos, pi, sin
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_obj(path, parts):
    with path.open("w") as out:
        index = 1
        for name, vertices, faces in parts:
            out.write(f"o {name}\n")
            for p in vertices:
                out.write("v %.8f %.8f %.8f\n" % p)
            for face in faces:
                out.write("f " + " ".join(str(index + i) for i in face) + "\n")
            index += len(vertices)


def cyl(name, a, b, radius, sides=12):
    """A convex cylinder from a to b with a local radial basis."""
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    u = (b - a) / np.linalg.norm(b - a)
    seed = np.array((0., 0., 1.)) if abs(u[2]) < .8 else np.array((0., 1., 0.))
    v = np.cross(u, seed)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    ring = [radius * (cos(2*pi*i/sides)*v + sin(2*pi*i/sides)*w) for i in range(sides)]
    pts = [tuple(a + q) for q in ring] + [tuple(b + q) for q in ring]
    faces = [tuple(range(sides-1, -1, -1)), tuple(range(sides, 2*sides))]
    faces += [(i, (i+1)%sides, sides+(i+1)%sides, sides+i) for i in range(sides)]
    return name, pts, faces


def box(name, lo, hi):
    x0,y0,z0=lo; x1,y1,z1=hi
    pts=[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
         (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    return name,pts,faces


def main():
    # 4 steel turns: the centreline radius is compensated for the 24-sided
    # chord, so each *inner* opening remains 90/80/70/60 mm respectively.
    coil=[]
    for j,(inner,z) in enumerate(((.090,.257),(.080,.207),(.070,.156),(.060,.106))):
        r=(inner/2+.002)/cos(pi/24)
        for i in range(24):
            t0=2*pi*i/24; t1=2*pi*(i+1)/24
            a=(r*cos(t0),r*sin(t0),z-.1815)
            b=(r*cos(t1),r*sin(t1),z-.1815)
            coil.append(cyl(f"turn{j}_segment{i}",a,b,.002,8))
    # Two short arms join the coil to its steel mounting plate, which is
    # bolted through the porcelain wall tile.
    coil.append(box("mount_plate",(.069,-.045,-.0775),(.0796,.045,.0775)))
    coil.append(box("upper_arm",(.040,-.006,.070),(.069,.006,.075)))
    coil.append(box("lower_arm",(.025,-.006,-.075),(.069,.006,-.070)))
    write_obj(HERE/"holster.obj",coil)

    # Dryer coordinate frame: barrel axis +x, handle projects toward -y.
    # Its bounding box is 154 x 140 x 79 mm.  The working barrel is 70 mm
    # nominal; the rear housing is 79 mm across and remains above the coil.
    dryer=[
        cyl("barrel",(-.077,.0305,0),(.028,.0305,0),.035,16),
        cyl("rear_housing",(.028,.0305,0),(.077,.0305,0),.0395,16),
        box("handle",(.041,-.070,-.0175),(.066,.007,.0175)),
        box("handle_web",(.040,-.009,-.0175),(.067,.032,.0175)),
    ]
    write_obj(HERE/"dryer.obj",dryer)


if __name__ == "__main__":
    main()
