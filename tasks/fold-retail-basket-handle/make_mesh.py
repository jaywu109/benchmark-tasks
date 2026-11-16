"""Generate the five convex pieces of a tapered, open retail basket shell."""
from pathlib import Path

OUT = Path(__file__).with_name('basket.obj')

def hexa(name, bottom, top, lines):
    # Four corners per level in winding order, in the basket's local frame.
    bx0, bx1, by0, by1, bz = bottom
    tx0, tx1, ty0, ty1, tz = top
    vs = [(bx0,by0,bz),(bx1,by0,bz),(bx1,by1,bz),(bx0,by1,bz),
          (tx0,ty0,tz),(tx1,ty0,tz),(tx1,ty1,tz),(tx0,ty1,tz)]
    start = 1 + sum(1 for s in lines if s.startswith('v '))
    lines.append('o '+name)
    lines += ['v %.6f %.6f %.6f' % v for v in vs]
    for face in ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)):
        lines.append('f '+' '.join(str(start+i) for i in face))

s=[]
# 430 mm along y, 300 mm along x at rim; 330 by 205 mm at base.
# 5 mm wall/floor gauge, independent convex strips preserve the open cavity.
hexa('floor', (-.1025,.1025,-.165,.165,-.1125), (-.1025,.1025,-.165,.165,-.1075), s)
hexa('near_wall', (-.1025,-.0975,-.165,.165,-.1075), (-.150,-.145,-.215,.215,.1125), s)
hexa('far_wall', (.0975,.1025,-.165,.165,-.1075), (.145,.150,-.215,.215,.1125), s)
hexa('right_wall', (-.0975,.0975,-.165,-.160,-.1075), (-.145,.145,-.215,-.210,.1125), s)
hexa('left_wall', (-.0975,.0975,.160,.165,-.1075), (-.145,.145,.210,.215,.1125), s)
OUT.write_text('\n'.join(s)+'\n')
print(OUT)
