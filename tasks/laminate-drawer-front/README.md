# Pull the spacer dowels and bond the laminate to the drawer front

A laminate sheet lies on three spacer dowels over a drawer-front blank coated with dry contact cement. Contact cement
grabs on first touch and the sheet cannot be moved after it bonds. So the robot must pull the dowels out from under
the sheet without dragging it off its place, letting each part of the sheet settle flat onto the blank and bond while
the sheet still overhangs every edge for flush trimming.

Instruction: *Pull out the three spacer dowels so the laminate sheet bonds flat to the drawer front, still overhanging
every edge.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `task.json`, `test_task.py`
(offline checks), `robocurve_ext.py` (spring hinges, mesh parts and the bond joints), `dowel.obj` (the dowel mesh).

## What is on the table

Positions are in metres. x runs forward from the arm bases, y runs left from the midline, and the table top is
z = 0.7475.

- **Drawer-front blank (fixed):** MDF, 160 (x) x 240 (y) x 18 mm, centred at x 0.30, y 0.0 (top z = 0.7655),
  colour (0.50, 0.36, 0.24). Its top face carries dry contact cement.
- **Spacer dowels (loose, 3):** `dowel_n`, `dowel_m`, `dowel_f`, 12 mm round birch dowels (24-sided prisms,
  `dowel.obj`), 300 mm long along y, 60 g, colour (0.86, 0.72, 0.50). They lie on the blank at x = 0.241, 0.300, 0.359,
  from y -0.17 to +0.13, so each sticks 50 mm out past the blank's -y edge.
- **Laminate sheet (loose):** grey-blue (0.45, 0.55, 0.65), 176 x 260 mm, made of three 58 x 260 x 4 mm panels
  (`panel_n`, `panel_m`, `panel_f` at x = 0.241, 0.300, 0.359; 40 g each; 1 mm apart). Two spring hinges join them
  along y (0.004 N m/rad toward flat, 0.002 N m s/rad damping, +-60 deg). The sheet rests on the dowels with its
  underside 12 mm above the blank and overhangs the blank by 8 mm on each x edge and 10 mm on each y edge.
- Nothing else is on the table.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to `~/laminate-drawer-front` on
the GPU machine with `inspect-robots` installed. Always run with `-E object_jitter=0`: a seeded reset shifts every
loose part separately by up to 3 cm, which would pull the hinged panels apart.

```bash
export INSTR="Pull out the three spacer dowels so the laminate sheet bonds flat to the drawer front, still overhanging every edge."

# a policy (here Dream-1), 1,200 steps = 60 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$HOME/laminate-drawer-front/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/laminate-drawer-front-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$HOME/laminate-drawer-front/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/laminate-drawer-front-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/laminate-drawer-front-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

- **Bonding:** a panel bonds the first control step that its lowest corner is within 1 mm of the blank's top and it
  is within 3 deg of flat. From then on a fixed joint holds it where it lies for the rest of the trial; it never
  unbonds. A panel that touches down tilted, such as one end of the sheet dropping while the rest still rides on a
  dowel, does not bond until it lies flat.
- **Registered:** every panel's centre is within 6.0 mm in x and 8.0 mm in y of its set position, so the sheet still
  overhangs every edge of the blank by at least 2 mm.
- A panel **counts** when it is bonded, the sheet is registered, and both grippers' TCPs are at least 3 cm from every
  point of the panel (released).
- `score()` = (counting panels) / 3, up to 1.0 (`SCORE_OVER = "end"`).
- `success()` = all three panels count, held for 1 s.
- `distance()` = sum over unbonded panels of the lowest corner's height above the blank (m), plus how far (m) the
  worst panel is outside the registration window.
- The order of the dowels is not scored. Shops differ: some pull the middle dowel first and work outward, others
  start at one end and hold the sheet in place.

## Design numbers

- **Dowel grip:** straight down with the jaws closing along x across the 12 mm dowel (`yaw = 90 deg`), at y -0.155,
  35 mm past the blank's -y edge. Pad tips are 13 mm below the TCP, so at the grip height (the dowel's centre, z
  0.7715) they end 7 mm below the blank's top, 25 mm clear of its edge. The jaws approach 40 mm open: fully open
  (94 mm) the pads would hit the neighbouring dowel ends 59 mm away.
- **Pull:** 275 mm along -y in 3 s, which takes each dowel's +y end 15 mm past the sheet's -y edge. The dowel is then
  set down 10 mm above the table and released. The worst IK error over the plan at 20 Hz is under 0.1 mm and
  0.02 deg.
- **Hinge spring:** a side panel pivots on its own dowel, so for the middle panel to reach the blank each hinge must
  bend about 0.41 rad (12 mm over a 29.5 mm lever). The middle panel's half weight gives 0.0058 N m on that lever.
  The 0.004 N m/rad spring takes 0.0016 N m of it, so the panel lands. A stiffer sheet stops short of the blank.
- **Registration:** the sheet is cut 8 mm (x) and 10 mm (y) oversize per edge. A flush-trim bit needs some
  overhang, so the bar keeps at least 2 mm on every edge.
- **Bond rule:** within 1 mm and 3 deg of flat, so a panel resting tilted with one edge down does not bond.

## What makes this task distinctive

The parts move under a rule that cannot be undone. Wherever the sheet first lies flat on the cement, it stays. The
robot cannot place the sheet; it can only take away what holds it up. Each dowel lies under the sheet's full length
and carries part of its weight. A pull that drags the sheet with it, or lets one end drop first so the sheet tips and
slides, can bond the sheet off its mark, and nothing can fix it afterwards. The score depends on where the sheet ends up,
not on how it got there.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at step 517 (26 s of trial time); 59 s wall clock including
  Isaac start-up. Success means all three panels bonded with the sheet registered (every panel within 6 mm in x and
  8 mm in y of its set position) and both grippers clear.
- Dream-1: `task_score` 0.0 after the full 1,200-step run limit; 117 s wall clock. The right arm reached toward the
  dowel ends several times but pulled none out, and no panel bonded.
- Offline tests: 17 passed.
- Top camera: in the first frame the sheet, the blank's near edge and the three dowel ends are visible between the
  arms. In the demo's last frame the sheet lies flat and square on the blank, the three dowels lie out on the table
  at -y, and the right gripper is clear of the sheet.

## Third-party assets

none
