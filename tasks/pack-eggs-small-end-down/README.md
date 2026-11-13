# Pack the last two eggs small end down

Egg graders and packers set eggs in cartons small end down, so that the air cell at the blunt end stays up and the
yolk stays centred. The USDA *Egg-Grading Manual* (Agricultural Handbook No. 75, 2000, p. 32) says eggs returned to a
carton are "always packed with the small end down". Eggs that arrive lying on
their side have to be turned on end as they are set in a cup. Here two
loose eggs lie on rubber cradles on the left and right, and the carton's two end cups are empty while its four inner
cups already hold eggs standing small end down.

Instruction: *Put the two loose eggs into the empty carton cups, small end down.*

Files: `task.py` (objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py`, `make_meshes.py`, `egg.obj`,
`carton.obj`, `task.json`, `test_task.py` (offline checks).

## What is on the table

Positions are in metres. x runs forward from the arm bases, y runs left from the midline, and the table top is
z = 0.7475.

- **Loose eggs (2):** `egg_l` at `(0.22, +0.20)` and `egg_r` at `(0.22, -0.20)`, each a 57 mm long by 44 mm equatorial
  diameter Hügelschäffer egg (`egg.obj`, `w = 3 mm`, small end at local `+z`, 58 g, colour `(0.93, 0.87, 0.76)`,
  `approximation="convexHull"`), lying horizontally along x with its small end pointing toward `+x`.
- **Cradles (fixed, 2 pairs of rails):** two blue rubber rails `(0.25, 0.45, 0.80)` per loose egg at `x = 0.22`,
  `y = +-0.20 +- 0.012`, each `40 (x) x 6 (y) x 8 (z) mm` (18 mm gap between inner faces).
- **Empty carton cups (fixed, 2):** `cup_l` at `(0.34, +0.20)` and `cup_r` at `(0.34, -0.20)`, each a
  `60 x 60 x 35 mm` moulded-pulp block `(0.72, 0.68, 0.60)` with a 25 mm deep conical cup tapering from 46 mm diameter
  at the rim (`TOP + 0.035`) to a 24 mm flat seat floor (`TOP + 0.010`), exact triangle mesh (`carton.obj`,
  `approximation="none"`).
- **Bridging plate and filled inner cups (fixed):** a 3 mm thick pulp plate (`carton_plate`) bridging the top of the
  carton row across `y = -0.17` to `+0.17`, `x = 0.31` to `0.37` (`z = TOP + 0.032` to `TOP + 0.035`), together with
  four fixed inner cups (`cup_1` to `cup_4`) and four fixed eggs (`fixed_egg_1` to `fixed_egg_4`) standing small end
  down at `y = -0.105, -0.035, +0.035, +0.105`, `x = 0.34`.
- Nothing else is on the table.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to
`~/pack-eggs-small-end-down` on the GPU machine with `inspect-robots` installed. Run with `-E object_jitter=0` so the
two loose eggs start centred on their 18 mm cradle rails.

```bash
export INSTR="Put the two loose eggs into the empty carton cups, small end down."

# a policy (here Dream-1), 600 steps = 30 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$HOME/pack-eggs-small-end-down/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/pack-eggs-small-end-down-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$HOME/pack-eggs-small-end-down/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/pack-eggs-small-end-down-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/pack-eggs-small-end-down-*/*.json; do inspect-robots video "$log"; done
```

Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py` (NumPy, SciPy and pytest only).

## Scoring

- `tilt_deg(state)` is the angle between the egg's small-end axis (local `+z`) and world `-z` (`0 deg` means small end
  straight down).
- Why `20 deg`: a conical carton cup (`46 mm` rim, `24 mm` floor, `25 mm` deep) holds an egg leaning against the wall
  rather than strictly vertical; a correctly packed egg rests at `13.3 deg`, whereas an egg set blunt end down rests at
  `126-141 deg` and an egg left on its cradle sits at `~71 deg`.
- An egg (`egg_l` or `egg_r`) is **packed** in an empty cup (`cup_l` at `(0.34, +0.20)` or `cup_r` at `(0.34, -0.20)`)
  when all of these hold:
  - its centre's xy is within `4.0 mm` of that empty cup's axis;
  - its lowest point (world bounding-box `low[2]`) is below that cup's rim (`TOP + 0.035`) and on the table;
  - its small-end axis (local `+z`) is within `20.0 deg` of world `-z`;
  - `resting()` (linear speed under `1 cm/s` and angular speed under `0.5 rad/s`);
  - `released()` (both grippers' TCPs at least `3.0 cm` from every point of the egg's bounding box).
- Each empty cup counts once.
- `score()` = `0.5` x the number of packed eggs (`SCORE_OVER = "end"`).
- `success()` = both loose eggs packed (held for `1.0 s` ends the trial).
- `distance()` = sum over the two loose eggs of the horizontal xy distance (m) to the nearer empty cup, plus `0.001`
  per degree of tilt above `20.0 deg`.

### Egg integrity (real-world trials)

The simulated eggs are rigid convex hulls: they cannot crack or leak, so a gripper that squeezes an egg hard enough to
crack a real one still earns placement credit in simulation. The simulation score cannot detect damage. On real eggs
this check is applied before any placement credit:

- **Before the trial**, check all six eggs (the two loose ones and the four prepacked ones) under a bright light, and
  replace any egg that is cracked or leaking.
- **After the run**, check each egg the same way, in place. An egg is **damaged** if it shows any crack (a hairline
  crack included), a dent in the shell, or any leak (a wet shell, or liquid on the carton or the table).
- **Loose eggs:** a damaged loose egg earns no credit for its cup, however well it is placed.
- **Prepacked eggs:** if any of the four prepacked eggs is damaged, or has been lifted or knocked out of its cup, the
  trial scores 0. (In simulation these four eggs are fixed, so they cannot be disturbed.)
- **Reset:** replace every damaged egg before the next trial.

## Design numbers

- **Gripper:** fully open, the pads are `93.9 mm` apart (`FINGER_OPEN = -0.04695 m`). A `44 mm` egg leaves `49.9 mm`
  total clearance (`24.95 mm` per side).
- **Partial close (`FINGER_GRIP = -0.015 m`):** commands `30 mm` pad separation on the `44 mm` egg equator (`w/2 - 7 mm`),
  avoiding the dual-finger saturation drift while holding the 58 g egg during the 75 deg wrist turn.
- **Wrist turn (the scripted demo):** both arms run simultaneously on their own sides (`y = +0.20` for the left arm,
  `y = -0.20` for the right arm), staying at least `20 mm` apart throughout. Each arm approaches at `(0.22, +-0.20)`
  with the gripper pitched `-60 deg` about world `+Y` (`pitched_grip(-60)`), lowers to `TOP + 0.0285 m` (pad tips
  `8.4 mm` above the `8 mm` cradle rails), closes to `FINGER_GRIP = -0.015 m`, lifts to `TOP + 0.110 m`, and carries
  forward to `(0.34, +-0.20)` while pitching `+75 deg` about world `+Y` to `pitched_grip(+15)` so the egg's small end
  (initially `+x`) rotates downward to point into the cup. Each arm then lowers to `PLACE_Z = TOP + 0.055 m` (pad tips
  `6.2 mm` above the `35 mm` cup rim), opens to `FINGER_RELEASE = -0.026 m` (`52 mm` opening), retreats to
  `TOP + 0.110 m` with `FINGER_OPEN`, and parks clear of the carton at `(0.20, +-0.32, TOP + 0.120)`.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, seed 0):

- Scripted demo: `task_score` 1.0, ended by success at step 269 (13.45 s of trial time); 39 s wall clock including
  Isaac start-up. At the end `egg_l` sits 1.5 mm from its cup's axis at 12.8 deg tilt and `egg_r` 1.4 mm at
  13.3 deg, both at rest with both arms parked clear of the carton.
- Dream-1: `task_score` 0.0 after the full 600-step run limit (each arm reached its egg on the cradle, but neither egg
  left its cradle; both stayed at 71-72 deg tilt).
- Offline tests: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py`: 18 passed.

## Third-party assets

none
