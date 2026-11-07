# Load five glass slides into a staining rack

Five 75 x 25 x 1.0 mm glass microscope slides stand upright in a five-slot drying rack on the left side of the table.
Beside it on the right sits a five-slot staining rack with 4.0 mm wide slots. The robot must slot all five
upright into the staining rack, one slide per slot, and let go of each, without tipping any slide flat or dropping it
off the table.

Instruction: *Move the five slides from the drying rack into the staining rack, one slide per slot.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `task.json`, `test_task.py`
(offline checks).

## What is on the table

Positions are in metres. x runs forward from the arm bases, y runs left from the midline, and the table top is
z = 0.7475.

- **Slides (loose, 5):** `slide_1` to `slide_5`, soda-lime glass microscope slides, 75 mm (y) x 1.0 mm (x) x 25 mm (z),
  5 g each, colour (0.80, 0.88, 0.90), standing upright on a long edge in the drying rack's grooves at x = 0.25, 0.28,
  0.31, 0.34, 0.37 and y = +0.30.
- **Slide drying rack (fixed, source):** grey plastic (0.55, 0.57, 0.60), centred at x 0.31, y +0.30. A 150 (x) x
  45 (y) x 3 mm floor slab on the table (top z = 0.7505) with six 45 mm long, 10 mm tall groove blocks forming five
  4.0 mm wide grooves at 30 mm pitch along x (x = 0.25, 0.28, 0.31, 0.34, 0.37).
- **Staining rack (fixed, destination):** white plastic (0.92, 0.92, 0.90), centred at x 0.31, y +0.12. Same 150 x 45 x
  3 mm floor slab and five 4.0 mm wide grooves at 30 mm pitch along x, with six 12 mm tall compound groove blocks
  (`robocurve_ext.Compound`, 8.0 mm chamfered mouths at the top narrowing to the 4.0 mm groove body 4 mm down) and open
  ends so the slides overhang by 15 mm on each side.
- Nothing else is on the table.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to `~/load-staining-rack` on
the GPU machine with `inspect-robots` installed. Always run with `-E object_jitter=0`: a seeded reset shifts loose
objects by up to 3 cm, which would move the slides out of their 4.0 mm grooves.

```bash
export INSTR="Move the five slides from the drying rack into the staining rack, one slide per slot."

# a policy (here Dream-1), 1,600 steps = 1 minute 20 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$HOME/load-staining-rack/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/load-staining-rack-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$HOME/load-staining-rack/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$INSTR" --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/load-staining-rack-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/load-staining-rack-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

- `tilt` of a slide is the angle between its local z axis (25 mm side) and world z.
- A slide is **in rack slot k** (k = 1..5, centres at x = 0.25, 0.28, 0.31, 0.34, 0.37, y = +0.12) when its centre x is
  within 2.0 mm of slot k's centre, its centre y is within 10 mm of the rack's centre y, its lowest point is within
  3 mm of the rack slab's top (z = 0.7505), and its tilt is under 10 deg.
- **Latched violation per slide (`<slide> fell`):** if a slide's tilt ever exceeds 60 deg (lying flat) or its centre
  goes below the table top (or beyond the table footprint), that slide counts 0 for the rest of the trial even if
  stood up again.
- Two or more slides in the same slot: all of them in that slot count 0.
- A slide counts only when both grippers' TCPs are at least 3 cm from every point of it, so a slide still held does not count.
- `score()` = 0.2 x (slides in a rack slot, `resting()` (under 1 cm/s and 0.5 rad/s), not fallen, alone in their
  slot, released), up to 1.0 (`SCORE_OVER = "end"`).
- `success()` = all five slides counted and every slide `resting()` for 1 s.
- `distance()` = sum over slides not counted of the horizontal distance from the slide centre to the nearest free rack
  slot centre (m).

## Design numbers

- **Gripper:** fully open, the pads are 93.9 mm apart (`FINGER_OPEN` in `yam_arena/constants.py`). A 75 mm slide leaves
  18.9 mm total clearance (9.45 mm per side).
- **Grip:** fingers close along y (`yaw = 0`) on the slide's two 25 x 1 mm end faces to `-(SLIDE_L / 2 - 0.0017)`
  (-0.0358 m), and release to `-(SLIDE_L / 2 + 0.006)` (-0.0435 m, 6 mm off each end) before rising until the pads are
  3 cm above the slide. The TCP sits 20 mm above the slab
  top (pad tips 7 mm above the slab, pads covering the upper 18 mm of the 25 mm slide). Because the groove blocks are
  45 mm long along y, each slide overhangs the blocks by 15 mm at both ends so the pads clear the groove blocks.
- **Pitch:** grooves are spaced at 30 mm pitch along x so the ~27 mm wide fingers gripping one slide's ends stay clear
  of the neighbouring slides.
- **Demo:** the right arm waits at `(0.20, -0.32, TOP + 0.12)` with fingers open while the left arm transfers slides
  1 to 5 in sequence (hover 8 cm up, descend to grip height, close to `GRIP_FINGER`, lift over 2.0 s until the slide
  bottom is 4 cm above the 12 mm groove tops, carry to the matching staining-rack slot, lower over 3.0 s until the
  slide bottom is 2 mm above the slab, open to `RELEASE_FINGER`, rise until the pads are 3 cm above the slide, then open
  to `FINGER_OPEN` on the next approach); after `slide_5` the left arm rises and moves aside to
  `PARK_LEFT = (0.20, +0.40, TOP + 0.12)`, leaving the filled rack in view of the top camera.

## What makes this task distinctive

The part is a 1 mm thick glass slide. The scripted demo holds each one only by its two 25 x 1 mm end faces, the way
technicians keep fingers off the specimen face (the scorer does not check how a slide is held). Each slide has to go
into a 4.0 mm groove, which leaves 1.5 mm of clearance on each side, and stand there upright on its own after release: a
slide counts only once both grippers are at least 3 cm away. The task repeats this five times in one 80 s trial, and
every slide is scored separately. If a slide ever tips flat, it scores 0 for the rest of the trial, even if it is stood
back up. Two slides in one slot both score 0, because slides pressed together stain unevenly.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at step 1254 (63 s of trial time); 143 s wall clock including
  Isaac start-up.
- Dream-1: `task_score` 0.0 after the full 1,600-step run limit; 151 s wall clock.
- Offline tests: 25 passed.
- Top camera: in the first frame both racks and the five slides are visible between the arms; in the demo's last frame
  all five slides stand in the staining rack, the drying rack is empty and both grippers are clear of the racks.

## Third-party assets

none
