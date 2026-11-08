# Hold a cage door shut and slide its bolt across

A small-animal cage stands on the midline of the station table with its front door swung 35 degrees open under a weak
hinge spring and a horizontal slide bolt retracted on the latch-side post. Because the door does not stay shut on its
own and a bolt thrown early blocks the door from closing, the door must be held shut against the frame while the bolt is
slid across its front face; the scripted demo uses two arms (one holds the door, one slides the bolt), though the scorer
checks only the door and bolt poses and gripper clearance, not which arm does what.

Instruction: *Shut the cage door and bolt it.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no edits to `yam_arena`; adds the door's revolute hinge and the bolt's prismatic slide via an import hook on
`yam_arena.scene`), `task.json`, `test_task.py` (offline checks).

## What is on the table

Positions are in metres. x runs forward from the arm bases, y runs left from the midline, and the table top is
z = 0.7475. All fixed parts use stainless grey (0.62, 0.64, 0.66).

- **Cage body (fixed `Box`):** 140 (x) x 240 (y) x 130 (z) mm, standing on the table centred at x 0.44, y 0.0 (front
  face at x 0.370).
- **Front frame (three fixed `Box`es, each 10 mm deep in x from x 0.360 to 0.370):**
  - `hinge_post`: y +0.077 to +0.120 (43 mm wide), 130 mm tall;
  - `latch_post`: y -0.120 to -0.077 (43 mm wide), 130 mm tall;
  - `sill`: y -0.077 to +0.077 (154 mm wide), 8 mm tall.
  - The door opening spans y -0.077 to +0.077 from z = TOP + 0.008 up.
- **Door (loose `Box` + `rx.Revolute`, 150 g):** powder-coated dark green (0.15, 0.35, 0.22), 8 (x) x 150 (y) x
  110 (z) mm. In its closed zero pose it sits centred at x 0.364, y 0.0, z = TOP + 0.065 (front face at x 0.360 flush
  with the frame front, 2 mm from each post and 2 mm above the sill). Hinged about world z along its front hinge-side
  edge at (x 0.360, y +0.075) (`anchor = (-0.004, +0.075, 0.0)`, `axis = "Z"`, `lower = -100`, `upper = 0`,
  `initial = -35`, `target = -35`, `stiffness = 0.0004` N*m/deg, `damping = 0.00003` N*m*s/deg), it opens outward
  toward -x and spawns at 35 deg open, at rest.
- **Slide bolt (loose `Box` + `rx.Prismatic` along world y, 20 g):** zinc grey (0.70, 0.70, 0.72), 20 (x) x 40 (y) x
  14 (z) mm, riding on the front of the latch-side post at x 0.349 (back face at x 0.359, 1 mm in front of the closed
  door's front face), z = TOP + 0.065. Retracted (zero pose, spawn) it spans y -0.117 to -0.077 (centred at y -0.097),
  2 mm short of the door's free edge (y -0.075). Travel is 0 to 25 mm toward +y (`damping = 2.0` N*s/m); thrown d mm,
  it overlaps the shut door's front face by d - 2 mm.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to `~/secure-cage-door` on the
GPU machine with `inspect-robots` installed. `task.py` imports `robocurve_ext.py` from its own directory to attach the
PhysX revolute and prismatic joints without modifying `yam_arena`. Always run with `-E object_jitter=0`.

```bash
export INSTR="Shut the cage door and bolt it."

# a policy (here Dream-1), 600 steps = 30 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$HOME/secure-cage-door/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/secure-cage-door-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$HOME/secure-cage-door/task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/secure-cage-door-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/secure-cage-door-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

- `door_angle_deg` is the door's rotation from its closed zero pose, `2 * acos(|w|)` in degrees.
- `bolt_throw_mm` is the bolt's +y travel from its spawn pose in mm (`(bolt_pos_y - (-0.097)) * 1000`).
- **Door shut:** `door_angle_deg <= 1.5`.
- **Released:** both TCPs are at least 3 cm from every point of the door's box and at least 3 cm from every point of
  the bolt's box (`released()` for each).
- `score()` (`SCORE_OVER = "end"`):
  - **1.0** when the door is shut, `bolt_throw_mm >= 15`, both are released, and both are `resting()`;
  - **0.5** when the door is shut, `5 <= bolt_throw_mm < 15`, both are released, and both are `resting()`;
  - **0.0** otherwise.
- `success()` = the 1.0 condition (held for 1 s).
- `distance()` = `door_angle_deg / 35 + max(0, 15 - bolt_throw_mm) / 15`.

## Design numbers

- **Left-arm door push (`yaw = -pi / 2`, fingers closed):** turning the gripper by `-90 deg` places `tip_left`'s
  26.4 mm wide broad face in `+x`. At TCP height `z = TOP + 0.105` m, `tip_left` contacts the door's upper front face
  (`z <= TOP + 0.120` m) at `+8.72` mm ahead of the TCP, so commanding `x = 0.3545` m presses `3.2` mm past the closed
  front face (`x = 0.360` m) while clearing the 130 mm tall cage body (`x = 0.370` m) by `3.8` mm and keeping `link4`
  high enough to clear the right arm's `link4` by at least `7.0` mm throughout the bolt throw.
- **Right-arm bolt push (`yaw = 0.0`, fingers closed):** at TCP `x = 0.323` m, `z = TOP + 0.025` m (lowest pad tip
  `12.1` mm above the table), the entire right gripper stays at least `3.0` mm in front of the latch-side post
  (`x = 0.360` m), while at the bolt's height (`z = TOP + 58..72` mm, `rel_z = +33..47` mm) the finger body spans
  `x = 0.294..0.3526` m (`13.6` mm overlap across the bolt's `-y` end face `x = 0.339..0.359` m) with its `+y` face
  at `+19.7` mm from the TCP. Pushing `+y` by `22` mm from `y = -0.137` m to `y = -0.115` m over `1.5` s drives the
  bolt's `-y` face from `-0.117` m to `-0.0953` m (`bolt_throw_mm` about `21` mm).
- **Order-of-operations physics:** if the bolt is thrown before the door is shut, the closing door's back face
  (`u = 8` mm from the front face) hits the bolt's back face (`x = 0.359` m) at `11.22` deg open (at 5 mm throw) to
  `12.47` deg open (at 20 mm throw), more than 7x `DOOR_SHUT_DEG` (1.5 deg). When the door is shut first and bolted,
  backing off the left arm lets the door swing out only `0.38` deg against the bolt (`<= 1.5` deg).

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, seed 0):

- Scripted demo: `task_score` 1.0, ended by success at step 202; 30 s wall clock including Isaac start-up. The door
  sat at 0.0-0.1 deg while held, the bolt went 22 mm across, and the door settled at 0.4 deg against the bolt once the
  left arm backed off.
- Dream-1: `task_score` 0.0 after the full 600-step run limit (both grippers reached toward the cage front; the door
  stayed open and the bolt was not thrown).
- Offline tests: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py`: 22 passed.

## Third-party assets

none
