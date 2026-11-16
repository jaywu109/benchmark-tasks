# Stow a hair dryer barrel-down in a tapered bathroom wall holster

Putting the hair dryer back in its bathroom wall holder is an everyday chore, and coil holders like this are common in hotel, gym
and home bathrooms: an open steel spiral that narrows toward the
bottom, so the barrel drops in from the top and the taper catches it. After use the dryer lies on the vanity.
Putting it away means picking it up by its handle, turning it from lying on its side to barrel-down, bringing the
barrel over the top opening and lowering it until the coil bears its weight, then letting go. The barrel fills most
of each turn it passes through. The top turn leaves 10 mm per side around the 70 mm barrel, and the second turn
5 mm. Full credit needs the coil alone to hold the dryer after the hand has let go.

Instruction: *Stow the unplugged hair dryer in the wall holder so it remains supported.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; replaces the dryer's and holster's boxes with the meshes below), `dryer.obj` (barrel,
rear housing, handle and its web as convex parts), `holster.obj` (four rigid wire rings of 24 convex segments, two arms and
a mounting plate), `make_meshes.py` (writes both OBJ files), `task.json`, `test_task.py` (offline checks).

## What is on the table

Positions in metres from the midpoint between the arm bases (x out, y left), heights above the table.

- **Wall tile** (fixed, white porcelain): 298 x 298 x 9 mm, standing on the table with its front face at x 0.4055,
  centred at y -0.125.
- **Hair dryer holster** (fixed, stainless steel; Smedbo HK323 style, 155 mm high x 129 mm long x 100 mm deep):
  mounted to the tile with its axis at (0.325, -0.125). In the simulation, its spiral is four separate rigid 4 mm wire rings at
  257, 207, 156 and 106 mm above the table, with clear openings of 90, 80, 70 and 60 mm. Two short arms join the
  turns to a steel mounting plate on the tile.
- **Hair dryer** (loose, 368 g; Panasonic EH-NA2C style, 154 x 140 x 79 mm): it lies on its side at (0.190, -0.220),
  barrel toward the holster (+x) and handle toward the right arm (-y). It settles rolled 13 deg onto the handle tip.
  The working barrel is a 70 mm (nominal) cylinder, 105 mm long. Behind it is a 79 mm rear housing, 49 mm long. The
  handle is 25 mm wide and 35 mm thick, and projects 61 mm from the housing. The attached cord is not modelled. In a
  physical setup it is routed out of the workspace.

Sizes: holster from Smedbo's published HK323 dimensions (155 x 129 x 100 mm, 90 mm top and 60 mm lower opening,
302 g). Dryer from Panasonic's published EH-NA2C dimensions and mass (154 x 140 x 79 mm, 368 g without nozzle). The
70 mm barrel diameter is nominal; check it on an actual unit. The gripper opens to 94 mm; the handle is 25 mm across
where it is gripped.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to
`~/stow-hair-dryer-in-wall-holster` on the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="Stow the unplugged hair dryer in the wall holder so it remains supported."
T=$HOME/stow-hair-dryer-in-wall-holster/task.py

# a policy (here Dream-1), 600 steps = 30 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$T -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/stow-hair-dryer-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$T -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/stow-hair-dryer-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/stow-hair-dryer-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only). The scorer reads contact reports from the running simulation.
In a physical trial, use video and a ruler instead.

## Scoring

Measured every control step; the trial score is the last one (`SCORE_OVER = "end"`).

- **0** for the rest of the trial (latched) once the dryer drops more than 10 mm below the tabletop or leaves the
  700 x 900 mm table, or the holster moves more than 10 mm.
- **1.0**: the dryer is barrel-down (axis within 32 deg of vertical) with its barrel centre within 12 mm of the
  holster axis. Its tip is at least 20 mm below the top turn and no lower than 15 mm under the lowest turn. It rests on a turn of
  the holster (contact reported). The dryer is still (12 mm/s, 0.15 rad/s) and no robot link touches it. How far the
  hands are open does not matter. All of this holds without a break for
  1.0 s.
- **0.5**: barrel-down, centred and in at least 20 mm, touching a turn, but not yet released and still for 1.0 s
  (for example, still held).
- **0** otherwise, including a dryer left on the table, resting on the rim, or tilted against the tile.

## What makes this task distinctive

The dryer is stowed at a right angle to how it lies. It starts on its side, is held by its handle as a person does, and has to end
barrel-down, so the hand turns it a quarter turn about the handle and carries it with the barrel hanging below the
grip. The barrel then enters a coil whose turns narrow from 90 to 60 mm. The barrel has 10 mm of play per side at the
top turn and 5 mm at the next, and the hand cannot follow it in, because the turns surround the barrel. Nothing
latches or clamps: the dryer stays only because the coil carries it, so the score waits until the hand has let go
and the dryer has stayed put for a second.

What the simulation does differently from a real holder: the coil is four separate rigid rings, with no spring and
no cord. The dryer does not hang by its handle on the top turn. It ends wedged barrel-down, its tip in the 70 mm
turn, held by the coil and gravity (default friction 1.2).

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at step 450 (about 22 s of trial time); 55 s wall clock including
  Isaac start-up. The right hand takes the handle 31 mm out from the housing, its fingers closing across the handle's
  25 mm width and tilted 48 deg from vertical so the pads lie along the handle. It lifts the dryer and turns it
  barrel-down, the tip 30 cm above the table. It then brings the barrel over the coil (7 mm off its axis) and lowers it
  through the turns at about 3.4 cm/s. The barrel comes to rest in the 70 mm turn, tip 156 mm above the table, with
  the handle 15 mm above the top turn. The hand opens and slides off along the handle. For the last 2 s
  the dryer did not move: centre (0.3235, -0.1527, 0.9825) m, quaternion xyzw (-0.0094, -0.7271, 0.0375, 0.6855),
  axis 5 deg from vertical, barrel centre 3.7 mm from the holster axis, 12.6 N on the turns, 0 N from the robot.
- Dream-1: `task_score` 0.0 over the full 600 steps (73 s wall clock). It touched the dryer and dragged it
  about 55 mm toward the arms and turned it about 15 deg about the vertical, but never lifted it: the dryer stayed on the table.
- The demo is tuned to the measured rest pose of the dryer and to the sag of the grip under load, and has one passing
  run; its robustness to other poses has not been checked.
- Offline tests: 5 passed.
- Top camera: in the first frame, the empty coil on the tile and the dryer lying on the table in front of it are
  visible, with no arm over either. In the last frame, the dryer stands barrel-down in the coil with its handle
  pointing right, and the right arm is parked to the right, clear of the holster.

## Third-party assets

none
