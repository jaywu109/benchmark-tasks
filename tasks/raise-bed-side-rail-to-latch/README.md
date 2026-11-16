# Raise a bed's half side rail to its top position until its latch catches

Nursing assistants (O*NET 31-1131.00, about 1.4 million in the US) and home health and personal care aides (about
4 million; BLS Occupational Employment and Wage Statistics, May 2024) work at care beds all shift. After bed baths,
repositioning, linen changes or transfers—and before leaving a patient whose care plan calls for raised rails—the side
rail is swung back up to full height until its spring latch clicks into the keeper catch, and checked that it holds. A
half side rail is a hinged steel frame of a few kilograms: until its latch catches at the top, nothing but the hand
keeps it from swinging back down onto its lower stop.

Here a 3.0 kg half side rail starts lowered at 100 deg in front of a mattress; a spring-loaded ramped latch on the +y
end of the rail engages a fixed steel keeper bracket on the bed frame as the rail reaches 0 deg.

Instruction: *Put the bed's side rail up.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; builds the mattress keeper, the rail and its ramped latch as jointed compound bodies made
of boxes), `task.json`, `test_task.py` (offline checks).

## What is on the table

Positions in metres from the midpoint between the arm bases (x out, y left), heights above the table.

- **Mattress and latch keeper**: fixed mattress block 0.50 x 0.90 x 0.18 m, near edge at x 0.44, centred at y 0, with a
  fixed steel latch keeper bracket at y +0.255 (x 0.419 to 0.494, 24 mm wide) carrying a 20 deg entry ramp that slopes
  down to a vertical locking lip at x 0.451 (0.360 to 0.368 m above the table) and a 23 mm catch pocket behind it.
- **Half side rail**: 3.0 kg steel frame, 0.60 m long along y and 0.30 m tall above its hinge, 25 mm thick: a round
  30 mm top tube, a bottom bar and three vertical posts. Its hinge runs along y at x 0.42, 0.11 m above the table, so
  the upright rail's top tube is 0.23 m above the mattress top. It spawns lowered at 100 deg toward -x on its lower
  stop on a free gravity hinge (center of mass 6 mm toward -x so an unlatched rail always swings down under gravity).
- **Ramped spring latch**: 80 g assembly at the +y end of the rail on a vertical 15 mm slide with a preloaded upward
  spring (80 N/m, 1.2 N upward preload at rest): a red release paddle (26 x 15 x 80 mm at y +0.285) linked to a rear
  catch arm and 24 deg bevelled tooth at y +0.255. As the rail swings up over its last 8 deg to 0 deg, the keeper ramp
  depresses the tooth downward by 8.8 mm against its spring; once the rail is within 0.6 deg (3.0 mm) of the top stop,
  the tooth clears the locking lip at x 0.451 and snaps upward into the catch pocket, holding the rail upright by
  contact alone.

The 0.23 m from mattress top to rail top is above the 220 mm minimum that IEC 60601-2-52 (medical beds) sets for side
rails over an uncompressed mattress.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to
`~/raise-bed-side-rail-to-latch` on the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="Put the bed's side rail up."

# a policy (here Dream-1), 360 steps = 18 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$HOME/raise-bed-side-rail-to-latch/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 360 --store-frames --log-dir ~/yam-eval/raise-bed-side-rail-to-latch-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$HOME/raise-bed-side-rail-to-latch/task.py -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 360 --store-frames --log-dir ~/yam-eval/raise-bed-side-rail-to-latch-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/raise-bed-side-rail-to-latch-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

Checked every control step; the trial ends with 1.0 as soon as the end state has held for 1 s (20 steps), otherwise
the score is 0.

- **1.0**: the rail's top tube within 5.0 mm of its 0.0 deg top position and moved no more than 0.5 deg over the last
  1.0 s, the spring latch caught in the keeper pocket (downward depression 3.0 mm or less and locking tooth corner past
  x 0.4505 m), and both gripper tool points at least 40 mm from every rail and latch collider, all held for 1.0 s.
- **0 for the episode** (kept until reset, whatever happens afterwards): `rail_dropped`, the rail's top tube lifted
  more than 50 mm above its 100 deg lower stop and then allowed to fall back within 5 mm of the lower stop.
- **Measured on real objects**: a rotary encoder or an inclinometer (IMU) on the rail gives its angle, top-bar height
  and motion over the 1 s window at every step; a linear sensor or optical marker on the latch body gives its vertical
  engagement into the keeper pocket; the gripper tool points come from the arms' joint encoders and their distance to
  the rail from its measured angle and known shape.

## What makes this task distinctive

The rail is a 3.0 kg load on a free hinge whose arc sweeps 28.5 cm in both reach and height from near the table edge
(x 0.14 m, 6 cm above the table) to high in front of the mattress (x 0.42 m, 40 cm above the table), and its latch
catches only in the last 3.0 mm before the 0 deg stop after pushing the spring-loaded tooth past the keeper's entry
ramp. Stopping even 10 mm short leaves the tooth on the outside of the ramp so the rail swings all the way back down to
its lower stop on release and triggers `rail_dropped`.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using `./workstation/run_task.sh raise-bed-side-rail-to-latch` (`object_jitter=0`, default seed):

- Scripted demo: `task_score` 1.0, ended by success at step 239 (11.95 s of trial time); 61 s wall clock including Isaac start-up. The right arm closes partway onto the 30 mm top tube at 100 deg (`-15.4 mm` per finger), swings the rail smoothly up through its 100 -> 0 deg arc (peak angular speed `94.51 deg/s`), pushes the ramped latch tooth past the keeper ramp (`8.83 mm` peak downward latch depression) until the tooth snaps up into the catch pocket (`0.00 mm` depression, locking corner at `x = 0.4533 m`), opens and retreats to park (`87.1 mm` TCP clearance), while the rail remains latched upright at `0.16 deg` (`0.79 mm` from `0 deg`, `0.00 deg` motion over the 1.0 s hold window).
- Offline tests: 18 passed.

## Third-party assets

none
