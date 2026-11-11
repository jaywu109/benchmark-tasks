# Riddle a row of sparkling-wine bottles an eighth turn in the pupitre

In traditional-method sparkling wine, the yeast left in each bottle after its second fermentation is worked down
into the neck by riddling. The bottles rest neck-down in the holes of a pupitre, a wooden A-frame rack, and every day
the riddler gives each one a quick eighth or quarter turn, reading how far it went from a chalk mark on its base. The
bottle must stay in its hole the whole time. Here one row of four half bottles sits in the rack at 45 degrees, and
each has to be turned an eighth of a turn about its own axis, in place, and let go.

Instruction: *Give each bottle in the rack an eighth turn.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; builds the rack as one compound rigid body of box pieces and each bottle from
`bottle.obj` plus its chalk stripe), `bottle.obj` (the bottle, written already tilted to 45 degrees, as convex
parts), `make_bottle.py` (writes `bottle.obj` from the bottle's profile), `task.json`, `test_task.py` (offline
checks).

## What is on the table

Positions in metres from the midpoint between the arm bases (x out, y left), heights above the table.

- **Pupitre row** (fixed, wood): a board 0.52 wide, 0.35 along its face and 25 mm thick, leaning back 15 degrees, with four square holes
  (31.6 mm across, 40 mm deep) 0.13 apart at y -0.195, -0.065, +0.065, +0.195, each running down into the board at
  45 degrees.
- **Four bottles** (loose, 875 g each), one per hole, neck-down at 45 degrees below horizontal: the neck in the hole,
  the shoulder resting on its mouth, the base end facing the robot about 0.24 above the table at x about 0.31.
- Each bottle: 243.5 mm tall, 70.6 mm across the body, a 29.6 mm neck; dark green, with a white chalk stripe from the
  centre of its base out to its upper edge.

Sources: the bottle is Saverglass's 37.5 cl "Couronne Champ 29" (243.5 mm, 70.6 mm, 500 g of glass; 375 ml of wine
makes 875 g). The Comité Champagne (champagne.fr, "Riddling"): each bottle "rotates by an 1/8 or a 1/4 turn at a
time, with a chalk mark on the bottom of the bottle for reference". The hole pitch is a 60-hole-a-side French
pupitre's (77 x 152 cm). Bottles go into the rack nearly level and end nearly upright over the weeks; 45 degrees is a
mid-riddling stage.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to `~/riddle-the-bottles` on
the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="Give each bottle in the rack an eighth turn."

# a policy (here Dream-1), 900 steps = 45 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$HOME/riddle-the-bottles/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/riddle-the-bottles-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$HOME/riddle-the-bottles/task.py -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/riddle-the-bottles-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/riddle-the-bottles-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

Measured once everything is still (`SCORE_OVER = "end"`), each bottle against the pose it rested in at the trial's
first step. A bottle **counts** when it has turned 30 to 60 degrees about its own axis, either way (the chalk
stripe's angle round the axis), and is still seated: its base centre within 5 mm and its axis within 3 degrees of
where they started.

- **0.25 per bottle that counts**, once both TCPs are at least 3 cm from every bottle (released) and nothing moves.
  1.0 when all four count.
- **0** for everything if any bottle was pulled out of its hole (base centre moved more than 30 mm, or axis more than
  15 degrees) at any time during the trial (latched: pushing it back does not undo it).

## What makes this task distinctive

The motion is a turn of a loose part about its own long axis, in place. Nothing holds the bottle but its hole: it is
not keyed, threaded or hinged, so the hand has to grip the base end and turn it about the bottle's own 45-degree
axis, which the wrist meets only at a slant, and let go without drawing it out. The amount is read from a mark, not a
stop: an eighth turn, with no detent to say when it is reached. The four bottles are 13 cm apart, so the two hands
work the outer pair together and then take the inner pair one after the other.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at about step 660 (33 s of trial time); 96 s wall clock including
  Isaac start-up. Each hand comes in along the bottle's axis, closes on the body 30 mm in from the base, turns it and
  lets go: the right hand bottles 1 and 2, the left hand bottle 4 at the same time as bottle 1, then bottle 3 once
  the right hand has parked. Turns read at the end: +50.7, +51.2, -48.8 and -48.6 deg; base centres 1.9-3.4 mm and
  axes 0.33-0.58 deg from their start.
- Dream-1: `task_score` 0.0 after the full 900-step run limit; 120 s wall clock. The right hand pushed bottle 1 about
  12 mm along its hole (axis 3.1 deg, turned 2.3 deg) and the left hand nudged bottle 4 (4.5 mm, 4 deg); no bottle was
  turned and none came out of its hole.
- Offline tests: 23 passed.
- Top camera: the first frame shows the board, the four bottles and their chalk stripes, all pointing straight up.
  In the demo's last frame the stripes of bottles 1-3 are turned about 50 deg; the left hand, swinging away to its
  park, still covers part of bottle 4.

## Third-party assets

none
