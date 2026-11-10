# Put a comb frame back in a honey super, pushed up against the others

A 10-frame Langstroth shallow super stands on the table with seven drawn frames hanging in it, pushed together against
the far wall. An eighth frame hangs on a frame perch hooked over the near wall. The robot has to lift the 48 cm frame
high enough for its bottom to clear the wall (the top bar about 30 cm above the table), lower it into
the space in the box until both lugs sit on the frame rests, and slide it along the rests until its shoulders touch
the next frame. Self-spacing (Hoffman) frames are meant to touch: a gap invites brace comb.

Instruction: *Put the frame back in the super, pushed up against the others.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; builds the box, perch and frames as compound rigid bodies of box pieces), `task.json`,
`test_task.py` (offline checks).

## What is on the table

Positions in metres from the midpoint between the arm bases (x out, y left), heights above the table.

- **Super** (fixed): pine, inside x 0.245 to 0.620 and |y| <= 0.2335, 146 mm tall, 19 mm walls. Each end wall has a
  16 mm wide frame-rest rebate whose floor is 9.5 mm below the rim, with a 3 mm lip outside it.
- **Seven frames** (loose, 0.35 kg each), hanging by their lugs, centres x 0.6025 to 0.3925 (35 mm apart, shoulders
  touching, the last against the far wall). The nearest shoulders face the robot at x 0.375.
- **Frame perch** (fixed, steel): two bars 16 mm wide and 10 mm tall, their tops at rebate height, x 0.135 to 0.226, hooked over the near wall.
- **The frame to return** (loose, 0.35 kg) hangs on the perch, centre x 0.170.
- Each frame (pine, with a drawn beeswax comb): top bar 448 x 27 x 19 mm with 17.5 mm lugs (483 mm over the lugs), Hoffman end bars 9.5 mm thick and
  35 mm wide at the shoulders (61 mm under the top bar) and 27 mm below, a 20 mm bottom bar, a 25 mm comb, 137 mm
  tall. Hanging, its top is level with the rim and its bottom 9 mm off the table; its end bars clear the end walls by
  9.5 mm.

Sizes: shallow frame 19 x 1 1/16 x 5 3/8 in, 10-frame box inside 18 3/8 x 14 3/4 in, shallow super 5 3/4 in deep,
rebate 5/8 x 3/8 in (dummies.com "Vital stats and materials list for Langstroth frames" and "Cut list for the
ten-frame Langstroth hive"); Hoffman shoulders 1 3/8 in, bee space 1/4 to 3/8 in (American Bee Journal, March 2020,
"Frames and their forgotten diversity, part 2: their spacing").

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to `~/return-frame-to-super`
on the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="Put the frame back in the super, pushed up against the others."

# a policy (here Dream-1), 600 steps = 30 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$HOME/return-frame-to-super/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/return-frame-to-super-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$HOME/return-frame-to-super/task.py -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/return-frame-to-super-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/return-frame-to-super-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

Measured on the returned frame's top-bar centre once everything is still (`SCORE_OVER = "end"`):

- **1.0**: it hangs by both lugs on the super's rebates (top within 2 mm of rim height, tilt <= 2 deg, turned
  <= 3 deg from square, |y| <= 5 mm), no two neighbouring frames (it included) have more than 2 mm between their
  shoulders, all seven other frames still hang on the rebates, nothing moves, and both TCPs are at least 3 cm from
  the frame (released).
- **0.5**: all of that, but a gap between two neighbouring frames is over 2 mm.
- **0** otherwise: on the perch, held, on one lug or its bottom bar, skewed, or any of the seven other frames knocked off its
  rebates at any time during the trial (latched: hanging it back does not undo it).

## What makes this task distinctive

A long, light part has to go up and over before it can go down: the frame's bottom must clear the 146 mm wall, so
it is carried with the top bar at least 28 cm up, then lowered straight down into the 13 cm space between the near
wall and the other frames, so that both 17.5 mm lugs land on 16 mm rests 46.7 cm apart. Once it hangs, it is let go and slid sideways
along the rests until it touches the next frame: a push, not a place.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at step 436 (22 s of trial time); 86 s wall clock including
  Isaac start-up. Both hands lift the frame by its top bar to about 30 cm (tilting at most 3.1 deg on the way), lower it at
  x 0.296 until both lugs sit on the rebates, let go, and push it with the closed fingertips to x 0.357: the shoulder
  gap to the next frame is 0.5 mm, the frame is level (tilt 0.03 deg) and square (0.13 deg), and the seven other
  frames still hang.
- Dream-1: `task_score` 0.0 after the full 600-step run limit; 84 s wall clock. Both arms moved out beside the super's end
  walls, but neither gripped the frame, which stayed on the perch.
- Offline tests: 27 passed.
- Top camera: in the first frame the super, the seven frames, the perch and the frame on it are all visible between
  the arms. In the demo's last frame the perch is empty and the returned frame hangs against the others, with the
  hands raised above it.

## Third-party assets

none
