# Cue a record at the start of its third track

Putting on a particular song from a vinyl record is a familiar everyday activity for anyone with a record player at home.
With the platter stopped, you lift the tonearm by the small finger lift on its headshell, swing it over the record and lower the
stylus by hand into the smooth gap before the song you want, then let go, so that pressing start plays that song from
its first note. The wanted song is found by counting the gaps from the edge. A stylus dropped onto the vinyl can
chip it, and one dragged across the grooves scratches the record. Here the LP's five tracks are of unequal length, and the arm lies in its rest clip.

Instruction: *Cue the record at the start of track 3 and leave it ready to play.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; builds the tonearm as two hinged bodies, a yaw yoke and a pitch arm, each with damping
only, and the deck as fixed pieces), `task.json`, `test_task.py` (offline checks), `disc_*.obj` (96-sided prisms for
the platter and the record's rings).

## What is on the table

Positions in metres from the midpoint between the arm bases (x out, y left), heights above the table.

- **Plinth** (fixed): walnut, 0.44 x 0.38 x 0.04 m, centred at x 0.43, y 0.04.
- **Platter** (fixed, stopped): aluminium, 304 mm across, 10 mm thick, spindle at x 0.40, y 0.06.
- **Record** (fixed): a 300 mm vinyl LP, 2 mm thick. From the edge in: a 3 mm smooth edge; then five tracks, dark
  grooved bands 18, 12, 22, 9 and 10 mm wide; a lighter smooth gap 4 mm wide before each of tracks 2 to 5; a smooth
  run-out and a red paper label 100 mm across. The gap before track 3 lies 109 to 113 mm from the spindle.
- **Tonearm**: an aluminium tube 8 mm thick, 229 mm from the pivot to the stylus, 120 g with its 35 mm counterweight,
  balanced to put about 2.4 g on the stylus. The counterweight is a 35 mm steel cube on a 50 mm aluminium stub
  behind the pivot. It turns about a vertical bearing and a horizontal one, on an aluminium bearing post
  20 x 20 x 22 mm, at its pivot at x 0.53, y -0.11, 20 mm above the record, with damping and no springs, so it stays
  where it is left. Masses are set to balance the arm, not derived from the materials.
- **Rest**: a black ABS post 12 x 12 mm, 28 mm tall, 150 mm out from the pivot, with two black ABS clip fingers
  12 x 4 x 8 mm, 11 mm apart; the arm lies in it pointing toward the robot, the stylus off the record to its right.
- **Headshell**: black ABS, 30 x 18 x 8 mm, carrying a blue ABS cartridge 18 x 12 x 10 mm and its stylus (an aluminium
  cantilever with a diamond tip, 3 x 3 x 2 mm).
- **Finger lift**: sticks out 22 mm from the right side of the headshell and rises as an aluminium fin
  12 x 5 x 30 mm.

Sources for the handling procedure: a college radio station's guide "Playing vinyl" (techdocs.wrbbradio.org): "Only pick the headshell up with your
pointer finger under the small hook shaped arm"; "keep the first track off, but
cued". KB Audio, "Can you skip tracks on a vinyl record?": the spaces between tracks show "as slightly wider gaps";
"Count the number of gaps from the start of the record to figure out which track is which"; lower the arm gently and
"release it slowly to prevent any sudden drops". Arm sizes follow a 9-inch arm; track widths and gap width are
assumed.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to
`~/cue-record-track-three` on the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="Cue the record at the start of track 3 and leave it ready to play."

# a policy (here Dream-1), 600 steps = 30 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$HOME/cue-record-track-three/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/cue-record-track-three-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$HOME/cue-record-track-three/task.py -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/cue-record-track-three-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/cue-record-track-three-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

Checked every control step; the trial ends as soon as it scores 1.0, otherwise the score is read when the run ends.

- **1.0**: the stylus tip rests on the record (no more than 1.5 mm above its surface, the arm still) inside the gap
  before track 3 (109 to 113 mm from the spindle), no gripper is within 8 cm of the finger lift or within 5 cm of
  the arm between its pivot and the stylus, and all of this holds for 1 s.
- **0.5**: the same, but in one of the other three gaps.
- **0**: anything else, including the stylus on a grooved band, the smooth edge, the run-out or the label, off the
  record or still held up, the arm moving, a gripper at the finger lift or on the arm, or the arm back on its rest.
- **Dropped or dragged**: if the stylus ever meets the record moving faster than 5 cm/s, or slides more than 3 mm
  across the record from where it touched down (lifting it clear starts over), the trial scores 0 whatever happens
  afterwards (kept until the next episode).

## What makes this task distinctive

The robot works a hinged arm by its finger lift, a fin 5 mm thick at the side of the headshell. Held there, the arm swings about its own pivot, so the hand has to follow the arm's arc while lifting it
clear of the rest clip, then lower the stylus slowly onto a 4 mm gap and let go without nudging it. The gap is chosen
by counting: the tracks are of unequal length, so the third gap from the edge cannot be found by spacing, only by
counting the lighter rings. Letting the arm fall onto the vinyl, or setting it down early and sliding it into the gap, ends the trial at 0,
however well it ends up placed.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at step 337 (17 s of trial time, the cued state held for 1 s); 49 s wall clock including
  Isaac start-up. The right hand pinches the finger lift from above, raises the stylus 20 mm, swings the arm about
  24 degrees over the record, lowers it slowly (well under 5 cm/s) into the gap before track 3 at 111.5 mm from the
  spindle, opens and moves away.
- Dream-1: `task_score` 0.0 after the full 600-step run limit; 72 s wall clock. The tonearm stayed on its rest the
  whole trial (the stylus moved under 0.5 mm).
- Offline tests: 21 passed.

## Third-party assets

none
