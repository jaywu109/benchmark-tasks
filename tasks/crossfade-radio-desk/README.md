# Crossfade from the song on air to the cued one on a radio desk

A radio board operator changes from one source to the next with a crossfade: they bring the fader down on the audio
that is finishing while pushing up the fader of the source that should start, so the change sounds smooth. Pulling
the first fader down before the next is up leaves a silent gap (dead air); pushing the next one up before the first
comes down plays both songs at full over each other. A source normally runs with its fader at the 0 mark. Here the
song on air is on fader 1 at its 0 mark and the next song is cued on fader 8, closed; the green lamps above faders 1
and 8 show both channels switched on.

Instruction: *The song on air is ending: crossfade to the cued one.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; puts each fader cap on a sliding joint with damping), `task.json`, `test_task.py`
(offline checks).

## What is on the table

Positions in metres from the midpoint between the arm bases (x out, y left), heights above the table.

- **Desk** (fixed): a dark grey steel panel 0.20 x 0.36 x 0.04 m centred at x 0.38, y 0.
- **Faders**: eight long-throw faders side by side, 40 mm apart from y +0.14 (fader 1) to y -0.14 (fader 8), left to
  right seen from the operator's seat. Each slides along x over 100 mm of travel in a slot (a black opening 110 x 4 mm cut in the
  panel), closed nearest the
  robot (cap centre at x 0.30) and fully up at x 0.40, with damping and no springs, so it stays where it is left.
  Caps: light grey ABS, 20 x 12 x 16 mm (20 mm along the travel), riding 2 mm above the panel.
- **Marks and lamps**: a 0 mark beside each slot 75 mm up from closed (white paint, 2 x 10 mm); a lamp above each
  slot's far end (a polycarbonate LED lens 10 x 10 x 4 mm), green on faders 1 and 8 (switched on), dark on 2 to 7.
- **Start**: fader 1 at its 0 mark (the song on air); faders 2 to 8 closed.

Source: a community radio station's "Live Panel" operator sheet (plfm01w1.azurewebsites.net/media/1589/
live-panel-operations.pdf): "When changing from one audio source to another, you should do what is called a
'crossfade'. This when you bring the fader down on some audio that is finishing and push up the fader on the audio
source that you want to start"; "the fader labelled as 'Airplay' is switched on and the fader open above 0"; "If
there is nothing on air - call the Helpdesk". Fader travel and the 0 mark at 75 % of it are typical of broadcast
desks; the quarter and half lines below are assumed.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to
`~/crossfade-radio-desk` on the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="The song on air is ending: crossfade to the cued one."

# a policy (here Dream-1), 600 steps = 30 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$HOME/crossfade-radio-desk/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/crossfade-radio-desk-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$HOME/crossfade-radio-desk/task.py -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/crossfade-radio-desk-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/crossfade-radio-desk-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

Checked every control step; the trial ends as soon as it scores 1.0, otherwise the score is read when the run ends.

- **1.0**: fader 8 within 4 mm of its 0 mark and fader 1 within 3 mm of closed, both still, no gripper within 4 cm of
  either cap, and all of this held for 1 s.
- **0**: anything else.
- **Dead air or both up**: if at any step faders 1 and 8 are both in the bottom quarter of their travel (under 25 mm
  up, near silent), or both past halfway (over 50 mm up, two songs near full level), the trial scores 0 whatever
  happens afterwards (kept until the next episode).

## What makes this task distinctive

The rule is on the move, not only on the end state: the two faders have to travel in opposite directions over
75 mm at the same time, so that one is always up past the quarter line and they are never both past halfway. Moving
them one after the other fails either way round (down first is dead air, up first is both up); one hand can pass
only by staging it (cued fader into the 25-50 mm band, outgoing fader closed, then cued fader to its mark), while two
hands, one on each cap 28 cm apart, do it as one smooth move. The
cued fader then has to stop on its 0 mark to within 4 mm. Which faders matter is read from the desk: the one standing up is on air, and the other lit channel is
cued.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at step 246 (12.3 s of trial time, the end state held for 1 s);
  42 s wall clock including Isaac start-up. Each hand pinches its fader's cap fore and aft, then over 4 s fader 1
  comes down and fader 8 goes up together (their sum stayed at 73.9 to 76.5 mm, sampled every 10 steps; at the crossing about 36 mm and
  39 mm), ending at 0.0 mm and 75.9 mm.
- Dream-1: `task_score` 0.0 after the full 600-step run limit; 78 s wall clock. No fader moved.
- Offline tests: 9 passed.

## Third-party assets

none
