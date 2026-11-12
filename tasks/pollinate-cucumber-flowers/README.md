# Hand-pollinate the female cucumber flowers with a brush

Cucumbers, squash and melons bear separate male and female flowers, and fruit sets only when pollen is carried from
a male to a female. Where bees are too few, growers do it by hand with a small artist's paintbrush: load the brush on
a freshly opened male flower's anthers, then brush the stigma in the centre of each female. A female is recognised by
the miniature fruit (its ovary) behind her petals; cucumber males sit on very short stalks, in clusters. Here a
stretch of vine carries four open flowers, two of each, and the brush lies on its rest.

Instruction: *Hand-pollinate the female cucumber flowers with the brush.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; builds the vine, each flower and the brush as compound rigid bodies of box pieces, each
piece with its own colour), `task.json`, `test_task.py` (offline checks).

## What is on the table

Positions in metres from the midpoint between the arm bases (x out, y left), heights above the table.

- **Vine** (fixed, 7 mm thick, 360 mm long): tied along a 2 mm steel support wire across the table at x 0.45, 0.12 up,
  between two wooden posts (12 mm square, 150 mm tall) at y -0.22 and +0.22; five 65 mm leaves stand above the wire.
- **Four open flowers** (fixed), facing the robot, 40 mm across, five yellow petals and an orange centre. From right
  to left:
  - female at y -0.12, centre 0.110 up;
  - male at y -0.04, 0.125 up;
  - female at y +0.04, 0.110 up;
  - male at y +0.12, 0.125 up.
- Each **female**: an 8 mm stigma, its front at x 0.400; behind the petals a dark green miniature cucumber (35 mm
  long, 9 mm thick) on a stalk to the vine.
- Each **male**: a 5 mm anther column, its front at x 0.422, on a very short stalk, with two closed buds (16 x 6 x 6 mm) of its
  cluster beside it.
- **Brush** (loose, 12 g, 178 mm): a red lacquered-wood handle 7 mm square, a metal ferrule, a cream
  synthetic-bristle tuft 6 mm thick. It lies along x on a rest at y -0.20, the tip pointing out at x 0.289: two grey painted-wood blocks,
  12 x 30 x 30 mm (30 mm tall), 105 mm apart.

Source: University of California Cooperative Extension, Leaflet 21242, "Fruit Set Problems in Squash, Melons, and
Cucumbers in Home Gardens" (H. Johnson Jr.): "pollen from the male flower must be transferred to the female flower";
"You can use a small artist's paintbrush to transfer pollen ... onto the stigma in the center of the female flower";
"The female flower in cucurbits can be recognized easily by the presence of a miniature fruit (ovary) at the base of
the flower"; "In melons and cucumbers, male flowers have very short stems and are borne in clusters of three to five,
while the females are borne singly on somewhat longer stems"; "Flowers open early in the morning and are receptive
for only one day". University of Maryland Extension, "Vegetable Plant Pollination in a Changing Climate": the anther
cones of male flowers "paint pollen grains on the stigma of female flowers". Flower, leaf and brush sizes are
assumed.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to
`~/pollinate-cucumber-flowers` on the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="Hand-pollinate the female cucumber flowers with the brush."

# a policy (here Dream-1), 600 steps = 30 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$HOME/pollinate-cucumber-flowers/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/pollinate-cucumber-flowers-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$HOME/pollinate-cucumber-flowers/task.py -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/pollinate-cucumber-flowers-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/pollinate-cucumber-flowers-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

Tracked every control step (`SCORE_OVER = "end"`, the credit is latched until the next episode).

- The tip (the end of the tuft) is **on** a stigma or anther column when it is within 4 mm of its front centre and
  no more than 1 mm in front of its face, so a tip hovering in the air in front of a flower does not count.
- A **stroke**: the tip stays on the same face while it moves at least 5 mm across it (sideways or up and down)
  from where it first touched. A still touch is not a stroke; leaving the face starts the stroke over.
- The brush **picks up pollen** with its first stroke across a male's anther column and keeps it for the rest of
  the trial.
- A female is **pollinated** by a stroke across her stigma after that.
- **0.5 per female pollinated**, 1.0 for both. A female stroked before the brush carries pollen earns nothing for
  that stroke but can be stroked again. Touching a flower with anything other than the brush tip earns nothing.
- Full score held for 1 s ends the trial; otherwise the score at the end of the run.

## What makes this task distinctive

The robot has to carry pollen the way a grower does: first load the brush with a stroke across a male flower's
anthers, then stroke it across the stigma of each female; a female counts only once the brush has been drawn across a male's anthers, and stroking a male again does no harm. Males and females alternate along the vine, starting with a female at the right, so a plain sweep from
right to left strokes the first female before the brush carries any pollen and earns half. Every stroke is made
through the tool: the end of a light brush's tuft, well ahead of the fingers, drawn 5 mm across a face a few
millimetres wide without lifting off it.

## Verification

Run on an RTX 5090 with `yam-isaac-lab-arena` at `bd62771`, using the commands above (`object_jitter=0`, default
seed):

- Scripted demo: `task_score` 1.0, ended by success at about step 380 (19 s of trial time); 51 s wall clock including
  Isaac start-up. The right hand takes the brush by the handle and draws the tip right to left across male 1's
  anthers (y -0.044 to -0.038, pollen loaded), female 1's stigma (-0.124 to -0.118) and female 2's (+0.036 to
  +0.041), each with the tip at the face.
- Dream-1: `task_score` 0.0 after the full 600-step run limit; 66 s wall clock. The right hand came down over the
  brush on its rest and pushed it about 10 mm sideways, but never lifted it.
- Offline tests: 17 passed.

## Third-party assets

none
