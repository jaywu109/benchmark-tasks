# Leave a parcel in a curbside mailbox: raise the carrier flag the instruction never names, lift the hanging door shut

The mailbox's front door hangs 82 deg open below the mouth. The hand has to lift it from below, up its hinge arc
against its own weight, and press it the last few degrees past a roller catch that only takes it within 4 deg of
closed. The instruction never mentions the red flag, but leaving mail for the carrier means raising it. This is the
familiar household chore of leaving outgoing mail in a curbside mailbox. The parcel has to end up completely past the
sill, or the door cannot close.

Instruction: *Leave this parcel in the mailbox for the carrier to collect.*

Files: `task.py` (the task file: objects, `success`, `distance`, `score`, `demo`), `robocurve_ext.py` (shipped in the
folder, no `yam_arena` edits; builds the `1:1` `Gibraltar E1100B00` mailbox tunnel, bottom-hinged front door, and
right-side signal flag as compound rigid bodies of box pieces with revolute joints), `task.json`, `test_task.py`
(offline checks).

## Real-world references

- USPS Standard 7C (curbside mailboxes) asks for a carrier signal flag on the right side seen from the front, and a
  door that opens by pulling outward and downward. These are paraphrased, not quoted. The task does not score the
  standard's force limit.
- Gibraltar Elite Medium curbside mailbox (model E1100B00), about `175 x 225 x 509 mm` (`6.9 x 8.87 x 20.05 in.`).
- USPS Small Flat Rate Box, `8-5/8 x 5-3/8 x 1-5/8 in.` (`219 x 137 x 41 mm`). The `0.25 kg` mass is assumed.

## What is on the table

Positions in metres from the midpoint between the arm bases (`x` straight out, `y` to the left), heights above the
table (`TOP = 0.7475 m`).

- **Gibraltar E1100B00 curbside mailbox body** (`mailbox`, fixed, powder-coated galvanized steel on a `16 mm` wooden
  mounting board): `509 mm` deep along `x` (`x = 0.300..0.809 m`, with its front mouth at `X_SILL = 0.300 m` and its
  rear extending beyond the reach zone to `0.809 m`) x `175 mm` wide along `y` (`y = -0.1075..+0.0675 m`, centred at
  `y = -0.020 m`) x `225 mm` tall along `z` (`z = TOP..TOP + 0.225 m`). The clear interior tunnel is `159 mm` wide
  (`y = -0.0995..+0.0595 m`) with its floor at `z = TOP + 0.026 m` and a top roller-catch keeper bracket at the upper
  mouth (`z = TOP + 0.207..TOP + 0.213 m`).
- **Bottom-hinged front carrier service door** (`door`, `190 g`, charcoal steel leaf with rolled top pull flange):
  `13 mm` deep x `175 mm` wide x `196 mm` tall, hinged along `Y` at `(0.296, -0.020, TOP + 0.014 m)` from `-82.0 deg`
  open to `0.0 deg` vertical closed. Starts open at `-82.0 deg` and latches into its top roller-catch detent
  (a scripted catch, see below) once carried within `4.0 deg` of closed.
- **Right-side outgoing carrier signal flag** (`flag`, `45 g`, high-contrast fluorescent orange-red steel/plastic arm
  and paddle): `170 mm` long x `16 mm` thick x `42 mm` tall, mounted on the right side wall (`y = -0.122 m` when facing
  the front mouth, per `USPS-STD-7C § 3.6`), pivoting about `+Y` at `(0.355, -0.122, TOP + 0.135 m)` from `0.0 deg`
  (horizontal, pointing forward toward `-x`) up to `+90.0 deg` (vertical upright) into a detent keeper
  (a scripted detent, `FLAG_DETENT_STIFFNESS = 0.012 N*m/deg`, snapping to 90 deg past 68 deg).
- **Prepaid outgoing USPS Small Flat Rate Box** (`parcel`, `0.25 kg`, sealed rigid cardboard): `219 mm` long (`x`) x
  `41 mm` wide (`y`) x `137 mm` tall (`z`), standing on its `219 x 41 mm` spine to the left (`+y` side) of the mailbox
  at `(0.260, +0.210, TOP + 0.0685 m)`.

## Run and score a trial

Built against the YAM Isaac Lab-Arena embodiment at commit `bd62771`. Copy this folder to
`~/post-parcel-curbside-mailbox` on the GPU machine. Always run with `-E object_jitter=0`.

```bash
export INSTR="Leave this parcel in the mailbox for the carrier to collect."

# a policy (here Dream-1), 500 steps = 25 seconds at 20 Hz
inspect-robots run --embodiment yam_arena --policy dream -E task=$HOME/post-parcel-curbside-mailbox/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score --max-steps 500 --store-frames --log-dir ~/yam-eval/post-parcel-curbside-mailbox-dream

# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$HOME/post-parcel-curbside-mailbox/task.py -E object_jitter=0 -E control_hz=20 --instruction "$INSTR" --scorer task_score --max-steps 500 --store-frames --log-dir ~/yam-eval/post-parcel-curbside-mailbox-demo

# one MP4 per camera, in each log's frames/ folder
for log in ~/yam-eval/post-parcel-curbside-mailbox-*/*.json; do inspect-robots video "$log"; done
```

The run prints `task_score` at the end. Offline checks: `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q
test_task.py` (NumPy and pytest only).

## Scoring

Evaluated at the end of the episode (`SCORE_OVER = "end"`, `SUCCESS_HOLD_S = 1.0 s`).

- **1.0**: the parcel is completely inside the mailbox tunnel past the front sill (`parcel.low[0] >= X_SILL = 0.300 m`,
  `parcel_inside`), the front door is closed and latched (`door_open_deg <= DOOR_CLOSED_MAX_DEG = 3.0 deg`), the
  carrier signal flag is upright (`flag_angle_deg` within `90.0 +/- 6.0 deg`, `FLAG_UP_TOL_DEG = 6.0 deg`), both
  gripper TCPs are at least `25 mm` (`GRIPPER_CLEAR = 0.025 m`) clear of the mailbox, door, flag, and parcel, and
  all moving parts are at rest (`all_resting`), held for `1.0 s`.
- **0.0** otherwise (no partial credit: parcel in and door shut with the flag down is the plain put-in-and-latch job, so it
  earns nothing), or if a latched fault is triggered at any step during the episode:
  - `door_forced_on_parcel`: the front door is forced against the parcel at any door angle (door-to-parcel contact
    force `> MAX_DOOR_PARCEL_FORCE_N = 15.0 N`, read via PhysX `RigidContactView`). The catch alone presses a parcel
    resting against the closed door with well under `1 N`, so only forcing trips it.
  - `parcel_off_table`: the parcel falls below or off the station table.
- **Measured on real objects**: parcel clearance past the front sill (`>= 0 mm` inside `X_SILL`) from a straightedge
  across the mailbox mouth before closing the door; door closure (`<= 3 deg`, top roller catch engaged) and flag
  upright angle (`90 +/- 6 deg` vertical) with an inclinometer or protractor against the mailbox side wall; door-parcel
  forcing (`> 15 N`) from a force gauge on the door or visible crushing of the cardboard box; no visible dent in the
  box from the gripper; gripper clearance (`>= 25 mm`) and `1 s` rest hold from a ruler and video timestamps.

## Reset

Nothing is consumed. Lower the right-side carrier signal flag to `0 deg` horizontal, swing the front carrier service
door open to `-82 deg`, and place the USPS Small Flat Rate Box upright on its `219 x 41 mm` spine at
`(0.260, +0.210, TOP + 0.0685 m)`.

## How the door catch works (a scripted drive, not free physics)

The door is a real revolute joint driven by gravity and the hand until it comes within `4 deg` of closed
(`DOOR_CATCH_ENGAGE_DEG`), so the hand has to carry it almost all the way up and press it past the roller. Then
task.py switches the joint's drive to a spring toward `0 deg` with stiffness `0.016 N*m/deg`, which draws the door
shut and holds it (about `0.064 N*m` at the edge of the window, about `0.3 N` at the door's top edge; at most
`0.14 N*m`, about `0.7 N`, just before it lets go). If the door is pulled back past `9 deg` the drive is switched back
to the light open-hold, so the catch has a hysteresis of `5 deg`. A real roller catch is a spring-loaded roller on a
keeper; this models its window and holding torque, not its geometry. No drive closes the door from further open. The
flag detent is the same kind of drive (snaps up past `68 deg`).

## What makes this task distinctive

The door is not pushed down like a lid: it hangs 82 deg open below the mouth, and it has to be lifted from below
against its own weight along its hinge circle, then pressed through its last degrees past the roller, which takes it
only within 4 deg of closed. The instruction never mentions the flag; raising it is the signal the carrier reads, so
the robot has to know what leaving mail for collection means. The steps: (1) carry the `219 x 137 x 41 mm` parcel past
the front sill so its rear end no longer blocks the door arc; (2) lift the bottom-hinged door along its hinge circle
and press it past the catch. The scripted demo slides open jaws in from the side around the door's top edge and lifts
it to 28 deg. The jaws cannot follow the door shut without hitting the roof, so the hand slides up to the edge, rolls
the wrist onto its lower fingertip and presses the pull flange along the arc to 1 deg; (3) raise the right-side flag
`90 deg`. The flag may go up at any time (the demo raises it first); only the end state counts.

## What is modelled at real size, and what the sim simplifies

- Real size: the mailbox (`175 x 225 x 509 mm`) and the flat rate box (`219 x 137 x 41 mm`).
- Simplified: the mailbox body is a fixed compound of boxes; the door and flag are compound boxes on revolute joints
  with the scripted drives above; the parcel is a sealed rigid box (`0.25 kg` assumed).

## Verification

Run on the workstation GPU with `yam-isaac-lab-arena` at `bd62771`, `object_jitter=0`, default seed: the scripted
demo reaches `task_score` 1.0 (flag upright by step 80, door 69.5 deg at step 280 and shut by step 390, success
after step 420 of 500).

Dream-1: `task_score` 0.0 after the full 500 steps (59 s wall clock): it picked up the parcel and carried it around
(lowest x 0.06-0.08 m, never past the 0.300 m sill), the door stayed open and the flag stayed down; no fault.

## Third-party assets

none
