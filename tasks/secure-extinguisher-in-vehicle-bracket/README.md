# Secure a fire extinguisher in its vehicle bracket

A full-size 2.5 lb extinguisher lies nested in its quick-release vehicle bracket, which is installed horizontally with the latch facing up. The strap is around the discharge head and the latch loop is already on its hook, but the yellow latch lever stands raised. The maker's installation step that remains is to flip the lever down. The instruction is: **Secure the fire extinguisher in its vehicle bracket.**

The lever is not free to swing down. The strap's tension acts on it through the loop, so the lever pushes back toward open until it passes the over-centre point at 32 degrees. Past that point the same tension holds it closed against its stop. A lever left short of the catch springs back open, and a lever closed by force that unhooks the loop or pulls the extinguisher out of its seat scores 0.

## Real-world setup

- **Extinguisher:** a Rusoh Eliminator 2.5 lb ABC extinguisher, model 0930003 (175 x 167 x 410 mm, 3.155 kg).
- **Bracket and plate:** its Rusoh PN 81-674 vehicle bracket (146 x 127 x 277 mm, 0.34 kg), screwed to a 430 x 210 x 18 mm wood board. The board is the sturdy mounting surface the installation sheet asks for.
- **Placement:** put the board's long side left-right, centred 245 mm in front of the arm-base midpoint and 304 mm to the right of it.
- **Bracket orientation:** install it horizontally with the latch up, as the sheet allows. The extinguisher's bottom rim rests against the bracket's end tab and its head points left.
- **Start state:** hook the latch loop and raise the lever to its open stop, about 100 degrees from closed.

How to measure:

- the lever angle, with a protractor or side video;
- extinguisher and bracket movement, with a ruler against marks on the board;
- whether the loop is still on the hook, from a photo;
- whether both gripper centre points are at least 30 mm clear of the lever and the extinguisher, from video.

The strap, loop, hook and lever dimensions are nominal, inferred from the bracket's envelope. Measure them on a sample: 70 mm lever, 22 mm wire loop with a 10 x 14 mm hole, 6 mm hook peg.

## Run and score

Offline tests (numpy and pytest only):

```bash
YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q tasks/secure-extinguisher-in-vehicle-bracket/test_task.py
```

On the GPU workstation, from the repository root:

```bash
./workstation/run_task.sh secure-extinguisher-in-vehicle-bracket
```

The runner plays the scripted demo at 20 Hz with `object_jitter=0`, a 500-step limit and the default seed. It writes logs, videos and `setup.png` under `~/robocurve_runs/secure-extinguisher-in-vehicle-bracket/`.

**Demo:** the right hand comes in from the side with closed pads and sweeps the lever down along its arc. It presses the lever onto its stop, lifts off and parks clear to the right. The left arm does not move. The scripted policy uses fixed waypoints and reads no object state.

**Demo result:** 1.0, ending at about 315 control steps (15.8 s simulated) after 41 s of wall clock. Over the full-credit hold, the measurements were:

- lever angle 0.33-0.34 degrees, varying by 0.007 degrees;
- extinguisher 0.2 mm from its start position in x/y and 1.0 mm lower, tilted 0.1 degrees, moving 0.5 mm/s;
- loop hole 2.2 mm from the peg;
- no fault.

Scoring:

- **0 for the whole trial** if any of these happens at any time:
  - the loop's hole goes more than 12 mm off the hook peg, or more than 10 mm along it;
  - the extinguisher moves more than 15 mm sideways, lifts more than 8 mm, sinks more than 4 mm or tilts more than 5 degrees;
  - the bracket moves more than 10 mm or tilts more than 5 degrees;
  - a latch part is pulled more than 10 mm off its joint.

  A trial whose lever is not raised 90-105 degrees at the first measured step also scores 0.
- **0.5** once the lever has been lowered through 45 degrees of its arc (to 55 degrees or less) with no fault.
- **1.0** when all of these have held for 1 s:
  - the lever is within 10 degrees of its closed stop, which is past the catch;
  - both gripper centre points are at least 30 mm from the lever and the extinguisher (the scorer reads the grippers only, not the arm links);
  - the lever's angle has varied by less than 0.5 degrees over the last second;
  - the extinguisher is still (under 1 cm/s and 0.5 rad/s).

  Full score held for 1 s ends the trial.

**How the lever's stillness is read:** from its angle, not its velocity. The lever sits on its hard stop under strap tension. There the simulator reports a velocity for it (about 75 mm/s and 2 rad/s in the demo) even though its angle does not change.

**How the latch is modelled:** the strap's stretch is a 600 N/m spring along the strap's free end, with 24 N of tension when the lever is closed. Through the loop's geometry this gives the catch at 32 degrees and about 0.27 N m holding the lever closed. Passing the catch takes about 3 N, pushing 45 mm out along the lever. There is no release button, no scripted latch and no spring on the lever itself.

## Verify on Brev

- In the first top frame, the extinguisher sits nested and level, the loop is on the hook and the lever is raised. Nothing moves during the settle.
- The lever is driven down only by the hand's contact. Short of the catch it springs back toward open; past the catch it closes onto its stop and stays there after the hand lifts away.
- The loop stays on the hook peg and the extinguisher stays in its seat while the lever is pushed.
- In the last top frame, the lever lies flat across the neck with the right arm parked to the side. From this camera the parked right arm hides the extinguisher's bottom end; the lever stays in view. Only its gripper is scored for clearance.

## Third-party assets

none. The extinguisher and hook-peg meshes are generated by `make_mesh.py` in this folder.
