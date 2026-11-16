# Fold a retail shopping basket handle

An empty, full-size polypropylene hand basket starts upright with its bow handle raised. Fold the handle down for storage while the basket stays on the table. The task is societal relatability: almost everyone has carried one of these baskets.

## Real-world setup

Use a 22 L Intech EK-4330-EG single-handle basket. The top is 430 × 300 × 225 mm, the tapered base is 330 × 205 mm, and the basket weighs 620 g. Put the long side left-right, with the basket centre 280 mm in front of the arm-base midpoint. Raise the bow to its upright stop, 5° past vertical; it folds down 95° to a horizontal stop, pointing away from the arms. The bow's 20 mm section, 120 mm height and pivot location (on the far long wall, 10 mm above the rim) are estimates, as is the 0.4 friction; measure them on a sample. The raised stop angle and low pivot friction stand in for the real hinge detent. The instruction names only the goal.

## Run and score

Run offline tests with `YAM_ARENA=~/yam-isaac-lab-arena python -m pytest -q test_task.py`. On the GPU workstation, run `./workstation/run_task.sh fold-retail-basket-handle` from the repository root. This runs the scripted demo at 20 Hz with a 260-step limit and writes logs, video and a setup screenshot under `~/robocurve_runs/fold-retail-basket-handle/`. Use `-E object_jitter=0` if running Inspect Robots directly. The scripted policy uses fixed waypoints; it does not read object state. The runner uses seed 42 by default. The verified demo ended at 124 control steps (6.2 seconds simulated time), took 25 seconds wall clock on the GPU workstation, and scored 1.0. Over the final two seconds, printed handle angles were 95.01–95.07° from the raised stop, with basket tilt 0° and no latched fault.

Measure handle travel from the raised stop with a side-view protractor or video, basket tilt with a level or side photo, and whether both grippers have released and are at least 30 mm from the bow with video or a ruler, and whether the basket rests on the table (base within 3 mm of the table top, not moving, no gripper within 30 mm of it) with a level or ruler. Check the entire trial for off-table motion (including lifting the basket more than 2 cm), tilt over 30 degrees, or a detached handle. These faults score 0 even if corrected. Otherwise, a handle within 5 degrees of folded for 1 second with both grippers clear and the basket resting on the table scores 1.0 (a basket held in the air scores 0); travel of at least 60 degrees with the basket upright scores 0.5; all other outcomes score 0.

The model uses a hollow five-piece convex basket body, a three-piece bow and a true revolute joint. Its 0.4 polypropylene contact friction is an estimate, as are bow dimensions and its 80 g share of the 620 g whole-basket mass. The hinge has low passive pivot friction (0.015 N m), physical stops (the raised one 5° past vertical) and no spring.

## Verify on Brev

- The basket settles flat without sliding or tipping; its hollow body and rim remain visible in the top camera.
- The open finger surface pushes the top of the bow; the arm clears the basket as the bow rotates.
- The bow reaches and rests against the folded stop without a scripted drive, and both hands withdraw at least 30 mm.
- The reported joint angle, body tilt and fault latch agree with the video throughout the run.

## Third-party assets

none
