# Reconnect a door closer's forearm link to its spring-preloaded main arm and lock the elbow pin so it closes the door

General maintenance and repair workers (SOC 49-9071, 1,529,700 US workers, BLS OEWS May 2025, series OEUN000000000000049907101, https://api.bls.gov/publicAPI/v2/timeseries/data/OEUN000000000000049907101) re-couple surface door closers as routine building upkeep. After a door or frame repair the arm has to be joined again: the closer's spring pulls the main arm back to its rest stop, where its elbow pin is out of reach of the door-side forearm link, so one hand holds the main arm turned against the spring while the other drops the forearm's eye on the pin and locks it. Letting go early snaps the arm back.

Instruction: *"Reconnect this door closer's arm so that it pulls the door shut against its stop."*

## Setup

Two YAM arms at the station, table top z = 0.7475 m, positions in cm from the table origin (x forward, y left). A fixed aluminum door header (45 x 14.5 x 7.2 cm, steel-grey) stands at x 32.75, y -2.5 with its front stop face at x 31.25; a dark-bronze cast-iron closer body (31 x 8.8 x 5.8 cm) is part of it, with the pinion shaft at (26.5, +15.5). A hinged door leaf slice, the top rail section of a door at the closer's mounting height (26 x 4.5 x 6.4 cm, 0.38 kg, solid wood) hangs on a vertical hinge at (29.0, -24.0), starting 18 deg ajar, hinge drive stiffness 0. A door-shoe bracket (2.8 x 2.8 x 1.6 cm, 0.10 kg, steel) swivels on the door leaf at (18.7, -21.1) and carries the forearm link (21.5 x 2.8 x 1.6 cm, 0.18 kg, steel; 18.5 cm between shoe pin and clevis eye; 2.4 cm grip flat; 18 mm eye bore with a 4 mm entry chamfer), spawned swung out 25 deg. The main arm (22.8 x 6.4 x 3.6 cm, 0.24 kg, steel; 19.8 cm from pinion to elbow pin; 2.6 cm grip bar at 8.8 cm from the pinion) turns on the pinion shaft, rests at its slack stop with a 0.022 N m/deg spring drive (0.93 N m, 10.5 N at the grip bar, at 42.2 deg) and can be pulled up to 42.2 deg; its elbow pin is 14 mm across, 18 mm tall with a 5 mm lead-in. A red quarter-turn lock lever (5.2 x 1.2 x 1.9 cm, 0.04 kg, steel) sits on the arm's knuckle beside the pin and turns clockwise over the eye. Gripper opening is 9.4 cm; every grasped part is at least 12 mm thick. A 6 mm steel spring-ball detent (1 g) is guided vertically on the main arm, 12 mm outward from the lock pivot; it enters a 5 mm wide, 2 mm deep underside groove at 90 deg. Its 1500 N/m spring has 2 mm preload and 2.5 mm travel.

## Scoring

Checked at the end of the run; 1.0 held for 1 second ends the trial. 1.0: the forearm's eye sits on the elbow pin (centres within 4 mm, 4 mm height gap), the lock lever is turned clockwise 75 deg or more over the eye, the main arm is still drawn at least 20 deg against its spring, the door is closed to within 2.5 deg of its stop, nothing moves faster than 5 cm/s (0.2 rad/s angular) and both grippers are at least 4 cm from every part. 0.7: eye seated, main arm drawn 20 deg or more and lever turned 75 deg or more, but the door is not within 2.5 deg of its stop, or something still moves (5 cm/s linear or 0.2 rad/s angular), or a gripper is closer than 4 cm to a part. 0.35: eye seated with the main arm drawn 20 deg or more, lever not turned 75 deg. 0 otherwise. Latched to 0 for the rest of the trial: the lever turned past 45 deg before the eye is seated, or the door pushed against its stop with more than 25 N by a robot link. On the real parts: feeler gauge for the eye-to-shoulder gap and the closed door gap, protractor for the arm and lever angles, force gauge on the door edge. The score only reads the end state and the two latched faults; it does not check which hand does what. Real trial: a screw-down check in place of the lever.

## What the scorer enforces, and the two hands

`task.py` scores only the end state and the two latched faults above; it does not check which hand does what.
What forces two hands is the geometry: with the main arm at its slack stop the elbow pin is about 303 mm from the
door-shoe pin, but the forearm link reaches 185 mm. With the door left ajar the eye meets the pin only with the main arm
held at about 42 deg against its spring; if the door is pushed shut first, about 27 deg is enough. Either way the arm
must be held off its stop until the pin is seated, or it snaps back. The scripted demo
keeps the left hand on the main arm from the pull until after the lever is turned, while the right hand brings the
forearm over the pin, seats it, lets go, then grips the lever and turns it about 95 deg; the left hand then releases and
the spring draws the door shut. The lever acts by rigid contact only (fixed joint limits): a lever turned more than 45 deg
sits over the eye's path and the eye cannot be seated. A 6 mm spring-loaded ball detent
(next section) holds the lever near 90 deg after release; the scorer does not test the detent itself, only that the lever
reads 75 deg or more at the end.

## Real parts

The LCN 4040XP installation sheet for the top-jamb and push-side mount says: "Pre-load arm to 90°.
Fasten with screw provided." (Allegion LCN, `Inst_4040XP_Top_Jam-PushSide.pdf`, fetched from
https://cdn.shopify.com/s/files/1/1415/1594/files/Inst_4040XP_Top_Jam-PushSide.pdf). Here the preload is 42 deg and a
quarter-turn lever stands in for the screw. The LCN catalog and cut sheet give the reveal and opening limits but not the
main-arm and forearm lengths, so the 228 mm and 215 mm arms are assumed sizes (not scaled down).

Spring: the modelled main-arm spring is 0.022 N m/deg, so 0.93 N m at the 42.2 deg stop (10.5 N at the grip bar, 88 mm
from the pinion). A real closer is much stronger: EN 1154 asks a grade 3 closer for a closing moment of at least 18 N m
between 0 and 4 deg (EN 1154:2003 classification table, https://cdn.shopify.com/s/files/1/0005/0862/4949/files/EN1154_2003_r0_Controlled_door_closing_devices_-_classification_En.pdf),
about 19x our 0.93 N m (the two torques are at different arm angles). The LCN sheets quote no torque, so the real arm's
90 deg preload torque is not known here. The spring is softened to the range of the YAM wrist: the pull runs along the
gripper's closing direction, so one finger face pushes the bare bar, and a closed finger held about 26 N in the GPU
GPU contact test on the bench, which at 0.088 m is 2.3 N m, about 2.5x our spring. Friction on the
pads does not bind here, because the load is not along the pad faces. The arms carry no grip collars, as on the real part.

## Real trial

Mount a real surface closer of this kind (LCN 4040XP or Norton 1601, regular arm) on a real door and frame, with the
forearm uncoupled and the door ajar. The robot pulls the main arm to the 90 deg preload the LCN sheet asks for, seats the
forearm's eye on the elbow stud and drives the stud screw in place of the quarter-turn lever (pass: screw seated, eye
cannot lift). Then it lets go. Measured: the eye-to-shoulder gap with a feeler gauge (4 mm or less), the main arm angle
with a protractor at the pinion (20 deg or more from its rest stop with the door shut), the closed-door gap at the stop
with a feeler gauge (2.5 deg is about 4 cm at a 0.9 m latch edge for the full leaf), and the push on the door edge with a
force gauge held between the robot and the door (25 N limit).

## Reset

Deterministic layout (object_jitter=0): set door_leaf to 18.0 deg ajar, set door_shoe and forearm_rod to +25.0 deg (+90.0 deg world yaw) at 0.0 m vertical slide, set main_arm to 0.0 deg slack stop (-48.0 deg world yaw), set elbow_lock to 0.0 deg unlocked, set elbow_detent ball to its spring-preloaded top position, and clear all latched faults.

## Simulation

The six original parts (plus the 6 mm detent ball, a sphere) are rigid bodies on revolute and prismatic joints (`robocurve_ext.py` builds the joints). The main arm has a spring drive toward its slack stop; the door leaf has none, so it closes only through the coupled linkage. Run limit: 580 steps (29 seconds at 20 Hz).

## Elbow lock detent

A 6 mm steel ball (1 g) sits 12 mm outward of the lock pivot, guided vertically relative to the main arm by a
prismatic joint with 2.5 mm travel. A constant 1500 N/m spring with 2 mm preload presses it upward. The lever's cam is
52 x 12 x 5 mm with symmetric 5 mm wide, 2 mm deep underside grooves at +/-12 mm along its length. At 90 deg clockwise
the outward groove seats over the ball; turning away compresses the ball against the groove edges. The spring and guide
are idealized by the joint; there is no modeled cartridge housing.

## Third-party assets

none
