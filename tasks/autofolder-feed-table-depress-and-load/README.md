# Depress an autofolder feed table and load a squared newsletter stack to the retarder stop

In high-volume office mailrooms, print shops, school and church administrative offices, and billing departments,
friction-feed desktop paper folders such as the Martin Yale 1501X AutoFolder
(https://www.martinyale.com/product/1501x-paper-folder/, machine net weight `31 lb` / `14.1 kg`) fold up to 150 sheets
of 20 lb bond paper at a time. Per the Martin Yale 1501X operation manual (source named only; the
steps are described here in our own words), loading goes like this: put a neat
stack on the tray, push the spring-raised feed table down, slide the stack between the side guides until its front edge
meets the retarder frame, then let the table come back up so the stack presses against the overhead rubber feed roller.

Because the upward spring holds the empty feed table at `0.0 deg` with only a `7.4..8.9 mm` vertical gap under the
`58 mm` diameter overhead rubber feed wheel, the `20.0 mm`-thick squared newsletter stack is physically blocked by the
feed wheel unless one hand holds the feed table depressed (`~-14.0 deg`, opening a `~39 mm` vertical gap under the
wheel) while the other hand slides the stack between the `223 mm` side paper guides up to the retarder stop. If the
holding hand releases the feed table before the stack enters the nip, the feed table springs back to `0.0 deg` and
blocks the stack. This is why the job takes two hands: the table rises the moment it is released, so one hand must
keep it down while the other slides the stack. The scorer does not check which hand does what or in what order; it
checks the end state, the latched faults, and (for 0.5) that the table was pressed to <= -12 deg before the stack
reached the stop. In the scripted demo the left hand presses the paddle first, the right hand then slides the stack,
and the left hand lets go only after the right hand has opened and retracted. Once the stack reaches the retarder stop and the holding hand lifts off the feed table, the constant
passive spring lifts the feed table and stack until the top face of the `20.0 mm` stack clamps against the underside of
the overhead rubber feed wheel at `-5.5 deg` (inside `-7.5` to `-3.5 deg`).

According to the U.S. Bureau of Labor Statistics Occupational Employment and Wage Statistics (BLS OEWS May 2024),
this document-preparation and mailroom workflow spans **Office Clerks, General** (SOC `43-9061`, `2,487,670` U.S.
workers, https://www.bls.gov/oes/current/oes439061.htm) and **Secretaries and Administrative Assistants, Except Legal,
Medical, and Executive** (SOC `43-6014`, `1,722,060` U.S. workers, https://www.bls.gov/oes/current/oes436014.htm).

## Instruction

> Load this newsletter stack into the folder so it is ready to feed.

## Setup (`SCENE = "station"`, `CONTROL_HZ = 20.0`, `SCORE_OVER = "end"`, `SUCCESS_HOLD_S = 1.0`)

- **AutoFolder Housing (`autofolder_housing`, free `Box` + `rx.Compound`, `252 x 380 x 176 mm`, `14.1 kg` / `31 lb`):**
  Powder-coated steel frame resting on the station table on non-skid rubber feet at `(0.255, +0.010, TOP + 0.088)` m
  (matching the `31 lb` / `14.1 kg` net machine weight of the Martin Yale 1501X per
  https://www.martinyale.com/product/1501x-paper-folder/). Includes:
  - Right-hand entry staging apron (`y = -0.180..-0.046 m`, `z = TOP + 0.114 m`) with a `104 mm`-wide central
    finger-clearance slot (`x = 0.203..0.307 m`) between the front and rear support decks,
  - Fixed pre-set side paper guides spaced `223.0 mm` apart along `x` (`x = 0.1435..0.3665 m`, giving `3.5 mm`
    clearance per side for `216.0 mm` letter-width paper),
  - Overhead feed-shaft arch and upper retarder frame (`y = +0.076..+0.136 m`), and
  - Central `58 mm` diameter rubber feed wheel (`46 mm` wide along `x = 0.232..0.278 m`, centred at `y = +0.092 m`)
    whose stepped `-5.50 deg` bottom contact contour leaves a `7.38..8.92 mm` empty nip gap above the `0.0 deg` feed
    table (`<< 20.0 mm` stack thickness) and contacts the top face of the `20.0 mm` stack flat across `y = +0.080..+0.104 m`
    at `theta_table = -5.50 deg`.
- **Spring-Elevated Feed Table (`feed_table`, free `Box` + `rx.Compound` + `rx.Revolute`, sheet steel, `210 x 226 x 28 mm`, `0.240 kg`):**
  Hinged on `autofolder_housing` (`parent = "autofolder_housing"`) about the horizontal `X` axis at world
  `(0.255, -0.036, TOP + 0.112)` m with `lower = -22.0 deg`, `upper = 0.0 deg`, `target = +13.0 deg`,
  `stiffness = 0.055 N*m/deg`, `damping = 0.0045 N*m*s/deg`, and `initial = 0.0 deg`. Features front and rear entry
  decks flanking the `104 mm` central finger cutout (`y = -0.036..+0.042 m`), a full-width clamping platen and flush
  cork friction pad (`y = +0.042..+0.130 m`), a retarder stop lip at `y = +0.130 m`, and a left-side (`+y`) amber
  press-down paddle (`60 x 54 x 18 mm` at `y = +0.136..+0.190 m`).
- **Squared Half-Letter Newsletter Stack (`newsletter_packet`, free `Box` + `rx.Compound`, `216 x 140 x 20 mm`, `0.540 kg`):**
  Squared half-letter (statement-size) newsletter stack (`216 mm` width along `x` by `140 mm` length along
  `y` by `20.0 mm` thickness along `z`, no invented tabs, handles, or rims) resting flat on the right-hand entry staging
  apron at `(0.255, -0.1060, TOP + 0.124)` m (`166.0 mm` back from the retarder stop, leading edge at `y = -0.0360 m`).

### Real-World Setup and Physical Measurement

In a physical mailroom setup with a Martin Yale 1501X AutoFolder (`14.1 kg` / `31 lb`) on the station bench and a
banded stack of half-letter (statement-size) newsletters (`216 x 140 x 20 mm`, `0.540 kg`):

- **Packet leading-edge gap to retarder stop (`<= 5.0 mm`):** Measured with a `5.0 mm` **feeler gauge** (or machinist
  **ruler** / depth caliper) between the leading (`+y`) edge of the paper stack and the inner face of the retarder
  stop.
- **Packet squareness in guides (`yaw <= 5.0 deg`, tilt `<= 4.0 deg`):** Measured with a bevel **protractor** against
  the fixed side paper guides and a digital **inclinometer** / **protractor** comparing the stack top plane to the feed
  table deck.
- **Feed-table clamped angle (`-7.5` to `-3.5 deg`) and wheel nip contact (`<= 2.5 mm`):** Measured with a digital
  **protractor** / **inclinometer** on the feed table tray after release and a `2.5 mm` **feeler gauge** between the
  top sheet of the stack and the underside of the `58 mm` rubber feed wheel.
- **Gripper clearance (`>= 30 mm`) and 1.0 s rest hold:** Verified with a `30 mm` **ruler** / gauge block and timestamped
  trial **video**.
- **Latched faults over the trial:**
  - `folder_moved` (`> 15 mm` shift or `> 8 deg` tilt): Measured with a **ruler** against bench witness tape around the
    folder's rubber feet and an **inclinometer** on the housing frame.
  - `packet_dropped` (`> 50 mm` vertical drop while neither gripper touches the stack): Measured from calibrated height
    markers on the trial **video** and bench **ruler** when the stack descends more than `50 mm` while unheld by both
    grippers.
  - `table_pushed_past_stop` (`< -16.5 deg`): Measured against a `-16.5 deg` mechanical witness stop / **protractor**
    mark on the feed table frame over the trial **video**.

## Scoring (`SCORE_OVER = "end"`, `SUCCESS_HOLD_S = 1.0`)

- **`1.0` (Full success):** No latched fault has occurred, `autofolder_housing` is stable on the table (`<= 15 mm`
  shift, `<= 8 deg` tilt), `newsletter_packet` has its leading edge within `5.0 mm` of the retarder stop (`<= 6.0 mm`
  lateral offset, `<= 4.0 mm` deck gap, `yaw <= 5.0 deg`, and tilt relative to the feed table `<= 4.0 deg`),
  `feed_table` is released and clamping the stack against the overhead rubber feed wheel (`theta_table` in
  `[-7.5, -3.5] deg` and wheel nip gap `<= 2.5 mm`), both grippers are `>= 30 mm` clear of all task objects, and all
  parts are at rest for `1.0 s`.
- **`0.5` (Partial credit):** No latched fault has occurred, the folder is stable on the table, and either (a) the
  stack is seated at the retarder stop and clamped against the feed wheel at rest while a gripper is still within
  `30 mm`, or (b) the feed table has been depressed (`<= -12.0 deg`) and the stack slid into the guides within `15 mm`
  of the retarder stop (`yaw <= 5.0 deg`) before releasing the feed table.
- **`0.0` (Incomplete or faulted):** Any other state, or permanently `0.0` for the rest of the episode if
  `folder_moved`, `packet_dropped`, or `table_pushed_past_stop` latches.

## Files in This Folder

- `task.py`: Scene specification, compound geometries (`autofolder_housing`, `feed_table`, `newsletter_packet`),
  constant passive spring hinge (`FEED_TABLE_JOINT`), purely observational latched fault tracker (`advance`), scorer
  (`score`, `success`, `distance`), and open-loop bimanual `demo()`.
- `robocurve_ext.py`: Task-local compound collider and USD joint helper (`Piece`, `Compound`, `Revolute`, `Prismatic`,
  `Fixed`).
- `test_task.py`: Offline pytest suite verifying geometry, passive spring contact blocking and clamping angles, `1.0`
  goal, `0.5` partial credit, `0.0` shortcuts and latched faults, 20 Hz 6-DOF IK reachability, table/inter-arm/part
  clearances, top-camera visibility, and metadata/README guidelines.
- `task.json`: Task metadata and benchmark card fields.

## Third-party assets

none
