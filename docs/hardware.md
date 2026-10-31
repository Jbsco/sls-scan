# Hardware

The reference rig: a camera and a projector on a rigid bar, on a tripod, with a
calibration board, coded targets, scale bars and surface preparation.

## Bill of materials

| Item | Specification | Price | Notes |
|---|---|---|---|
| Camera | Raspberry Pi HQ camera, IMX477, 4056 × 3040, 1.55 µm pixels, C/CS mount | $50 | 12 MP of static capture, scriptable from a Raspberry Pi or over USB |
| Lens | 6 mm CS-mount, manual iris and focus | $25 | f ≈ 6 mm / 1.55 µm ≈ 3,870 px, field 1.05 × 0.78 m at 1 m |
| Projector | Used 1080p DLP, 2,000 lumens or more, manual focus, HDMI | $80 to $150 | DLP, not LCD. Keystone off. Lamp hours checked |
| Bar | 40 × 40 aluminum extrusion, 0.6 m, with T-nuts, corner brackets and two camera plates | $40 | Rigidity holds the calibration |
| Tripod | A photo tripod rated for 3 kg or more, with a 3/8″ or 1/4″ stud to the bar | $30 to $60 | |
| Calibration board | ChArUco, A2, printed and mounted flat on foam board | $15 | The printed square size, measured with calipers, sets the scale |
| Targets | ArUco markers printed on adhesive paper, 20 to 40 | $5 | Three or more per view, left in place for the whole scan |
| Scale bars | Two bars with a target at each end, lengths measured with calipers | $10 | Different orientations in every scan |
| Spray | Matte scanning spray or dry developer | $15 to $25 | Gloss and black surfaces |
| Controller | A Raspberry Pi 4 or 5, or any Linux machine with HDMI out and a USB camera | owned | The projector is the second display |

About $250 to $350 from nothing. A tethered mirrorless camera through gphoto2 replaces
the first two lines when optics and dynamic range matter more than cost.

## Rig geometry

- **Baseline.** 0.35 m between the camera and projector optical centres, toed in so the
  two axes cross at 1 m. Depth resolution scales with 1/b, and occlusion in recesses
  scales with b, so 0.2 m is the alternative for deep features at lower precision.
- **Working distance.** 1 m to the surface for the reference figures. Closer raises
  precision (δz ∝ z²) and shrinks the field.
- **Depth band.** The projector's depth of field is about ±0.25 m around its focus, and
  the camera's is set by the iris. Each view is set up with the surface inside both.
- **Field.** 1.05 × 0.78 m for the camera at 1 m. A 1080p projector at a 1.3 to 1.5 throw
  ratio covers a 1 m width from 1.3 to 1.5 m, so the projector may sit further back than
  the camera.

## Build notes

- Mount both units to the bar on plates with two bolts each, never a single bolt, so
  nothing rotates. Mark the plates' positions.
- Run the HDMI and camera cables along the bar with strain relief at the bar, so a
  tugged cable moves the bar and not a unit.
- Set the lens focus and iris once at the working distance, then lock them with tape or
  the lens's set screws. Any change means a new calibration.
- Let the projector warm up for ten minutes before calibrating or scanning. Lamp output
  and the DLP's temperature drift in the first minutes.
- Carry the calibration board flat. A bowed board calibrates a bowed camera.
- Check the calibration after any knock, and before and after carrying the rig to a job
  (`ROADMAP.md` Phase 5 acceptance).

## Surface preparation

- Matte scanning spray or dry developer on glossy paint, black plastic and bare metal.
  Sublimating sprays leave nothing behind after a few hours.
- Painter's tape or dot stickers give plain panels texture for a photogrammetry pass,
  not needed for structured light, which carries its own texture in the patterns.
- Doors closed or night work against ambient light. The patterns need contrast, and
  sunlight through a windshield washes them out.
