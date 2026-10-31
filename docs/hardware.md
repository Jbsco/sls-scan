# Hardware

The reference rig: a camera and a projector on a rigid bar, on a tripod, with a
calibration board, coded targets, scale bars and surface preparation.

## Bill of materials

| Item | Specification | Price | Notes |
|---|---|---|---|
| Final camera | 12 MP IMX477 class: Raspberry Pi HQ camera (on a Pi capture node) or a USB module, C/CS mount | $50 to $100 | Full-resolution static views |
| Preview camera | 1 MP monochrome global shutter, OV9281 class, USB, with a hardware trigger input, 100 to 120 fps | $75 to $140 | The live path, synchronized to the projector |
| Lens | 6 mm CS-mount, manual iris and focus | $25 | f ≈ 6 mm / 1.55 µm ≈ 3,870 px, field 1.05 × 0.78 m at 1 m |
| Projector | Used 1080p DLP, 2,000 lumens or more, manual focus, HDMI | $80 to $150 | DLP, not LCD. Keystone off. Lamp hours checked |
| Bar | 40 × 40 aluminum extrusion, 0.6 m, with T-nuts, corner brackets and two camera plates | $40 | Rigidity holds the calibration |
| Tripod | A photo tripod rated for 3 kg or more, with a 3/8″ or 1/4″ stud to the bar | $30 to $60 | |
| Calibration board | ChArUco, A2, printed and mounted flat on foam board | $15 | The printed square size, measured with calipers, sets the scale |
| Targets | ArUco markers printed on adhesive paper, 20 to 40 | $5 | Three or more per view, left in place for the whole scan |
| Scale bars | Two bars with a target at each end, lengths measured with calipers | $10 | Different orientations in every scan |
| Spray | Matte scanning spray or dry developer | $15 to $25 | Gloss and black surfaces |
| Host | A Linux machine with a Vulkan-capable GPU (AMD or NVIDIA), HDMI out for the projector, USB 3 for the cameras | owned | Mesa lavapipe runs the same kernels on the CPU without a card |

About $350 to $500 from nothing with both cameras. A tethered mirrorless camera through gphoto2 replaces
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

## Live path

- **Synchronization.** The projector's frame is the clock. The preview camera is
  triggered per frame, from the projector's sync output where the unit offers one, or
  from the host's own frame timing with the exposure set shorter than the frame, and the
  decode matches frames to patterns by their timestamps.
- **Rate.** A 60 Hz projector and a four-step cycle give 15 depth frames per second,
  a three-step cycle 20. A unit that accepts 120 Hz at 1080p doubles both. A DLP
  development kit with binary patterns above 2 kHz lifts the ceiling to the camera's
  frame rate, and is a candidate, not a requirement.
- **Both cameras on the bar.** The preview and final cameras are calibrated to the
  projector on the same board in one session, so the preview's coverage and the final's
  views share the frame.
