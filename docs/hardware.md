# Hardware

The reference rig: a camera and a projector on a rigid bar, on a tripod, with a
calibration board, coded targets, scale bars and surface preparation.

## Bill of materials

| Item | Specification | Price | Notes |
|---|---|---|---|
| Final camera | Raspberry Pi HQ camera, IMX477, 4056 × 3040, 1.55 µm pixels, C/CS mount, on the Pi capture node | $50 | Full-resolution static views at 10 fps, faster binned |
| Preview camera | Raspberry Pi Global Shutter camera, IMX296, 1456 × 1088, 3.45 µm pixels, C/CS mount, external trigger input, on the Pi capture node | $50 | The live path at up to 60 fps, triggered per pattern |
| Capture node | Raspberry Pi 5 with both CSI connectors used, HDMI to the projector, a GPIO line to the trigger input, gigabit Ethernet to the host | owned or $80 | Display timing, trigger and capture on one clock |
| USB alternative | A 1 MP OV9281-class USB global-shutter camera with a trigger input ($75 to $140) and a 12 MP USB module, on the host | optional | The same capture interface without a Pi |
| Lenses | 6 mm CS-mount, manual iris and focus, one per camera | $25 each | HQ: f ≈ 3,870 px, field 1.05 × 0.78 m at 1 m. GS: f ≈ 1,740 px, field 0.84 × 0.63 m at 1 m |
| Projector | Used DLP with a native 1920 × 1080 DMD, 2,000 ANSI lumens or more, manual focus reaching 1 m, HDMI, 120 Hz input a plus | $80 to $150 | Requirements below |
| Bar | 40 × 40 aluminum extrusion, 0.6 m, with T-nuts, corner brackets and two camera plates | $40 | Rigidity keeps the calibration valid |
| Tripod | A photo tripod rated for 3 kg or more, with a 3/8″ or 1/4″ stud to the bar | $30 to $60 | |
| Calibration board | ChArUco, A2, printed and mounted flat on foam board | $15 | The printed square size, measured with calipers, sets the scale |
| Targets | ArUco markers printed on adhesive paper, 20 to 40 | $5 | Three or more per view, left in place for the whole scan |
| Scale bars | Two bars with a target at each end, lengths measured with calipers | $10 | Different orientations in every scan |
| Spray | Matte scanning spray or dry developer | $15 to $25 | Gloss and black surfaces |
| Host | A Linux machine with a Vulkan-capable GPU (AMD or NVIDIA), HDMI out for the projector, USB 3 for the cameras | owned | Mesa lavapipe runs the same kernels on the CPU without a card |

About $300 to $450 from nothing with both cameras and a Pi 5. A tethered mirrorless
camera through gphoto2 replaces the first two lines when optics and dynamic range
matter more than cost.

## Projector requirements

- **DLP, native 1920 × 1080.** A digital micromirror device keeps a binary edge sharp
  where LCD and LCoS panels blur and polarize it. "1080p supported" on a listing means
  an input the unit downscales to a smaller DMD, so the specification sheet's native
  resolution is what counts. The preview path's phase pitch and the final path's
  0.5 mm lateral sampling assume the full 1920 columns.
- **Brightness.** 2,000 ANSI lumens or more for a 1 m field in a dim room with a
  short exposure. Brightness scales with field area: a 500 lumen LED pico unit covers
  a 0.3 m field for small parts and nothing larger.
- **Focus and throw.** Manual focus with a minimum projection distance at or under
  1 m, and a throw ratio (distance over image width) of 1.2 to 1.5 so a 1 m field sits
  at 1.2 to 1.5 m. Small parts at 0.3 to 0.5 m need a short-throw unit or one whose
  minimum focus reaches that close.
- **Refresh.** 60 Hz is the floor. A unit that accepts 1920 × 1080 at 120 Hz doubles
  the preview rate. Its EDID has to offer the native mode so the host drives it pixel
  for pixel.
- **Temporal processing off.** Keystone, auto keystone, image shift, frame
  interpolation, dynamic iris, eco or auto brightness, and any "brilliant color"
  mode are disabled, because each one changes the pattern or its brightness over time.
  A game or low-latency mode, where offered, shortens the input delay the sync has to
  absorb.
- **Color wheel.** A single-chip DLP shows red, green and blue through a wheel at a
  multiple of the frame rate, so an exposure shorter than a frame catches part of the
  wheel and brightness varies. The camera exposure is a whole number of
  projector frames (16.7 ms at 60 Hz), which integrates a full wheel cycle. Otherwise
  the patterns are projected in one color and the exposure is set by trial against a
  flat target until frames repeat within 1 %. LED and laser-phosphor units have no lamp
  flicker and are the upgrade for the preview path.
- **Gamma.** Every unit is non-linear. A ramp projected and captured gives the
  response, and the phase-shift patterns are pre-compensated with its inverse, else
  the decoded phase has a periodic error.
- **Sync output.** Consumer units have none, so the capture node times the trigger
  from its own vertical blank. A DLP development kit (LightCrafter 4500 at 912 × 1140,
  LightCrafter 6500 at 1920 × 1080, binary patterns above 2 kHz, trigger in and out) is
  the Phase 10 candidate for a preview at the camera's frame rate.
- **Used-market checks.** Lamp hours under half the rated life, no dead mirrors (a
  white field shows them as fixed black dots), the fan noise acceptable for an hour of
  scanning.

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
  (`ROADMAP.md` Phase 8 acceptance).

## Surface preparation

- Matte scanning spray or dry developer on glossy paint, black plastic and bare metal.
  Sublimating sprays leave nothing behind after a few hours.
- Painter's tape or dot stickers give plain panels texture for a photogrammetry pass,
  not needed for structured light, which brings its own texture in the patterns.
- Doors closed or night work against ambient light. The patterns need contrast, and
  sunlight through a window washes them out.

## Live path

- **Synchronization.** The projector's frame is the clock. The capture node flips
  each pattern on the display's vertical blank and raises the trigger line in the same
  handler. The Global Shutter camera exposes for a whole number of projector frames,
  and the decode matches frames to patterns by their timestamps. The HQ camera, rolling
  shutter, serves the final path, where each pattern is held until its frame is read
  out in full.
- **Preview dimensionality.** With the Global Shutter camera's 6 mm lens (f ≈ 1,740 px)
  on the 0.35 m baseline at 1 m, δz = z²·δp/(f·b) is 0.16 mm at δp = 0.1 px and 0.5 mm
  at 0.3 px. Lateral sampling is 0.58 mm. That is coarser than the final path by the
  ratio of the two sensors' pixel counts, and enough to show coverage and gaps.
- **Rate.** A 60 Hz projector and a four-step cycle give 15 depth frames per second,
  a three-step cycle 20. A unit that accepts 120 Hz at 1080p doubles both. A DLP
  development kit with binary patterns above 2 kHz lifts the ceiling to the camera's
  frame rate, and is a candidate, not a requirement.
- **Both cameras on the bar.** The preview and final cameras are calibrated to the
  projector on the same board in one session, so the preview's coverage and the final's
  views share the frame.
