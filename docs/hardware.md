# Hardware

The reference rig is the active stereo head of `DESIGN.md` D25: two Global Shutter cameras
on a stiff bar, a pattern light between them, and a Raspberry Pi 5 as the capture node.

## Bill of materials

| Item | Specification | Price | Notes |
|---|---|---|---|
| Cameras | Two Raspberry Pi Global Shutter cameras, IMX296, 1456 × 1088, 3.45 µm pixels, C/CS mount, trigger input | $50 each | Exposed together by one trigger pulse |
| Lenses, near range | 16 mm C-mount, 1" format, rated 10 MP, focus to 0.2 m: Raspberry Pi or Waveshare 16 mm telephoto, one per camera | check vendor | f = 4,638 px, field 157 × 117 mm at 0.5 m |
| Lenses, a metre | 6 mm CS-mount, one per camera | $25 each | f = 1,739 px, field 0.84 × 0.63 m at 1 m |
| Capture node | Raspberry Pi 5 with both CSI connectors used, one GPIO line to both trigger inputs, gigabit Ethernet to the host | owned or $80 | Trigger and capture on one clock |
| Pattern light | An LED behind a random-dot slide and a lens, or an LED projector showing one random-dot image | low cost | Requirements below; only the LED and slide version switches per frame |
| Light switch | Logic-level N-channel MOSFET on one Pi GPIO line, in the LED's supply | $2 | Drops the light for chosen frames (`DESIGN.md` D63) |
| Bar | Steel or carbon fibre, cameras 0.2 m apart for 0.5 m, 0.35 m apart for a metre, toed in | $40 | Aluminium shifts depth scale about 0.01 mm per kelvin at near range |
| Calibration board | ChArUco: A2 printed on foam board for a metre, 11 × 8 squares of 12 mm, 132 × 96 mm, on glass or ceramic for near range (D44) | $15 or vendor | The measured square size sets the scale |
| Markers, optional | Printed ArUco markers on a mat around the part, 20 to 40 | $5 | Bench work where the part's shape gives tracking little to hold, never on the part (D63) |
| Reference dots, optional | Removable paper or vinyl dots of 6 to 10 mm, one every 100 to 150 mm along the path | $5 | Long smooth panels only, peeled off afterwards (D64) |
| Polarizers, optional | Linear polarizing film over the light, one polarizing filter per lens, crossed | $20 to $40 | May replace matte spray on glossy surfaces, untried |
| Scale bars and artifacts | Two bars at a caliper-measured spacing, with spheres of 50 mm or more at their ends for a metre (D45); for near range a 25 mm sphere, a ball bar and a flat, each calibrated | $10 or vendor | Acceptance tests of `docs/accuracy.md` |
| Spray | Matte scanning spray | $15 to $25 | Only for glossy, black or transparent surfaces |
| Host | A Linux machine with a Vulkan-capable GPU (AMD or NVIDIA) and Ethernet | owned | Mesa lavapipe runs the same kernels on the CPU without a card |

## Trigger wiring

- **XTR input.** Each Global Shutter camera takes its trigger on XTR, a 1.8 V input, after
  R11 is removed from its board. One GPIO line drives both inputs through a divider (the
  Raspberry Pi documentation uses 1.5 kΩ in series and 1.8 kΩ to ground).
- **Exposure and rate.** The low pulse width sets the exposure, plus 14.26 µs, and the
  pulse rate sets the frame rate. Both cameras expose on the same edge.
- **Rework.** Removing R11 is SMD work. Third-party IMX296 modules with an isolated
  trigger input avoid it, and their pinouts are checked before purchase.

## Pattern light

- **Pattern.** One fixed random-dot image. Dots about 3 to 8 camera pixels across work:
  at 0.5 m with 16 mm lenses that is 0.3 to 0.9 mm, and at 1 m with 6 mm lenses 1.7 to
  4.6 mm.
- **Focus.** Roughly right is enough. On the virtual bench a slight defocus improved
  matching, because soft dots locate more precisely than hard edges.
- **Light source.** LED. A laser dot projector adds speckle that differs between the two
  cameras.
- **Switching.** The node turns the LED off for chosen frames through the MOSFET, between
  exposures, so those frames see the scene under room light (D63). A video projector
  cannot do this in step with the trigger, since a black frame over HDMI arrives on the
  projector's own clock.
- **Brightness.** Enough for exposures of a few milliseconds over the field, so a moving
  scanner does not blur. Ambient light lowers contrast without stopping the match.
- **Not needed.** Calibration, synchronization, refresh rate, gamma and trigger outputs.

## Rates

- **Frame pairs.** 1456 × 1088 mono is 1.58 MB per frame, 3.17 MB per pair. At 30 fps a
  pair stream is 95 MB/s, which fits gigabit Ethernet. At 60 fps it is 190 MB/s, which
  needs 2 × 2 binning or a 2.5 GbE adapter on the Pi 5.
- **Depth frames.** One per pair, so 30 depth frames per second over gigabit at full
  resolution.
- **Motion.** Each depth frame is one exposure, so motion costs only blur within it:
  10 mm/s for 2 ms moves 0.02 mm, under a pixel at either working distance.

## Rig geometry

- **Baseline and toe-in.** 0.2 m at 0.5 m and 0.35 m at 1 m, each camera toed in so the
  axes cross at the working distance. Depth resolution scales with 1/b, and occlusion in
  recesses grows with b.
- **Working distance.** Closer raises precision (δz ∝ z²) and shrinks the field.
- **Overlap.** Both cameras must see a point to measure it. The pattern light sits
  between them so it lights what both see.

## Lenses

- **Resolution.** Raspberry Pi rates its 6 mm CS lens at 3 MP for a 1/2" format and its
  16 mm C lens at 10 MP for 1". Matching windows span many pixels, so lens resolution
  limits fine detail more than depth.
- **Depth of field.** At f/8 a 16 mm lens at 0.5 m keeps about 110 mm within two
  3.45 µm pixels of blur (2·N·c·u²/f²). A 6 mm lens at 1 m keeps far more.
- **Locking.** Focus and iris are set at the working distance and locked before
  calibration. Any change means a new calibration.

## Build notes

- Mount both cameras on plates with two bolts each, never a single bolt, so nothing
  rotates. Mark the plates' positions.
- Run cables along the bar with strain relief at the bar, so a tugged cable moves the
  bar and not a camera.
- Carry the calibration board flat. A bowed board calibrates bowed cameras.
- Check the calibration after any knock, and before and after carrying the rig to a job
  (`ROADMAP.md` Phase 8 acceptance).

## Surface preparation

- Most surfaces scan as they are, because the pattern light supplies the texture.
- Glossy paint, black plastic, bare metal and glass return too little light to match.
  A matte scanning spray covers them, and sublimating sprays leave nothing behind after a
  few hours.
- Glossy surfaces may also scan through crossed polarizers on the light and both lenses,
  which suppress mirror reflections of the dots. This is untried.
- Markers are optional (D63). When used, they go on the mat or table around the part, not
  on the part.
- Long smooth panels give tracking nothing to hold along their surface. Removable dots on
  their featureless spans, one every 100 to 150 mm, fix that and peel off afterwards
  (D64). Shapes with relief need nothing.

## Phase-shift option

The calibrated projector path of D11 to D24 stays in the code, for edge-sharp static scans.
Its projector has to meet the measurement, which the pattern light above does not:

- **DLP with a native resolution** near the camera's, so its image matches the field.
  At near range that means an image about 150 mm wide in focus at 0.5 m (D23).
- **Temporal processing off:** keystone, frame interpolation, dynamic iris, eco and
  "brilliant color" modes change the pattern over time.
- **Color wheel and gamma:** exposures of a whole number of frames, and patterns
  pre-compensated by the measured response.
- **Trigger:** the capture node's vertical blank, or a development kit's trigger output.
  TI's LightCrafter 4500 has one and focuses from 0.5 m with an image at least 0.36 m
  wide, and distributors list it as discontinued.
