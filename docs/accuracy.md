# Accuracy

Resolution is the floor set by the geometry, and accuracy sits above it, set by the
rig's systematic errors and measured by the acceptance tests here.

## The error model

A camera pixel's ray and a projector column's plane meet at the point. Along the camera
axis, depth sensitivity to the decoded projector column follows the stereo relation

    δz = z² · δp / (f · b)

with z the working distance, δp the precision of the decoded correspondence in camera
pixels, f the camera focal length in pixels and b the baseline. Active stereo follows
the same relation, with δp the disparity precision and b the distance between the two
cameras.

For the reference rig (`docs/hardware.md`), f ≈ 3,870 px and b = 0.35 m, at z = 1 m:

| Decode | δp | δz |
|---|---|---|
| Phase shift, clean surface | 0.1 px | 0.07 mm |
| Phase shift, realistic noise | 0.3 px | 0.2 mm |
| Gray code only | 1 px | 0.7 mm |

Lateral sampling at 1 m is 0.26 mm per camera pixel and about 0.5 mm per projector
pixel (a 1080p projector across a 1 m field). Phase shift pitch, 16 to 32 projector
pixels, sets the finest lateral code.

Halving z quarters δz and halves the field. Halving b doubles δz and halves occlusion
in recesses.

## Systematic errors

These sit above resolution and set accuracy:

- **Lens distortion residuals.** After calibration, 0.1 to 0.3 px across the field. They
  map into depth through the same relation, so a 0.3 px residual at the field edge is
  0.2 mm of depth.
- **Projector gamma.** A non-linear projector distorts the sinusoids and biases the
  phase. It is measured once by projecting a ramp and reading it back, and linearized in
  the pattern generator, or decoding uses a binary line shift instead.
- **Rig flex and drift.** A bar that moves 0.1 mm at the projector moves the baseline by
  that much, 0.03 %, and the scale with it. Projector thermal drift in its first minutes
  shifts the patterns.
- **Surface.** Gloss returns a specular highlight that saturates, and translucent
  surfaces scatter the pattern below the surface. Both are handled by the spray.
- **Registration.** Across many views, target-based poses set the global error. ICP
  alone accumulates drift with the view count.

Expected accuracy for the reference rig: 0.2 to 0.5 mm within one view over the 1 m
field, 0.5 to 1 mm over an object a few metres across with targets and scale bars.

## Acceptance tests

Each test is a scan of a known object with a number to beat, run after every
calibration and recorded with the scan.

- **Flat plate.** A granite tile, a glass sheet or a machined plate, 0.5 m across,
  sprayed. Its point cloud fits a plane with an RMS residual under 0.2 mm, and no bowl
  shape is visible in the residual map (a bowl means a distortion residual).
- **Scale bar.** Two bars with targets at caliper-measured spacing, placed in different
  orientations. Each measures within 0.2 mm in a single view (Phase 2) and within
  0.5 mm after a multi-view scan (Phase 6).
- **Sphere.** A bearing ball or a sprayed sphere of known diameter, scanned from four
  sides. The merged cloud fits its diameter within 0.3 mm, which tests registration as
  well as the views.
- **Datum points.** On a real job, three or more points measured on the object with
  calipers or a tape between features. After alignment, their residuals are within 1 mm
  over a few metres.

A scan's report contains the scale bar lengths, the plate RMS from the day's calibration
check and the datum residuals, so no result is quoted without its error.

## Virtual bench

The same tests run on synthetic captures (`DESIGN.md` D17). The virtual rig is a
1280 × 960 camera at f = 1500 px and a 1024 × 768 projector at 0.35 m baseline, with
sensor noise of one grey level and phase pitch 16. Its calibration is recovered from
twelve synthetic board poses, with k2 at 0 for both devices (D84). Results from
`slscan simulate` on 2026-11-04:

| Test | Result | Truth or limit |
|---|---|---|
| Camera and projector reprojection | 0.095 and 0.092 px | under 0.3 and 0.5 px |
| Baseline | 349.84 mm | 350 mm |
| Flat plate, 0.5 m, tilted | 0.092 mm RMS | under 0.2 mm |
| Sphere radius, form | 50.010 mm, 0.035 mm RMS | 50.000 mm |
| Ball bar spacing | 303.288 mm | 303.315 mm |

A synthetic pass checks the arithmetic. It says nothing about lens, projector gamma or
rig flex, which only the bench measures.

## Small-volume head

These results are the small-volume head of `DESIGN.md` D23 on the virtual bench. The HQ
camera runs in its 2 × 2 binned mode (2028 × 1520, f = 8,065 px), with a calibration
recovered from twelve synthetic poses of a 90 × 60 mm board. Commands: `slscan simulate --head small`, then
the printed calibrate, scan and measure steps.

| Test | Projector image 150 mm | Projector image 360 mm | Truth or target |
|---|---|---|---|
| Flat plate, 80 mm, tilted | 7.5 µm RMS | 17.1 µm RMS | under 10 µm |
| Sphere radius | -3.4 µm | 0.0 µm | 12.5 mm sphere |
| Sphere form | 5.2 µm RMS | 12.4 µm RMS | |
| Ball bar spacing | -2.8 µm | -1.0 µm | 80.424 mm |

Depth error of one preview cycle on a moving plate (`slscan motion-error --head small`):
Global Shutter camera, 60 Hz, four steps, projector image 150 mm.

| Sweep speed | Across the axis | Along the axis |
|---|---|---|
| Still | 7.1 µm | 7.1 µm |
| 1 mm/s | 7.3 µm | 9.0 µm |
| 2 mm/s | 7.7 µm | 13.3 µm |
| 5 mm/s | 10.3 µm | 28.9 µm |

These figures leave out lens and projector errors, thermal drift and target accuracy,
which only the bench measures.

## Active stereo

These results are the active stereo head of `DESIGN.md` D25 on the virtual bench. Two
Global Shutter cameras sit behind 16 mm lenses on a 0.2 m baseline crossing at 0.5 m.
An uncalibrated light throws a random-dot pattern on uncoated artifacts, and the
calibration is recovered from twelve synthetic poses of a 90 × 60 mm board.

This calibration reprojects to 0.076 and 0.065 px (limit 0.3 px) with a baseline of
200.17 mm against 200 mm, with k2 fixed for these narrow lenses (D80, D84). Commands:
`slscan simulate --head stereo`, then the printed calibrate-stereo, scan-stereo and
measure steps, with `--order 1 --window 21` for affine windows.

| Test | Affine, 21 px | Second-order, 25 px | Truth or target |
|---|---|---|---|
| Flat plate, 80 mm, tilted | 8 µm RMS | 10 µm RMS | under 20 µm |
| Sphere radius | 12.407 mm | 12.498 mm | 12.500 mm |
| Sphere form | 19 µm RMS | 8 µm RMS | 0 |
| Ball bar spacing, 25 mm spheres | 80.379 mm | 80.402 mm | 80.424 mm |

Ball bar spacing from one capture depends on where the rectified images sample the
dots. With the true calibration, shifting both rectified images by a fraction of a
pixel, which leaves the geometry unchanged, moved the affine result from -9 to -38 µm.
Cubic in place of linear interpolation for rectification moved it from -12 to -36 µm.

Window size and order trade curvature bias against noise. These runs use true
calibration and a sample every 8 px, and times are for the plate view in NumPy and
OpenCV on a Ryzen 7 7700X. Offsets are mean distances from the true sphere surface.

| Window | Order | Sphere offset | Fitted radius | Sphere form | Plate RMS | Time |
|---|---|---|---|---|---|---|
| 21 px, 2.3 mm | affine | -40.9 µm | -98.8 µm | 17.0 µm | 7.5 µm | 2.3 s |
| 15 px, 1.6 mm | affine | -21.5 µm | -54.8 µm | 22.3 µm | 18.2 µm | 2.5 s |
| 11 px, 1.2 mm | affine | -7.0 µm | -25.2 µm | 26.5 µm | 35.0 µm | 2.6 s |
| 25 px, 2.7 mm | second | +0.1 µm | +0.6 µm | 6.7 µm | 9.0 µm | 11.1 s |
| 31 px, 3.3 mm | second | +0.2 µm | +2.7 µm | 5.1 µm | 5.7 µm | 21.1 s |
| 41 px, 4.4 mm | second | +0.9 µm | +11.3 µm | 4.5 µm | 3.5 µm | 48.8 s |

Second-order fitted radii grow with the window while mean offsets stay within 1 µm.
Larger windows keep fewer samples near the sphere's rim, and it seems likely that the
smaller cap lets the fit trade radius against centre position.

## Metre head

These results are the metre head of `DESIGN.md` D45 on the virtual bench. It has 6 mm
lenses on a 0.35 m baseline crossing at 1 m, a calibration recovered from twelve poses of
the A2 board, and direct matching with 21 px windows. Command: `slscan stereo-bench`, with
`--seed` for other noise.

| Test | Result | Phase 2 limit |
|---|---|---|
| Calibration | 0.091 and 0.082 px, baseline 350.13 mm | under 0.3 px |
| Flat plate, 0.5 m, tilted | 0.041 mm RMS over 467 × 476 mm | 0.2 mm |
| Scale bar, two 50.8 mm spheres 400 mm apart | +0.022, +0.070 and +0.040 mm on three seeds | 0.2 mm |

## Continuous sweep

A synthetic sweep of 25 frames by the active stereo head, yawing ±25° and pitching ±8°
around a mat with a ball bar of two 25 mm spheres and a 16 mm sphere, tracked and fused
as in `DESIGN.md` D26. Command: `slscan sweep-bench`, with `--recovered` for the
recovered calibration and `--order 2` for second-order windows of 25 px.

| Measure | Affine, true | Affine, recovered | Second-order, true | Second-order, recovered |
|---|---|---|---|---|
| Pose drift against truth | 107 µm, 0.047° | 299 µm, 0.042° | 141 µm, 0.046° | 292 µm, 0.046° |
| Fused ball bar spacing | -3.9 µm | -10.7 µm | -2.9 µm | -4.3 µm |
| Fused mat flatness | 19.9 µm RMS | 19.2 µm RMS | 14.2 µm RMS | 15.8 µm RMS |
| Ball bar sphere radii | -116 and -76 µm | -101 and -100 µm | -33 and +8 µm | -11 and -8 µm |
| Run time | 57 s | 59 s | 233 s | 209 s |

Three runs with affine windows and recovered calibration gave a largest pose error of
0.0424° to 0.0425°, with artifact results within 0.6 µm of each other.

## Static views

Ten synthetic views of a part about a metre across, registered from geometry as in
`DESIGN.md` D70. Cameras stand 1.2 m out and 0.9 m up, 36° apart, and ray-cast points
carry 40 µm of noise per coordinate. The part has two scale bars of 50.8 mm spheres, one
level and 401 mm long, one rising and 388 mm long. Command: `slscan view-bench`, with
`--voxel-mm` and `--seed`.

| Coarse voxel | Pose error, largest over three seeds | Scale bar errors, largest | Phase 6 limit | Time |
|---|---|---|---|---|
| 10 mm | 0.078 mm, 0.0028° | +0.003 and +0.005 mm | 0.5 mm | 30 s |
| 20 mm | 0.241 mm, 0.0083° | +0.009 and +0.010 mm | 0.5 mm | 12 s |

Ray-cast points have no matching errors and no holes, so these figures measure
registration alone.

With `--stereo`, each view is a pattern-lit pair for the metre head, matched directly at
an 8 px step from 0.8 m out and 1.0 m up (D71). Pose errors with a recovered calibration
seem likely to include an offset between recovered and true camera frames.

| Calibration | Pose error, largest | Scale bar errors | Phase 6 limit | Time |
|---|---|---|---|---|
| True | 0.204 mm, 0.0109° | +0.078 and -0.091 mm | 0.5 mm | 262 s |
| Recovered (`--recovered`) | 1.665 mm, 0.0946° | -0.126 and +0.008 mm | 0.5 mm | 250 s |

With `--datums`, the four scale bar spheres are found in the merged views without seeds
and aligned to their true centres (D81). Their residuals were 2 to 5 µm with ray-cast
views and 0.095 to 0.198 mm with stereo views and a recovered calibration.
