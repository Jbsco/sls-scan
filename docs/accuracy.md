# Accuracy

Resolution is the floor set by the geometry, and accuracy sits above it, set by the
rig's systematic errors and measured by the acceptance tests here.

## The error model

A camera pixel's ray and a projector column's plane meet at the point. Along the camera
axis, depth sensitivity to the decoded projector column follows the stereo relation

    δz = z² · δp / (f · b)

with z the working distance, δp the precision of the decoded correspondence in camera
pixels, f the camera focal length in pixels and b the baseline.

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
  0.5 mm after a multi-view scan (Phase 5).
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
twelve synthetic board poses. Results from `slscan simulate` on 2026-11-01:

| Test | Result | Truth or limit |
|---|---|---|
| Camera and projector reprojection | 0.096 and 0.092 px | under 0.3 and 0.5 px |
| Baseline | 349.85 mm | 350 mm |
| Flat plate, 0.5 m, tilted | 0.116 mm RMS | under 0.2 mm |
| Sphere radius, form | 50.010 mm, 0.035 mm RMS | 50.000 mm |
| Ball bar spacing | 303.281 mm | 303.315 mm |

A synthetic pass checks the arithmetic. It says nothing about lens, projector gamma or
rig flex, which only the bench measures.

## Small-volume head

These results are the small-volume head of `DESIGN.md` D23 on the virtual bench. The HQ
camera runs in its 2 × 2 binned mode (2028 × 1520, f = 8,065 px), with a calibration
recovered from twelve synthetic poses of a 90 × 60 mm board. Commands: `slscan simulate --head small`, then
the printed calibrate, scan and measure steps.

| Test | Projector image 150 mm | Projector image 360 mm | Truth or target |
|---|---|---|---|
| Flat plate, 80 mm, tilted | 7.6 µm RMS | 17.1 µm RMS | under 10 µm |
| Sphere radius | -3.4 µm | +0.2 µm | 12.5 mm sphere |
| Sphere form | 5.2 µm RMS | 12.4 µm RMS | |
| Ball bar spacing | -3.3 µm | -1.4 µm | 80.424 mm |

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
calibration is recovered from twelve synthetic poses of a 90 × 60 mm board. Commands: `slscan simulate --head stereo`, then
the printed calibrate-stereo, scan-stereo and measure steps.

| Test | Result | Truth or target |
|---|---|---|
| Calibration | 0.076 and 0.065 px, baseline 200.19 mm | under 0.3 px, 200 mm |
| Flat plate, 80 mm, tilted | 8 µm RMS | under 20 µm |
| Sphere radius | 12.395 mm | 12.500 mm |
| Ball bar spacing, 25 mm spheres | 80.404 mm | 80.424 mm |

Window size trades curvature bias against noise, with the true calibration:

| Window | Points inside a 25 mm sphere | Plate RMS |
|---|---|---|
| 21 px, 2.3 mm | 42 µm | 9 µm |
| 15 px, 1.6 mm | 22 µm | 17 µm |
| 11 px, 1.2 mm | 11 µm | 35 µm |
