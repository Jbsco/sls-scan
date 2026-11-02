# Accuracy

Geometry sets resolution. Systematic errors of the rig set accuracy above it, and the
acceptance tests here measure it.

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
field, 0.5 to 1 mm across a cab-sized object with targets and scale bars.

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
  across a cab.

A scan's report contains the scale bar lengths, the plate RMS from the day's calibration
check and the datum residuals, so no result is quoted without its error.
