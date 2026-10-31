# Design decision ledger

Decisions are dated and numbered. A reversal gets a new entry, never an edit.

## D1. The GPL-3.0-or-later license (2026-10-31)

A scanner built from commodity parts and open libraries should stay open with its
improvements. OpenCV (Apache-2.0) and Open3D (MIT) are compatible dependencies.

## D2. A static rig, not a handheld (2026-10-31)

The handheld scanners in the $250 to $1,500 range capture only through their makers'
Windows and macOS applications, and their per-frame accuracy claims hold with markers
on the part. A camera and projector fixed to a bar on a tripod gives the same
measurement principle with a calibration the operator owns, a pipeline that runs on
Linux end to end, and per-view accuracy that matches the handhelds on large parts. The
cost is time: a view takes 10 to 15 s of capture and a minute or two of repositioning,
so a cab is one to two hours of capture against twenty minutes with a handheld. For
reference geometry taken once, the time is acceptable.

## D3. The camera (2026-10-31)

The Raspberry Pi HQ camera (IMX477, 4056 × 3040, 1.55 µm pixels) with a 6 mm CS lens
gives f ≈ 3,870 px and a 1.05 × 0.78 m field at 1 m for $75, and it is scriptable from a
Raspberry Pi or over USB. Its rolling shutter does not matter, because nothing moves
during a pattern sequence. A tethered mirrorless camera through gphoto2 is the upgrade
path for optics and dynamic range. Analog and digital FPV cameras were considered and
set aside: fisheye distortion, low resolution and compression cost more accuracy than
their availability saves.

## D4. The projector (2026-10-31)

A used 1080p DLP projector. DLP mirrors keep binary edges sharp where LCD panels blur
them, 1080p samples a 1 m field at about 0.5 mm per projector pixel, and used office
units cost $80 to $150. Keystone correction stays off, because it resamples the
patterns. The projector's depth of field, about ±0.25 m around focus, sets the usable
depth band per view, and each view is set up with the surface inside it.

## D5. Gray code plus phase shift (2026-10-31)

Gray code alone resolves the projector column per camera pixel without ambiguity, at
one projector pixel of precision. Phase shift alone resolves to a tenth of a pixel but
only within one period. Together, the Gray code picks the period and the phase shift
refines inside it, which is the standard combination and what OpenCV's
`structured_light` module implements. Inverse patterns make the binarization robust
to surface reflectance, and two or three exposures per pattern cover dark and light
areas in one view.

## D6. Python with OpenCV and Open3D (2026-10-31)

A scanner's value is in its calibration and its measurement, both checked by physical
acceptance tests (`docs/accuracy.md`), and its work is image and point cloud
processing, where OpenCV and Open3D are the established tools. Python with those two,
and NumPy for everything between, reaches a working scanner fastest. The pure functions
(pattern codes, decoding arithmetic, triangulation) are kept free of camera and display
dependencies and tested in isolation, so a later port of that core to a compiled
language has a specification and vectors to port against.

## D7. Registration by coded targets, checked by scale bars (2026-10-31)

Global accuracy over a large object is set by how the views are joined, not by the
sensor. Printed coded targets seen in three or more views give the view poses directly,
and ICP refines them on the overlapping geometry. ICP alone drifts with the number of
views, which is why it is a refinement and not the registration. Two scale bars of
caliper-measured length, placed in different orientations, give an independent check of
every scan's scale and are reported with every result.

## D8. The error model (2026-10-31)

Depth resolution follows the stereo relation δz = z²·δp/(f·b), with z the working
distance, δp the decoding precision in camera pixels, f the focal length in pixels and
b the baseline. `docs/accuracy.md` carries the numbers for the reference rig. Resolution
is the floor; accuracy is set above it by calibration residuals, projector gamma and
rig flex, and is measured, never assumed, by the acceptance tests.

## D9. Data outside version control (2026-10-31)

Captures, calibrations and scan outputs are large and machine-specific and live under
`data/`, which is ignored. Calibration results worth keeping are exported as small YAML
files that a rig directory can track by choice.

## D10. A self-hosted runner while the repository is private (2026-10-31)

GitHub provides hosted runners to public repositories; a private one supplies its own,
so the workflows run on a self-hosted runner, with `runs-on: [self-hosted, linux]`. The steps that only the hosted Ubuntu image needs
(the apt install of the markdown renderer, the Python setup and the package install)
are gated on `runner.environment == 'github-hosted'` rather than deleted, so a return
to hosted runners is a one-line change per job. The runner's host supplies the
renderer, Python, NumPy and pytest. Jobs run with that host account's access, which for
a private repository of one's own equals running the gates locally. At a public flip the
workflows go back to `ubuntu-latest` and the runner is removed, because a public
repository's fork pull requests would otherwise execute on the host.

## D11. Live scanning, the GPU and the languages (2026-10-31)

The static pipeline of D5 is about 30 projector frames per view, half a second at
60 Hz with the scene still, so a live scan is a different path and not a faster
version of this one. The target is a preview at 10 to 30 depth frames per second,
with the final object captured at full resolution and processed afterwards.

- **Preview path.** Three- or four-step phase shift at one pitch, unwrapped against
  the previous frame and the model instead of a Gray code sequence, from a global-shutter
  camera at 1 MP synchronized to the projector. At 60 Hz that is 15 to 20 depth frames
  per second, at 120 Hz 30 to 40. A one-shot pattern matched by stereo is the fallback
  for coverage-only preview at camera rate. Frames are fused into a coarse TSDF volume
  for display, with poses from the coded targets and ICP.
- **Final path.** The full Gray code and phase shift sequence per view at the full
  camera resolution, processed after capture on the GPU (seconds per view) or the CPU
  (tens of seconds), then registered and meshed as in D7.
- **GPU and fallback.** The kernels run through wgpu, the WebGPU API over Vulkan, which
  serves both an AMD and an NVIDIA card on Linux through their Vulkan drivers, in WGSL
  compute shaders that share buffers with the 3D view. The CPU fallback is Mesa's
  software Vulkan implementation (lavapipe) running the same shaders unchanged, slower,
  plus the reference implementation for the pure core, so no kernel exists twice.
  Vendor-only paths (CUDA, ROCm, Accelerate's PTX backend) are out, because the two
  machines carry different vendors.
- **Languages.** The application (capture, synchronization, kernels, fusion, the panels
  and the 3D scene) is Rust with wgpu and egui, chosen over C for memory safety at the
  same speed and over Python for the live loop's latency and the single-binary
  deployment. Python stays for the offline tooling where OpenCV has no equal: the
  calibration solve, target detection development, and mesh post-processing with
  Open3D, exchanging YAML and PLY files with the application. Haskell is the candidate
  for the executable reference of the pure core (pattern codes, decoding arithmetic,
  triangulation): it generates the vectors the kernels are checked against, as a second
  implementation in a different language catches what a port of the same code repeats.
  Haskell is not the kernel language, since Accelerate targets only NVIDIA through
  PTX, and Futhark, which does target AMD and NVIDIA and a multicore CPU from one
  source, is kept as the alternative kernel language if WGSL proves limiting.
- **Modes and markers.** Static (view, reorient, view, registered on targets),
  continuous (the rig moves slowly while the preview path runs and fuses), and keyed
  (the operator names the target IDs that define the object's frame, so a scan aligns
  to datums the way the job needs). The panels are configuration (devices, calibration,
  pattern set, exposure), preview and processing (live view, coverage, capture, the
  deferred queue), and the scene (the final mesh with its datums, scale bars and
  residual report, and measurement tools).

## D12. Calibration through the frontend (2026-10-31)

The calibration solve stays in Python (D11), and the operator never runs it by hand.
The configuration panel captures the board poses, shows the detection count and
coverage as they accumulate, runs the solve as a subprocess of the bundled tool,
reports the reprojection errors against the Phase 1 limits, and loads the resulting
YAML into the active rig. A failed or poor solve is reported with the poses that hurt
it, so the operator recaptures those rather than everything.
