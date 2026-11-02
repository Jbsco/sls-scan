# Design decision ledger

Decisions are dated and numbered. A reversal gets a new entry, never an edit.

## D1. The GPL-3.0-or-later license (2026-10-31)

We chose the copyleft so that a scanner built on this one stays open with its
improvements. OpenCV (Apache-2.0) and Open3D (MIT) are compatible dependencies.

## D2. A static rig, not a handheld (2026-10-31)

Handheld scanners in the $250 to $1,500 range capture only through their makers'
Windows and macOS applications, and their per-frame accuracy claims hold with markers
on the part. A camera and projector fixed to a bar on a tripod gives the same
measurement principle. The operator owns the calibration, the pipeline runs on Linux
end to end, and per-view accuracy matches the handhelds on large parts.

The cost is time: a view takes 10 to 15 s of capture and a minute or two of
repositioning, so an object a few metres across is one to two hours of capture against
twenty minutes with a handheld. For reference geometry taken once, that time is acceptable.

## D3. The camera (2026-10-31)

The Raspberry Pi HQ camera (IMX477, 4056 × 3040, 1.55 µm pixels) with a 6 mm CS lens
gives f ≈ 3,870 px and a 1.05 × 0.78 m field at 1 m for $75. It is scriptable from a
Raspberry Pi or over USB. Its rolling shutter does not matter, because nothing moves
during a pattern sequence. A tethered mirrorless camera through gphoto2 is the upgrade
path for optics and dynamic range. Analog and digital FPV cameras were considered and
set aside: fisheye distortion, low resolution and compression cost more accuracy than
their availability saves.

## D4. The projector (2026-10-31)

A used 1080p DLP projector. DLP mirrors keep binary edges sharp where LCD panels blur
them, 1080p samples a 1 m field at about 0.5 mm per projector pixel, and used office
units cost $80 to $150. Keystone correction stays off, because it resamples the
patterns. Projector depth of field, about ±0.25 m around focus, sets the usable depth
band per view, and each view is set up with the surface inside it.

## D5. Gray code plus phase shift (2026-10-31)

Gray code alone resolves the projector column per camera pixel without ambiguity, at
one projector pixel of precision. Phase shift alone resolves to a tenth of a pixel but
only within one period. Together, Gray code picks the period and phase shift refines
inside it, which is the standard combination and what OpenCV's `structured_light`
module implements. Inverse patterns make the binarization robust to surface
reflectance, and two or three exposures per pattern cover dark and light areas in one
view.

## D6. Python with OpenCV and Open3D (2026-10-31)

A scanner's value is in its calibration and its measurement, both checked by physical
acceptance tests (`docs/accuracy.md`), and its work is image and point cloud
processing, where OpenCV and Open3D are the established tools. Python with those two,
and NumPy for everything between, reaches a working scanner fastest. Pure functions
(pattern codes, decoding arithmetic, triangulation) are kept free of camera and display
dependencies and tested in isolation, so a later port of that core to a compiled
language has a specification and vectors to port against.

## D7. Registration by coded targets, checked by scale bars (2026-10-31)

Global accuracy over a large object is set by how the views are joined, not by the
sensor. Printed coded targets seen in three or more views give view poses directly,
and ICP refines them on overlapping geometry. ICP alone drifts with the number of
views, which is why it is a refinement and not the registration. Two scale bars of
caliper-measured length, placed in different orientations, give an independent check of
every scan's scale and are reported with every result.

## D8. The error model (2026-10-31)

Depth resolution follows the stereo relation δz = z²·δp/(f·b), with z the working
distance, δp the decoding precision in camera pixels, f the focal length in pixels and
b the baseline. `docs/accuracy.md` contains the numbers for the reference rig.
Resolution is the floor. Accuracy sits above it, set by calibration residuals,
projector gamma and rig flex, and the acceptance tests measure it rather than assume
it.

## D9. Data outside version control (2026-10-31)

Captures, calibrations and scan outputs are large and machine-specific and live under
`data/`, which is ignored. Calibration results worth keeping are exported as small YAML
files that a rig directory can track by choice.

## D10. A self-hosted runner while the repository is private (2026-10-31)

GitHub provides hosted runners to public repositories. A private one supplies its own,
so the workflows run on a self-hosted runner, with `runs-on: [self-hosted, linux]`.
Steps that only the hosted Ubuntu image needs (the apt install of the markdown
renderer, the Python setup and the package install) are gated on
`runner.environment == 'github-hosted'` rather than deleted. A return to hosted runners
is then a one-line change per job. The runner's host supplies the renderer, Python,
NumPy and pytest.

Jobs run with that host account's access, which for a private repository of one's own
equals running the gates locally. At a public flip the workflows go back to
`ubuntu-latest` and the runner is removed, because a public repository's fork pull
requests would otherwise execute on the host.

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
  for display, with poses from coded targets and ICP.
- **Final path.** The full Gray code and phase shift sequence per view at full camera
  resolution, processed after capture on the GPU (seconds per view) or the CPU (tens of
  seconds), then registered and meshed as in D7.
- **GPU and fallback.** Kernels run through wgpu, the WebGPU API over Vulkan, which
  serves both an AMD and an NVIDIA card on Linux through their Vulkan drivers, in WGSL
  compute shaders that share buffers with the 3D view. The CPU fallback is Mesa's
  software Vulkan implementation (lavapipe) running the same shaders unchanged, slower,
  plus the reference implementation for the pure core, so no kernel exists twice.
  Vendor-only paths (CUDA, ROCm, Accelerate's PTX backend) are out, because the two
  machines have cards from different vendors.
- **Languages.** The application (capture, synchronization, kernels, fusion, panels and
  the 3D scene) is Rust with wgpu and egui, chosen over C for memory safety at the same
  speed and over Python for live-loop latency and single-binary deployment. Python
  stays for offline tooling where OpenCV has no equal: calibration solve, target
  detection development, and mesh post-processing with Open3D, exchanging YAML and PLY
  files with the application.
- **Reference core.** Haskell is the candidate for the executable reference of the pure
  core (pattern codes, decoding arithmetic, triangulation). It generates the vectors the
  kernels are checked against. A second implementation in a different language catches
  what a port of the same code repeats. Haskell is not the kernel language,
  since Accelerate targets only NVIDIA through PTX. Futhark, which targets AMD, NVIDIA
  and a multicore CPU from one source, is kept as the alternative kernel language if
  WGSL proves limiting.
- **Modes and markers.** Static: view, reorient, view, registered on targets.
  Continuous: the rig moves slowly while the preview path runs and fuses. Keyed: the
  operator names the target IDs that define the object's frame, so a scan aligns to
  datums the way the job needs. The panels are configuration (devices, calibration,
  pattern set, exposure), preview and processing (live view, coverage, capture, the
  deferred queue), and the scene (the final mesh with its datums, scale bars and
  residual report, and measurement tools).

## D12. Calibration through the frontend (2026-10-31)

The calibration solve stays in Python (D11), and the operator never runs it by hand.
The configuration panel captures board poses and shows detection count and coverage
as they accumulate. It runs the solve as a subprocess of the bundled tool, reports
reprojection errors against the Phase 1 limits, and loads the resulting YAML into the
active rig. A failed or poor solve is reported with the poses that hurt it, so the
operator recaptures those rather than everything.

## D13. The capture node (2026-10-31)

The Raspberry Pi cameras are CSI devices and attach only to a Pi, so a Pi 5 is the
capture node. Both cameras sit on its two CSI connectors, the HQ camera for the final
path and the Global Shutter camera for the preview path, and its HDMI output drives
the projector. Patterns flip on the display's vertical blank, and a GPIO line from the
page-flip handler triggers the Global Shutter camera's external trigger input, so
pattern timing and exposure share one clock.

Frames stream to the host application over gigabit Ethernet. The preview at 1456 × 1088
mono and 30 fps is about 48 MB/s, binned 2 × 2 a quarter of that. A final view of 30
full-resolution frames moves in a few seconds.

A USB global-shutter camera and a USB 12 MP module on the host remain the alternative
where no Pi is wanted, behind the same capture interface, with the host's own frame
timing as the trigger source. The node does the real-time work (display timing,
trigger, capture) on a machine made for it. GPU work and the panels stay on the host.

## D14. A public repository with pull requests only (2026-11-01)

The repository is public from 2026-11-01. Its sanitization gate passes on every commit
(no local paths, no downstream project or sibling names), its license is in place, and
its README says that nothing has run on hardware yet. A public repository gets hosted
runners, so the workflows run on `ubuntu-latest` and the self-hosted runner of D10 is
removed, which keeps fork pull requests off the host.

`main` changes only through pull requests, each merged as one signed merge commit with
the Style and Tests checks green on its final commit. Branches start from `origin/main`
and rebase onto it before the merge. A ruleset on `main` requires signatures, the pull
request path and green checks, and refuses deletion and force pushes.
`.github/CODEOWNERS` routes review to the owner.

## D15. Synthetic views verify the calibration (2026-11-01)

The calibration tooling exists before the rig, so we test it against a virtual camera
and projector with known parameters. `slscan/synthetic.py` renders a board print in a
known pose through a pinhole camera. It renders each projector pattern onto the same
plane as the camera sees it, with print albedo, ambient light and sensor noise. The
tests require focal lengths, principal points, rotation and baseline recovered from
these views within stated tolerances.

A synthetic pass checks the arithmetic only. Phase 1 acceptance still needs the real
board, lens and projector on the bench. The same renderer will produce decode and
triangulation vectors for Phase 2.

Projector corners follow the local homography method of Moreno and Taubin ("Simple,
Accurate, and Robust Projector-Camera Calibration", 3DIMPVT 2012). Decoded columns and
rows in a window around each camera corner fit a homography, which maps the corner
into the projector at a fraction of a pixel. A RANSAC fit drops pixels decoded wrongly
at stripe edges.

## D16. Decode and triangulation in NumPy (2026-11-01)

We decode with our own NumPy functions instead of OpenCV's `structured_light` module.
The GPU kernels of Phase 4 are checked against this code, so it has to be a readable
reference with vectors, and OpenCV's Python binding decodes one pixel per call.

Gray code bits pick the period and the wrapped phase fixes the position inside it. The
result is the phase column nearest the Gray code column, so a Gray code error under
half a period does not move it. A pixel is dropped when a pair lacks contrast, the
sinusoid lacks modulation, or the two columns disagree by more than a pixel.

Each camera ray meets the plane of its projector column. Projector distortion bends
those planes, so the plane comes from the undistorted projector point at the row
where the previous estimate lands, refined over three iterations. Calibration stays
in OpenCV.

## D17. A virtual bench before the hardware (2026-11-01)

The renderer of D15 ray-casts plates and spheres from the camera, tests each point for
a shadow from the projector, and renders Gray codes as sharp stripes and sinusoids as
a slightly defocused projector shows them. `slscan simulate` writes a calibration
capture and three scenes in the file layout of real captures.

The scenes are the artifacts of VDI/VDE 2634 Part 2 for optical 3D measuring systems:
a flat plate for flatness, a sphere for probing error, and a ball bar for sphere
spacing. The tests run the whole pipeline on them with a calibration recovered from
synthetic board views and check each result against its acceptance limit.

Renode emulates microcontrollers, and nothing in this design runs on one, so it is
not used. The simulation that Phase 3 needs is a capture node stand-in: a process,
run in a container, that serves synthetic frames over the node's protocol, so the
application is developed against the same interface as the Pi.

## D18. The capture protocol (2026-11-01)

The application and the capture node share one TCP connection with JSON control
messages and binary frames, specified in `docs/capture-protocol.md`. JSON keeps the
control messages readable and lets fields be added. Frames stay binary, because a
preview frame at 1456 × 1088 mono is 1.6 MB and the stream runs near 48 MB/s (D13).

The node generates the patterns from the set name and its parameters, so no pattern
image crosses the network. Each frame has its pattern index and the node's timestamp
of exposure start. The application matches frames to patterns by index and checks
the timing by timestamp. A change to the format changes the magic.

## D19. The stand-in node in a container, checked in CI (2026-11-01)

The stand-in node runs from an image built by the root `Dockerfile`, with only
packages it needs, at versions CI installs. A container gives it an address of its
own, so the application connects to it as it will to the Pi. `node.do` runs a virtual
bench through that container over the capture protocol, and a Node workflow runs it on
every push and pull request. `verify.do` leaves it out, because it needs Docker and
adds an image build to every local run.

Amended 2026-11-01: the ruleset on `main` requires the node check, with style and test.

## D20. The application shell (2026-11-01)

The application is one Rust crate in `app/`. Its library has the capture protocol, a
capture client and three panels, and its binary opens the window. It uses eframe with
default features on wgpu, the configuration its authors test. Rust 1.93.0 is pinned in
`app/rust-toolchain.toml`, so formatting and lints agree between a workstation and CI.

A headless `capture` mode runs one sequence and prints frame count, missing frames,
order and mean interval. `app.do` runs it against the Python stand-in, so each
implementation of the protocol is checked against the other on every run. A client
waits for HELLO, not for an open port, because a container's port forward accepts
connections before the node inside listens.

## D21. Calibration from the panel (2026-11-01)

Capture pose runs a calibration sequence over the protocol and writes its frames as PNG
files into the next `pose_NNN` directory of a session, in the layout `slscan calibrate`
reads. The application runs the Python tool as a subprocess: `slscan detect` after each
pose and `slscan calibrate` for the solve, both with `--json`, so it parses one JSON
object and no text meant for a person. Until the tool ships with the application, its
command is a setting, `python3 -m slscan.cli` by default.

Dependencies build with optimization in development builds. Writing a session of 492
frames at 1280 × 960 took 5 min with unoptimized PNG and deflate code, and 25 s with
optimized dependencies and fast PNG compression.

## D22. Decode kernel on the GPU (2026-11-01)

The decode of D16 runs as one WGSL compute invocation per camera pixel through wgpu
(`app/src/decode.wgsl`), in single precision. `slscan decode-vectors` writes a
synthetic scan with its NumPy decode, and `sls-scan decode-check` compares the kernel's
output with it. On each virtual bench scene at 1280 × 960, masks agree on every pixel
and columns differ by at most 0.00006 px, decoded in 70 ms on a Radeon RX 7900 GRE
with upload and readback. Vectors are written at gate time, so no large files are
tracked.

The CPU fallback of D11 is Mesa's lavapipe running the same kernel. Without an
adapter, kernel checks skip unless `SLS_REQUIRE_GPU` is set, which makes a missing
adapter a failure.

## D23. The small-volume head (2026-11-01)

A second head measures objects up to about 120 mm across, with 0.02 mm as its target.
The HQ camera stays, behind a 25 mm lens at 0.5 m: f = 16,129 px at full resolution and
a 126 × 95 mm field. The Global Shutter camera previews behind a 16 mm lens over a
157 mm field. Both cameras keep their C and CS mounts, so the 1 m rig and this head
share camera bodies and the capture node, and differ in lenses, projector and bar.

Here the projector sets the noise floor. Depth error also reads z²·δu/(f_p·b), with δu
the decode error in projector pixels and f_p the projector's focal length in pixels, so
a projector image much wider than the field spends its pixels outside it. On the virtual
bench, a projector image matched to the field, 150 mm at 0.5 m, gives a plate flatness
of 7.6 µm RMS. A 360 mm image, the smallest a LightCrafter 4500 focuses at 0.5 m, gives
17.1 µm. This head therefore needs DLP optics focused at 0.5 m with an image near
150 mm wide.

With a calibration recovered from synthetic views of a 90 × 60 mm board, sphere radius
and ball bar spacing came within 3.4 µm. Real lenses, projector gamma, thermal drift
and target accuracy add to that. The head needs a precision glass target, a
low-expansion bar, and calibrated spheres and a ball bar for its acceptance. An
aluminium bar shifts depth scale by about 0.01 mm per kelvin at this geometry.

## D24. Continuous scanning at near range (2026-11-01)

Motion during a phase cycle biases the decoded phase. `slscan motion-error` renders one
cycle on a moving plate and reports its depth error. The case modelled is the Global
Shutter camera at 60 fps, a four-step cycle on a 60 Hz projector and a matched projector
image. Error per cycle stays under 0.02 mm up to 2 mm/s in any direction, and up to
5 mm/s across the optical axis. Motion along the axis costs most: 28.9 µm at 5 mm/s.

Continuous scanning with this head is therefore a slow sweep at up to 2 mm/s, guided by
the preview at 15 depth frames per second and fused into the model. Full-resolution
static views stay available where a single view must carry the measurement. A 120 Hz
projector halves the motion error only with a camera that keeps pace, and the Global
Shutter camera stops at 60 fps, so that step needs a faster global-shutter camera.

## D25. Active stereo as the main measurement (2026-11-01)

Two synchronized Global Shutter cameras measure depth by matching a fixed random pattern
that a light throws onto the part. Uncoated parts scan this way, and the scanner can move
continuously, since each depth frame is one exposure of both cameras. We chose this over
the calibrated projector of D11, D13 and D23, because a projector that has to focus,
refresh and trigger at measurement grade was the hardest part to source. Here the light
is neither calibrated nor synchronized: any LED projector showing one random-dot image,
or a slide projector, serves.

Matching runs on rectified images. Semi-global matching at quarter resolution gives a
coarse disparity, Lucas-Kanade tracks each sample, and an affine window fitted with ECC
refines it, which follows the different foreshortening each camera sees on a slanted
surface. A sample is kept when its window lies on lit, matched surface, correlates at
0.95 or better, and lands within 3 px of its guess.

On the virtual bench, with uncoated artifacts and a calibration recovered from synthetic
board views, an 80 mm plate measures 8 µm RMS and a ball bar's spacing lands within
0.02 mm. Windows average the surface, so curved surfaces read slightly inside. A 21 px
window spans 2.3 mm at 0.5 m and puts points on a 25 mm sphere 42 µm inside it, which
shortens its fitted radius by about 0.1 mm. Smaller windows trade that bias for noise.
D28 removes this bias with second-order matching windows.

This supersedes the projector parts of D13 and D23. The capture node takes two Global
Shutter cameras on its two connectors, fired by one trigger pulse. The motion limits of
D24 apply to the phase-shift path only. That path and its code stay, as an option for
edge-sharp static scans with a calibrated projector.

## D26. Tracking and fusion reference (2026-11-02)

Each depth frame is aligned to the model by point-to-plane ICP, starting from the
previous frame's pose advanced by the last frame-to-frame motion. Its points then join
the model, kept as averages over 0.5 mm voxels, and model normals come from the twelve
nearest points. The model's frame is the first frame's left camera frame. SciPy's
KD-tree finds the nearest points.

The synthetic sweep has 25 frames, yawing ±25° and pitching ±8° around a mat with a ball
bar and a sphere. Its fused ball bar spacing lands within 5 µm of the truth, with a true
or a recovered calibration. Tracked poses drift by up to 0.34 mm against the truth, and
the model stays self-consistent because every frame is aligned to it. The fused mat
measures 20 to 24 µm RMS, and the spheres keep the window curvature bias of D25.

ICP converges for a few millimetres and degrees of motion between frames, which a sweep
at 30 fps stays well inside. Markers on the mat, a TSDF and loop closure come next.

## D27. Stereo sets on the capture protocol (2026-11-02)

Stereo capture uses the SLS1 protocol unchanged. Two sets carry it: `stereo`, a sequence
of frame pairs with its count in the request's `frames` field, and `stereo-calibration`,
one board pair. The set fixes the frame order, left then right per pair, so the frame
header needs no camera field. Both frames of a pair have the trigger's timestamp, and a
request without `frames` gets one pair.

The stand-in renders a sweep around a part on a mat, about 2° per frame as a sweep at
30 fps would move, and the next board pose for each stereo-calibration sequence.
`slscan fuse` runs matching, tracking and fusion on a captured sweep, and fails when no
frames overlap, which is the sign of a sweep too fast to track.

## D28. Second-order matching windows (2026-11-02)

An affine window fits a plane to the surface it covers and reads a curved surface inside
it (D25). A second-order window maps its pixels to the right image through a quadratic
displacement field, six terms on each axis, which follows curvature across the window.
Gauss-Newton fits those terms, with a gain and an offset for brightness, starting from
each affine match. A sample stops iterating once its step falls under 10⁻⁵ px. A match
is dropped when its column moves 1 px or more from its affine match.

With true calibration, a sample every 8 px and windows of 25 px, points on a 25 mm
sphere lie within 0.1 µm of it on average. Affine windows of 21 px put them 40.9 µm
inside. The second-order radius fits 0.6 µm long, and an 80 mm plate measures 9.0 µm
RMS. In the synthetic sweep of D26, second-order windows shrink ball bar sphere radius
errors from -116 and -76 µm to -33 and +8 µm.

Second-order matching takes about five times as long. NumPy and OpenCV match that plate
in 11.1 s on a Ryzen 7 7700X, where affine windows take 2.3 s. `slscan scan-stereo` uses
second-order windows of 25 px by default, since one view is its result. `slscan fuse`
and `slscan sweep-bench` match every frame of a sweep, so they default to affine windows
and take `--order 2`. Second-order matching on every frame waits on GPU kernels
(Phase 4).

## D29. Second-order refinement on the GPU (2026-11-02)

The second-order refinement of D28 runs as WGSL compute kernels through wgpu
(`app/src/refine.wgsl`), in single precision. A first pass takes Scharr gradients of
the right image. A second pass runs one invocation per sample, which rebuilds the
Gauss-Newton normal equations over its window, scales them by their diagonal and solves
them by Cholesky. Window coordinates are scaled to ±1 so that single precision suffices.
Sampling is bicubic with OpenCV's coefficients and zero outside the image. An exact
bicubic of that form moved NumPy results by at most 0.00003 px against `cv2.remap`.

`slscan stereo-vectors` writes a rectified view with its affine matches and their NumPy
refinement, and `sls-scan refine-check` compares the kernel's output with it. On the
virtual bench plate, a Radeon RX 7900 GRE refines 6296 samples with 25 px windows in
302 ms with upload and readback. Kept flags agree on every sample, and refined points
differ by at most 0.00012 px. NumPy spends about 8.8 s on this stage: 11.1 s for the
view with second-order windows less 2.3 s with affine windows (D28).

One invocation per sample runs its iterations serially, so the 296 samples of a sphere
view take 234 ms. A workgroup per sample, sharing the window's pixels, is the next step
toward 15 depth frames per second (Phase 4). Rectification, coarse disparity and the
affine stage stay on the CPU.
