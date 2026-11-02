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
view take 234 ms. D30 gives each sample a workgroup instead. Rectification, coarse
disparity and the affine stage stay on the CPU.

## D30. A workgroup per sample for refinement (2026-11-02)

The refinement kernel of D29 gives each sample a workgroup of 64 lanes. Lanes split the
window's pixels and add their partial normal equations in workgroup memory, in three
chunks of 40 sums. Lane 0 solves the system and updates the parameters that all lanes
read in the next iteration.

On the plate vectors, a Radeon RX 7900 GRE now refines 6296 samples in 115 ms with
upload and readback, down from 302 ms. A sphere view's 296 samples take 14 ms, down from
234 ms. Kept flags still agree on every sample, and refined points differ from NumPy by
at most 0.00012 px. A plate view at 115 ms is still above the 66 ms per frame that 15
depth frames per second allow (Phase 4).

## D31. Rectification on the GPU (2026-11-02)

Rectification runs as a WGSL kernel (`app/src/rectify.wgsl`) that computes OpenCV's
undistort-rectify map for each output pixel and samples the raw image bilinearly, with
zero outside it. The map needs only each camera's inverse rectified projection times
rotation, focal lengths, principal point and five distortion terms, which
`slscan.stereo.rectification_params` exports. No map images go to the GPU. In NumPy
double precision this formula reproduces OpenCV's maps within 0.00006 px, the precision
of their single-precision storage, and bilinear sampling of that form reproduces
`cv2.remap` exactly.

`sls-scan rectify-check` compares the kernel with OpenCV on the stereo vectors. On a
Radeon RX 7900 GRE both 1456 × 1088 images rectify in about 1 ms with upload and
readback. Under 0.01 % of pixels differ, each by one grey level, where single precision
rounds a half level the other way. Coarse disparity is the remaining CPU stage before
refinement.

## D32. Matching by Gauss-Newton alone (2026-11-02)

`slscan.stereo.direct_matches` matches without Lucas-Kanade or ECC, so a GPU can run the
whole refinement. Each sample starts from its coarse disparity with an affine window
fitted by Gauss-Newton. It is kept when it moves under 3 px and its window correlates at
0.95 or better, and a second-order window refines it from there. Second-order windows
straight from the coarse disparity failed on the ball bar. Windows lying partly off a
sphere still correlated above 0.95 with centres up to 0.77 mm off, which read the
spacing 35 to 132 µm long. The affine stage drops them, as ECC does.

With true calibration, a sample every 8 px and 25 px windows, `direct_matches` keeps 293
of the 294 sphere samples that affine matching keeps and all 6296 plate samples, within
0.00002 px at the 99th percentile. It also keeps 51 sphere samples that affine matching
drops.

| Artifact | Affine start | Direct |
|---|---|---|
| Sphere, points | 294 | 344 |
| Sphere, mean offset | +0.1 µm | +0.3 µm |
| Sphere, fitted radius | +0.6 µm | +3.7 µm |
| Sphere, form | 6.7 µm RMS | 6.8 µm RMS |
| Plate, points | 6296 | 6334 |
| Plate, flatness | 9.0 µm RMS | 9.1 µm RMS |
| Ball bar spacing, three noise seeds | +27.3, +23.5, +17.7 µm | +25.0, +20.0, +17.7 µm |

NumPy takes about twice as long, since every sample runs two Gauss-Newton passes. Both
paths read this ball bar 18 to 27 µm long in single views, and affine windows alone read
it 12 to 16 µm short. D34 traces the long reading to coverage of the spheres.

## D33. Direct matching on the GPU (2026-11-02)

The refinement kernel runs both stages of D32. Each stage sets its window shape, affine
or second order, its move limit and an optional correlation gate. The gate is computed
by all lanes of a sample's workgroup on its final window. Each stage starts from the
right point of the stage before, with an identity shape, as `direct_refine` does.

`slscan stereo-vectors` adds the coarse samples with their direct refinement, and
`sls-scan refine-check` runs both stages on them. On the sphere, ball bar and plate
vectors, kept flags agree on every sample and refined points differ by at most
0.00012 px. A Radeon RX 7900 GRE takes the 6334 plate samples through both stages in
260 ms with upload and readback.

The kernel abandons a sample whose iterate strays 32 px from its start. Second-order
iterates started with an identity shape pass 2 px on the way to some matches that NumPy
keeps, so this guard sits well above any move limit. Coarse disparity and sample
selection remain on the CPU.

## D34. Block matching for the coarse disparity (2026-11-02)

Direct matching takes its coarse disparity from `slscan.stereo.block_disparity`. It
correlates 7 px blocks at quarter resolution over the whole disparity range and refines
the best disparity by a parabola. A disparity is kept when it correlates at 0.6 or
better, leads every disparity more than one step away by 0.05, and the right pixel picks
it back within one step. Two passes then fill each hole that has three or more valid
neighbours. Every step suits a GPU kernel, and the Rust application has no OpenCV
for semi-global matching.

Block matching is also the better start. Over three noise seeds, with 25 px windows and a
sample every 8 px:

| Calibration and start | Ball bar points | Ball bar spacing | Sphere radius | Sphere form |
|---|---|---|---|---|
| True, semi-global | 231 to 237 | +17.7 to +25.0 µm | +3.7 to +6.0 µm | 6.6 to 6.9 µm RMS |
| True, block | 615 to 617 | +1.0 to +2.7 µm | -0.4 to +1.5 µm | 6.6 to 7.0 µm RMS |
| Recovered, semi-global | 240 to 241 | +7.0 to +14.1 µm | -13.2 to -12.0 µm | 11.0 to 11.7 µm RMS |
| Recovered, block | 612 to 616 | +2.4 to +3.9 µm | -10.4 to -6.6 µm | 6.6 to 7.1 µm RMS |

The plate measures 9.1 µm RMS from either start. Semi-global matching leaves much of each
ball bar sphere without valid disparity, and it seems likely that fits on those partial
caps read the spacing long. NumPy takes 203 ms for block matching of a view and OpenCV
4 ms for semi-global matching, so the affine path of `slscan scan-stereo` keeps
semi-global matching.

## D35. Block disparity on the GPU (2026-11-02)

The block matching of D34 runs as four WGSL entry points (`app/src/coarse.wgsl`). One
pass averages 4 × 4 pixels of both rectified images and rounds half to even, as OpenCV's
area resize does for 8-bit images. One invocation per reduced pixel and disparity fills a
correlation volume, with box sums over borders reflected as `cv2.boxFilter` reflects
them. One pass per reduced pixel picks the disparity and applies the three checks. Two
fill passes alternate between a pair of buffers.

`sls-scan coarse-check` compares the kernel's reduced map with `block_disparity_small`
in the stereo vectors. On the sphere, plate and ball bar views, validity agrees on every
reduced pixel and disparities agree within 0.0001 reduced pixels. It seems likely that
box sums of whole grey levels, exact in single precision, keep the two in step. A Radeon
RX 7900 GRE matches 364 × 272 reduced pixels over 189 disparities in about 4.5 ms with
upload and readback, where NumPy takes 203 ms (D34). Sample selection and triangulation
are the CPU steps left in a depth frame.

## D36. A depth frame from a raw pair (2026-11-02)

`app/src/depth.rs` measures one raw stereo pair. Rectification, block disparity and both
stages of direct matching run on the GPU, and sample selection and triangulation run on
the CPU. Selection uses summed-area tables. A pixel is lit when its 5 × 5 sum reaches
138, which is where `cv2.blur` rounds to 6 or more. Triangulation solves OpenCV's linear
system by least squares in double precision, which matched `cv2.triangulatePoints`
within 0.00001 µm in NumPy.

`sls-scan depth-check` runs the path on the raw stereo vectors and compares its points
with `direct_matches` and `triangulate`. Both keep the same samples. On a Radeon RX 7900
GRE, with upload and readback at every stage:

| View | Points | Frame | Refinement | Point distance, median and largest |
|---|---|---|---|---|
| Sphere | 303 | 46 ms | 35 ms | 0.005 µm, 0.78 µm |
| Ball bar | 615 | 57 ms | 46 ms | 0.005 µm, 2.8 µm |
| Plate | 6374 | 276 ms | 265 ms | 0.006 µm, 2.2 µm |

By δz = z²·δd/(f·b) with z = 0.5 m, f = 4405 px and b = 0.2 m, 2.8 µm is 0.01 px of
disparity. It seems likely that those few samples start from guesses rounded to single
precision and settle slightly differently. Refinement takes almost all of a plate frame,
which runs four times over the 66 ms that 15 depth frames per second allow (Phase 4).
Each kernel also opens its own device and reads its output back, which a shared device
would avoid.

## D37. One device for every kernel (2026-11-02)

`app/src/gpu.rs` opens one wgpu device, and each kernel is built on it. Each kernel keeps
its byte-slice entry point for the checks and gains one that takes storage buffers. A
depth frame uploads the raw pair once and rectifies into buffers that block matching
and refinement read in place. The right image's gradients are computed once for both
refinement stages. The rectified left image, the reduced disparity and the refined
points still come back to the host, for sample selection and triangulation.

On a Radeon RX 7900 GRE the sphere frame of D36 takes 42 ms, down from 46 ms, and the
plate frame 272 ms, down from 276 ms. Refinement keeps almost all of the plate frame,
at 262 ms, so its speed is the next step toward 66 ms.

## D38. Refinement in registers and workgroup memory (2026-11-02)

The refinement kernel of D30 kept its 119 normal-equation sums and its 14 × 14 solve in
private arrays indexed by loop variables. RADV reported 80,896 bytes of scratch memory
for it, which is where a compiler puts arrays it cannot keep in registers, and
refinement took 262 ms of a plate frame (D37).

Three changes remove the scratch. The solve runs in workgroup memory, with the Cholesky columns spread over 14 lanes. Each
lane accumulates its pixels' sums in 3 × 3 matrices and vectors with fixed indices,
which stay in registers, and writes them to workgroup memory once per iteration. The
reduction runs in two halves, which cuts the workgroup memory to 17 KB and lets a
compute unit run more workgroups at once.

| Kernel state | Scratch | Workgroup memory | Plate direct set | Plate frame |
|---|---|---|---|---|
| D30 | 80,896 B | 11 KB | 260 ms | 272 ms |
| Solve in workgroup memory | 0 | 32 KB | 203 ms | 215 ms |
| Sums in registers | 0 | 32 KB | 64 ms | 74 ms |
| Two-pass reduction | 0 | 17 KB | 48 ms | 61 ms |

Results are unchanged: every check keeps the same samples and matches within
0.00012 px. On a Radeon RX 7900 GRE a plate frame of 6374 points takes 61 ms, under the
66 ms of 15 depth frames per second, a ball bar frame 23 ms and a sphere frame 19 ms.
Samples take 13 to 14 Gauss-Newton iterations on average with the exit at 10⁻⁵ px. An
exit at 10⁻⁴ px would save a quarter of them and move positions by under 0.001 px,
which we keep in reserve for the CPU fallback.

## D39. Block statistics once per pixel (2026-11-02)

The correlation pass of D35 summed five block products at every reduced pixel and
disparity: the left block's sum and sum of squares, the shifted right block's, and their
cross sum. Four of those depend on one image alone. A new pass takes each image's block
sum and sum of squares once per reduced pixel. The correlation pass reads the left's at
its pixel and the right's at the column its disparity points at, and loops over the
block for the cross sum alone.

Each image's block sums reflect at that image's own borders, in the cross sum too, where
D35 reflected the left coordinate and then shifted it. With the old cross sum, border
pixels' right-block sums no longer matched the pixels the cross sum paired, and a few
passed the checks with inflated correlations. The NumPy reference follows the same rule,
and the kernel matches it on every reduced pixel of the sphere, plate and ball bar views.

Direct matching keeps the same samples on all three views, and the ball bar, sphere and
plate figures of D34 are unchanged within 0.1 µm. On a Radeon RX 7900 GRE the coarse
pass stays at about 5 ms. The pass is for the CPU fallback, where block disparity took
0.7 to 1.7 s of a depth frame on the hosted runners, and the application gate's log
shows its effect there.

## D40. Live depth frames in the application (2026-11-02)

A stereo run in the application sends each frame pair through the depth frame of D36 on
a worker thread that owns the device. Pairs that arrive while a frame is in flight are
skipped and counted, so the preview follows the capture at the GPU's rate instead of
queueing behind it. The scene panel draws the latest frame's points from the front,
shaded by depth, with the point count, depth range and the frames run and skipped. The
preview panel shows each frame's point count and stage times.

`slscan depth-params` writes one JSON file with everything the depth frame takes for a
calibration: image size, matching settings, the block matching search and settings,
depth bounds and sampling, and both cameras' rectification. The application loads it
from its configuration panel. `sls-scan depth` runs the same path without the window
and writes the last frame's points as PLY, and the application gate runs it on two
frames from the stand-in node.

On a Radeon RX 7900 GRE the stand-in's sweep frames, a mat with a ball bar and a
sphere, give about 16,000 points in 108 to 152 ms each, with refinement at 97 to 135 ms
of it. Point count sets the frame time, so a sparser sample grid in the parameters
trades density for rate.

## D41. Stereo calibration from the panel (2026-11-02)

With the stereo set chosen, the calibration panel captures one board pair per pose into
the session directory, detects the board in each image through the Python tool, and
solves with `calibrate-stereo`. The solve writes `stereo.yaml` beside the poses, and
`depth-params` then writes `depth.json` next to it, which the panel takes as the
parameters of the next stereo run. `sls-scan calibrate --set stereo` runs the same steps
without the window, and the application gate runs it on twelve board pairs from the
stand-in node before the depth frames of D40.

On the stand-in's twelve poses the solve reprojects to 0.076 and 0.065 px with a
baseline of 200.19 mm, the figures of D25 for the same views. Two depth frames from that
calibration give about 16,000 points each in 112 to 146 ms on a Radeon RX 7900 GRE. The
projector path of D21 keeps its own branch of the panel, so one panel serves both heads.
