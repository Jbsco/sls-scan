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

## D42. Tracking and fusion in the application (2026-11-02)

`app/src/fuse.rs` ports `track.fuse` of D26 to Rust. Each depth frame is aligned to the
model by point-to-plane ICP from the last pose advanced by the last motion, then joins
the model as 0.5 mm voxel averages.

A hash grid of cubic cells replaces SciPy's KD-tree. ICP takes the nearest model point
within each distance limit from a grid of that cell size. Normals take each point's
twelve nearest neighbours by searching outward one shell of cells at a time. Each ICP
step solves its 6 × 6 normal equations, and each normal is the smallest eigenvector of
its neighbours' covariance, by Jacobi rotations.

`slscan fusion-vectors` writes the depth frames of a six-frame synthetic sweep with
their NumPy poses and model, and `sls-scan fusion-check` runs the Rust tracker on the
same frames. Poses, voxels and points agree to the last printed digit, under 0.001 µm,
and the application gate runs the check.

The window's depth worker fuses every frame it runs and the scene panel draws the model.
`sls-scan depth` fuses its frames and writes the model as PLY. On six stand-in sweep
frames the model grows to 51,701 points with an ICP residual of 20 to 21 µm, and fusion
takes 350 to 440 ms per frame on one core. Like the reference, it recomputes every
model normal for each frame, which is the cost to cut next.

## D43. Fusion on every core (2026-11-02)

Fusion of D42 took 350 to 440 ms per stand-in frame, in ICP for 258 to 282 ms and in
model normals for 51 to 122 ms. Two changes cut it. Grid cells hash with a
multiply-and-rotate function. The standard hasher resists keys chosen to collide, which
voxel indices from a scan are not. Each ICP iteration splits its correspondences
over scoped threads that each sum their own normal equations, added in a fixed order,
and model normals split over the same threads by point.

| Version | Fusion per frame, six stand-in frames |
|---|---|
| D42 | 53 to 444 ms |
| Cell hasher | 38 to 278 ms |
| Cell hasher and 16 threads | 10 to 58 ms |

Times are on a 16-thread Ryzen 7 7700X as the model grows from 16,292 to 51,701 points.
The model, poses and ICP residuals are unchanged, and the fusion check still agrees with
NumPy under 0.001 µm. A stand-in frame now takes about 110 ms for depth and 60 ms for
fusion at most, so refinement on the GPU leads again.

## D44. A board that fills the near-range view (2026-11-02)

Phase 1 asks that repeated calibrations agree within 0.1 % in focal length. `slscan
calibration-repeat` tests that on the virtual bench. It calibrates the stereo head from
independent sessions whose board poses each move by up to 5° and 10 mm, as a person
placing the board by hand would, and reports the largest change between any two
sessions.

With the 90 × 60 mm board of D25 the sessions disagree by 0.311 % in focal length and
0.305 mm in baseline, although each reprojects under 0.11 px. That board
covers the middle of the 157 × 117 mm field at 0.5 m, where focal length trades against
the distortion terms.

| Board | Poses | Focal length change | Baseline change |
|---|---|---|---|
| 9 × 6 squares of 10 mm, 90 × 60 mm | 12 | 0.311 % | 0.305 mm |
| 9 × 6 squares of 10 mm, 90 × 60 mm | 36 | 0.194 % | 0.102 mm |
| 12 × 9 squares of 10 mm, 120 × 90 mm | 24 | 0.130 % | 0.073 mm |
| 14 × 10 squares of 10 mm, 140 × 100 mm | 24 | 0.088 % | 0.069 mm |
| 11 × 8 squares of 12 mm, 132 × 96 mm | 12 | 0.112 % | 0.079 mm |
| 11 × 8 squares of 12 mm, 132 × 96 mm | 24 | 0.054 % | 0.038 mm |

Five sessions per row. We chose an 11 × 8 board of 12 mm squares and 9 mm markers,
which covers most of the field, and two passes of the pose table, 24 poses, for the
near-range head. Those are the command's defaults, and the test suite runs them.
Synthetic views, stand-in captures and the published accuracy figures still use the
90 × 60 mm board, so their numbers stay comparable with earlier entries.

## D45. One view at a metre on the virtual bench (2026-11-02)

Phase 2 asks for a plate under 0.2 mm RMS over 0.5 m and a scale bar within 0.2 mm in
one view. `slscan stereo-bench` checks both with the metre head of `docs/hardware.md`:
6 mm lenses on a 0.35 m baseline crossing at 1 m, calibrated from twelve poses of the
A2 board. It matches directly (D32, D34) with 21 px windows.

A 0.5 m plate tilted 10° and 15° measures 0.041 mm RMS over 467 × 476 mm. A scale bar of two 50.8 mm spheres
400 mm apart, sampled every 4 px, reads +0.022, +0.070 and +0.040 mm on three noise
seeds.

Smaller spheres miss the limit. Spheres of 38.1 mm are about 66 px across at 1 m, which
leaves 45 to 74 samples clear of their edges, and they read 0.2 to 0.96 mm long with
21 and 25 px windows. Affine windows of 15 px bring them to 0.09 to 0.13 mm. We chose
spheres of 50 mm or more for the metre scale bars, which the hardware notes list.

## D46. A paced stereo stream on the stand-in (2026-11-02)

Phase 3 asks for frame pairs captured in step at 30 fps with no drops over a minute. A
stereo request now takes `rate_hz`, the trigger rate in pairs per second, with 0 for as
fast as the node can. The stand-in renders an eight-pair sweep once and plays it in a
loop on a trigger clock at that rate.

It drops a pair whose trigger time passed more than
one period before it gets to it, as a node with a full queue would, so a client that
falls behind shows missing indices. The client reports the largest gap between
triggers and whether both frames of every pair carry one timestamp.

One minute at 30 Hz, 1800 pairs, reached the release build with no frame missing, every
pair in step, a mean interval of 33.333 ms and a largest gap of 33.333 ms. That is
95 MB/s of image data over loopback, about three quarters of a gigabit link. This tests
the protocol, the client and the host. The Pi's capture and its network link wait on
the rig. The application gate streams 150 pairs at 30 Hz.

## D47. Replay of recorded stereo pairs (2026-11-02)

Phase 4 asks for kernel outputs within tolerance of the Phase 2 reference on recorded
frames. `sls-scan capture --out DIR` writes each frame it receives as an 8-bit PNG under
its frame name, so a stereo set lands as `frame_NNN/left.png` and `frame_NNN/right.png`.
`slscan replay-reference` runs `direct_matches` and `triangulate` on each recorded pair
with a depth parameter file and writes `frame_NNN/reference.bin`. `sls-scan replay` runs
the GPU depth frame on each pair with that file and compares its points.

A frame matches when samples kept on one side only are 0.5 % of the reference or less,
the 99th percentile distance is 1 µm or less, and no more than 0.1 % of points lie
beyond 5 µm. `sls-scan depth-check` had the first two limits and kept every point within
5 µm, which one replayed point of 4042 missed at 7.4 µm, so it now uses this rule. By
D36's relation with z = 0.5 m, f = 4405 px and b = 0.2 m, 7.4 µm is 0.026 px of
disparity.

Three stand-in pairs recorded at a 16 px step with the near head's calibration gave 3979
to 4042 points each. No sample was missing or extra. Median distance was 0.007 µm and
the 99th percentile 0.48 to 0.64 µm, with one point beyond 5 µm on one pair. Each GPU
frame took 60 to 68 ms on a Radeon RX 7900 GRE, and NumPy took about 12 s per pair.

The application gate records two pairs after its panel calibration and replays them on
lavapipe. These pairs come from the stand-in's renderer. Pairs from the cameras wait on
the rig.

## D48. Triangulation on the CPU (2026-11-02)

Phase 4 listed a triangulation kernel. We keep triangulation on the CPU in double
precision, because it takes under a millisecond there. A release build triangulates
15,600 to 16,300 points of the stand-in's sweep in 0.6 to 0.8 ms on a Ryzen 7 7700X.
Each of those frames takes 114 to 157 ms, and refinement 103 to 140 ms of it. Fusion
runs on the host (D42), so points come back to it with or without a kernel. The 41 ms
in the application gate's log on hosted runners comes from a debug build.

Single precision would also move points. On 16,000 points of the near head between 0.4
and 0.6 m, the same least squares in float32 lands a median 0.03 µm, a 99th percentile
0.16 µm and at most 0.39 µm from true positions, measured in NumPy. In float64 every
point lands within 0.0001 µm.

## D49. A preview without refinement (2026-11-02)

Phase 4 asks for 3 depth frames per second on the CPU fallback. On llvmpipe on a
GitHub-hosted runner, a 16,000-point frame spent 20.7 to 24.6 s in refinement and 1.5 to
1.7 s in block disparity. No sample spacing brings refinement near 333 ms there, so the
CPU fallback previews from block disparity alone.

A preview matches blocks at 1/8 resolution. It keeps each reduced pixel whose 8 × 8 block
of the rectified left image averages `min_level` or more, so unlit background drops out,
and triangulates it at its block's centre. `slscan depth-params` writes the preview's
scale and search. `stereo.preview_matches` is the NumPy reference, and
`DepthFrame::preview` runs rectification and block matching on the GPU.

Over three noise seeds, previews of the near head's plate give about 7,100 points 0.16 mm
RMS from the refined plane, with a 99th percentile of 0.45 mm. On spheres the median
error is 0.33 to 0.41 mm. Blocks across a sphere's silhouette mix it with background and
land up to 40 mm off. A preview shows coverage and is not a measurement.

Replay compares previews with `preview.bin` under the depth frame's limits times the
preview scale, since a reduced pixel spans 8 px: a 99th percentile within 8 µm and at
most 0.1 % beyond 40 µm. Three stand-in pairs gave 18,483 to 18,624 points each with one
point missing, a 99th percentile of 0.59 to 0.96 µm and three points beyond 40 µm on one
pair. On llvmpipe one preview's 99th percentile reached 1.27 µm, past the depth frame's
1 µm.

A preview took 2.6 to 3.3 ms on a Radeon RX 7900 GRE. On llvmpipe on the hosted
runner, two previews of about 18,600 points took 137 and 142 ms in a debug build, about 7
per second, where full depth frames of about 4,000 points took 3.2 to 3.3 s.

## D50. Previews on software adapters (2026-11-02)

wgpu reports lavapipe as a CPU device. On such an adapter the application and `sls-scan
depth` run the preview of D49 in place of full depth frames, so the CPU fallback keeps its
rate. `sls-scan depth --mode depth` or `--mode preview` overrides the choice, and the
application gate runs both modes against the stand-in.

Previews are not fused. On three stand-in sweep frames, ICP aligned the second and third
previews with residuals of 137 and 148 µm, where full depth frames of the same sweep
aligned at 17 to 21 µm. Silhouette blocks up to 40 mm off would also enter the model. The
scene draws each preview's points, and the fused model waits for full depth frames on a
GPU.

## D51. TSDF fusion reference (2026-11-02)

Phase 5 asks for TSDF fusion of the depth stream, and Phase 4 for a coarse TSDF kernel.
`slscan/tsdf.py` is the NumPy reference for both. Each tracked depth frame becomes a depth
image at its sample grid. Every voxel projects into that image, reads depth by bilinear
interpolation and averages its distance to the surface along the camera axis. Distances
are truncated in front of the surface and dropped more than one truncation behind it.
Interpolation skips grid cells whose four depths differ by more than one truncation, so
silhouettes do not bridge to the background.

The surface is the set of zero crossings
between observed neighbours, and `slscan sweep-bench --tsdf` measures it in place of the
fused points.

On the 25-frame sweep of D26 with second-order windows and a sample every 8 px, over
three noise seeds:

| Fusion | Ball bar spacing | Sphere radii | Mat flatness |
|---|---|---|---|
| Points, 0.5 mm voxels | +2.4 to +11.4 µm | -7.4 to +21.3 µm | 15.5 to 17.0 µm RMS |
| TSDF, 0.5 mm voxels, 1 mm truncation | -10.5 to -5.2 µm | -18.3 to -9.9 µm | 4.1 to 4.8 µm RMS |
| TSDF, 1 mm voxels, 2 mm truncation | -23.8 to -12.6 µm | -33.1 to -19.6 µm | 4.1 to 5.0 µm RMS |

Tracked poses drifted by 36 to 170 µm and 0.025 to 0.034° against the truth. It seems
likely that the TSDF mat is flatter because each voxel averages distances from many
frames. TSDF spheres read small by an amount that grows with the voxel, perhaps from
linear zero crossings of a curved distance field. NumPy integrates the 25 frames into 18 million
voxels of 0.5 mm in 8 to 18 s.

Samples every 16 px, 1.8 mm apart at 0.5 m, are too sparse. On a five-frame sweep the
TSDF then read sphere radii up to 224 µm small and the ball bar 120 to 138 µm off. Grid
cells on a sphere's sides then span more than one truncation, which likely leaves those
sides unobserved. A coarse TSDF needs
samples every 8 px or closer.

## D52. TSDF integration on the GPU (2026-11-02)

`app/src/tsdf.wgsl` folds one depth frame into the volume of D51 with one invocation per
voxel. `app/src/tsdf.rs` builds each frame's depth grid on the CPU from its points and
the left camera's rectification, as `tsdf.depth_grid` does, and keeps distances and
weights on the device. The host composes one matrix and offset from the frame's pose and
the rectifying rotation, and the kernel moves voxel centres into the rectified frame with
them in single precision.

`slscan tsdf-vectors` writes four frames of a short sweep, their tracked poses and the
NumPy volume at 1 mm voxels and 2 mm truncation. `sls-scan tsdf-check` integrates the
same frames on the GPU and compares. Of 340,622 observed voxels, one has another weight.
Distances differ by a median of 0.0001 µm, a 99th percentile of 0.039 µm and at most
0.11 µm. The check allows 0.1 % of observed voxels to differ in weight and a 99th
percentile of 0.1 µm.

On a Radeon RX 7900 GRE the four frames take 13.8 ms with upload and readback. The
25-frame sweep of D51 at 0.5 mm voxels and 1 mm truncation, 17.9 million voxels, takes
77.9 ms, and 209 of its 9.3 million observed voxels differ in weight.

## D53. TSDF surface on the GPU (2026-11-02)

A second entry point of `app/src/tsdf.wgsl` finds the surface of D51 on the device, so
the application can draw it without reading the volume back. One invocation per voxel
checks its next voxel along each axis. It appends each zero crossing between observed
voxels to a list through an atomic counter, with its key: three times the flat index of
its lower voxel plus its axis. `tsdf.surface` returns the same keys, and `slscan
tsdf-vectors` writes the NumPy surface with them.

`sls-scan tsdf-check` matches GPU and NumPy crossings by key. On the four-frame vectors
at 1 mm voxels, all 9,917 crossings match, with a median distance of 0.021 µm and a
99th percentile of 0.31 µm. Two lie beyond 5 µm. It seems likely that a crossing between
two nearly equal distances moves far for a small change in either.

The check allows 0.5 % of crossings on one side only, a 99th percentile of 1 µm and 0.5 %
beyond 5 µm. On llvmpipe on a GitHub-hosted runner, 13 of the 9,917 crossings lie beyond
5 µm. The 25-frame sweep at 0.5 mm voxels has 64,309 crossings against 64,303, 48 on one
side only, and 60 beyond 5 µm.

On a Radeon RX 7900 GRE extraction takes 0.8 ms for 624,870 voxels and 3.4 ms for 17.9
million, with readback of the points.

## D54. Live TSDF in the application (2026-11-02)

Each depth frame of a stereo run now folds into a coarse TSDF at its tracked pose, and
the scene panel draws that volume's surface in place of fused points. Point fusion stays
underneath, since ICP aligns each frame to its points. A volume opens on a run's first
frame, 40 mm around its points, with 1 mm voxels and 2 mm truncation, and drops surfaces
outside that box. `sls-scan depth --surface` writes its last surface as PLY.

Eight frames of a stand-in sweep, run headless, gave 14,587 to 15,892 surface points. On
a Radeon RX 7900 GRE, opening the volume, integrating and reading a surface back took
20.2 ms for frame one and 1.8 to 2.2 ms for each later frame. Surface lists of two
million points are allocated on the device, which wgpu fills with zeros. Uploading zeros
from the host took each later frame to 4.7 to 7.3 ms.

Against truth, this surface reads the ball bar 0.9 µm long and its spheres 34.8 and
38.5 µm small, a voxel bias D51 shows, with a mat flatness of 4.8 µm RMS.
## D55. Tracking from the previous pose (2026-11-02)

`slscan sweep-bench --loop` sweeps a closed loop: yaw follows -25° cos and pitch 8° sin
of one turn about the target, and a last frame repeats the first. Tracking that started
each frame from the last pose advanced by the last motion (D26) diverged on it, at frame
38 of a 48-frame loop and frame 39 of a 96-frame one. Pose error grew from about 0.2 mm
to 9 mm over four frames as the ICP residual rose from 21 to 35 µm.

Started from true poses, ICP moved frames 8 to 38 of the 96-frame loop by 62 µm at most.
In every view the weakest direction of the point-to-plane system has about 1e-5 of the
strongest direction's weight, because the mat fills most of it. It seems likely that the
motion term carried each frame's error along that direction into the next start, where
ICP barely corrects it. We now start each frame from the previous pose, in `track.fuse`
and its Rust port. Two 25-frame sweeps of D51 reach the same largest and last pose
errors either way.

With affine windows and a sample every 12 px:

| Loop | Pose error, largest | Last frame | Ball bar | Mat | Mat, second half against first |
|---|---|---|---|---|---|
| 48 frames | 118 µm, 0.051° | 13.0 µm | -12.9 µm | 22.2 µm RMS | -0.2 µm mean, 18.5 µm RMS |
| 96 frames | 85 µm, 0.041° | 14.9 µm | -4.1 µm | 27.9 µm RMS | -0.2 µm mean, 23.1 µm RMS |

Mat surfaces met again on the way back land on those of the way out, with no doubling,
and the loop closes within 15 µm. Sphere centres fitted to each half alone differ by 12
to 56 µm, perhaps because affine window bias changes with the view. A pose graph for
loop closure waits for scans where drift outgrows this. A test tracks a 48-frame loop of
point frames in seconds, where the old start diverged at frame 37.
## D56. Isolated points dropped (2026-11-02)

Phase 2 lists filtering after triangulation. `stereo.depth_points` now drops each point
with no other point within three sample spacings, where one spacing at depth z is the
sample step times z over the focal length. A failed match tends to land far from any
surface, where it stands alone. Points near an edge or on a curved surface keep
neighbours on at least one side.

Against truth on rendered sweeps:

| Frames | Points | Dropped | Beyond 1 mm, dropped | Within 0.1 mm, dropped |
|---|---|---|---|---|
| 25, affine windows, every 8 px | 236,023 | 7 | 2 of 3 | 0 |
| 25, second-order windows, every 8 px | 230,241 | 7 | 0 of 0 | 3 |
| 96-frame loop, affine, every 12 px | 403,643 | 15 | 8 of 12 | 6 |

Every point more than 2 mm off is dropped, up to 95 mm. Points 1.0 to 1.6 mm off keep
neighbours and stay. Ball bar and sphere figures of these sweeps move by up to 8 µm and
the mat by 0.1 µm, since artifact fits take points near the artifact anyway. Exported
and fused clouds of these sweeps then contain no point more than 2 mm off. The
application's depth frames do not filter yet.
## D57. Meshes by surface nets (2026-11-02)

Phase 7 asks for meshing and export. `slscan/mesh.py` meshes a TSDF volume by surface
nets. Each cell between eight observed voxels whose distances change sign gets one
vertex, the mean of its edges' zero crossings. Each crossing edge joins the four cells
around it in a quad facing the positive side, split along the diagonal that leaves the
larger smallest triangle.

The plan named Poisson reconstruction from Open3D, which needs an oriented cloud. A TSDF
already has a signed distance, so its zero set meshes directly, and Open3D is absent on
the host. We chose surface nets over marching cubes because it needs no case table, and
its vertices come from the zero crossings `tsdf.surface` finds. `mesh.write_mesh` writes
binary PLY, ASCII OBJ or binary STL by extension.

A sphere of 12.5 mm radius, sampled as an exact distance field at 0.5 mm voxels, meshes
closed: every edge belongs to two triangles, and every triangle faces out. Its vertices
fit a sphere 4.5 µm small with 1.7 µm RMS, and none lies more than 7.5 µm off. A grid
whose voxel centres fall on the sphere gives coincident vertices and zero-area
triangles, so the test grid is offset by a fraction of a voxel.

`slscan sweep-bench --tsdf --mesh` meshes the sweep's volume and measures its vertices.
The 25-frame sweep with affine windows, a sample every 8 px, 0.5 mm voxels and 1 mm
truncation gives 63,014 vertices and 123,638 triangles, 6.2 MB as STL. Its vertices read
the ball bar 31.1 µm short, the spheres 111.2 and 140.2 µm small, mostly the affine
window bias of D26 with the voxel bias of D51, and the mat at 2.9 µm RMS. Decimation,
hole filling and datum alignment come next.

## D58. Decimation by quadric error (2026-11-02)

Phase 7 names decimation to the face count CAD accepts. `mesh.decimate` collapses edges
in order of least quadric error, after Garland and Heckbert. Each vertex sums the planes
of its triangles, weighted by area. Each boundary edge adds a plane through it at right
angles to its triangle, weighted a thousand times, so a scan's open border keeps its
outline. A collapse that would turn a triangle over is skipped. The merged vertex sits
where the summed quadric is least, or at the edge's midpoint when that point is undefined
or lies more than four edge lengths away.

A surface-nets sphere of 23,556 triangles decimated to 4,000 stays closed, with every
vertex within 11 µm of the true sphere and its radius 2.3 µm long. On the 25-frame sweep
with second-order windows, at 0.5 mm voxels and 1 mm truncation:

| Triangles | Off the full mesh, 99th percentile and largest | Ball bar | Time |
|---|---|---|---|
| 121,708 | | -10.6 µm | |
| 12,170 | 10 µm, 34 µm | -15.8 µm | 36 s |
| 2,434 | 26 µm, 50 µm | -37.5 µm | 41 s |

Deviation is measured from points sampled on decimated triangles, along the full mesh's
vertex normals. NumPy with a Python heap sets a reference pace, not a working one.
`slscan sweep-bench --mesh --faces` decimates before writing.

## D59. Live frames filtered (2026-11-02)

The application now drops isolated points, as D56 does, from each depth frame before
fusion and the TSDF. `fuse::drop_isolated` ports `stereo.drop_isolated` on the hash grid
that fusion already uses for neighbour searches. A depth frame itself stays unfiltered,
so `sls-scan depth-check` and `sls-scan replay` still compare it with direct matching.
Eight frames of a stand-in sweep, run headless, lost 0 or 1 points each out of 15,492 to
16,249.

## D60. Datum residual report (2026-11-02)

Phase 7 ends in a report of scale bar lengths and datum residuals. `slscan report` reads
a PLY cloud or mesh and a datum file. The file names spheres on the object, each with
its measured centre in the object's frame, its radius and a seed near it in the scan's
frame, and lists scale bars as pairs of spheres.

Each sphere is fitted to the points within its radius and 1.5 mm of its seed.
`geometry.rigid_fit` aligns the fitted centres to the measured ones by the SVD method of
Kabsch, and a datum's residual is its distance after that alignment. A scale bar's error
is its fitted length less its measured length. The report prints both, writes them as
JSON with `--out`, and fails above `--limit-mm`, 1 mm by default as in the Phase 7
acceptance.

`slscan sweep-bench --datums` writes the sweep scene's ball bar spheres and third
sphere as a datum file, seeded at their true positions. On the 25-frame sweep with
second-order windows and a sample every 8 px, the mesh of a 0.5 mm TSDF gives datum
residuals of 5.9, 8.5 and 8.3 µm and a ball bar 10.7 µm short. Seeds from markers or a
picked point wait on the rig.

## D61. Small holes filled (2026-11-03)

The mesh of the 25-frame sweep, with second-order windows and a 0.5 mm TSDF, has seven
boundary loops and no edge shared by more than two triangles. Two are holes of 2.0 and
6.4 mm perimeter. Four of 55 to 90 mm perimeter seem likely to be the unseen undersides
of the three spheres and the mat they shade, and one of 632 mm is the scan's outer edge.
A flat patch across a sphere's underside would read as surface that was never measured.

`mesh.fill_holes` closes each loop up to a perimeter limit with a fan of triangles from
a new vertex at the loop's centroid, wound to match the triangles along it. Larger loops
stay open. `slscan sweep-bench --mesh --fill-mm` sets the limit. At 10 mm the sweep's
two holes close, and the ball bar moves from -10.7 to -10.9 µm. A test opens a 1.5 mm
cap and a 6 mm cap in a meshed sphere and checks that a 20 mm limit closes the first,
facing out, and leaves the second open.

## D62. Meshes from the live volume (2026-11-03)

`sls-scan depth --volume DIR` reads the live TSDF of D54 back from the device after a run
and writes it as meta.json, tsdf.bin and weight.bin, the layout `slscan tsdf-vectors`
writes. `slscan mesh-volume` meshes such a directory with the reference mesher of D57,
fills holes with `--fill-mm` and decimates with `--faces`. We kept meshing in Python
rather than porting it to Rust, since an export runs once per scan and the NumPy code is
already the reference. The application gate saves the volume of its two depth frames and
meshes it.

Eight frames of a stand-in sweep gave a 1 mm volume of 21 MB of distances and as much
of weights. It meshes to 15,308 vertices and 29,474 triangles in about 2 s. Against
truth the mesh reads the ball bar 0.9 µm long and its spheres 46.0 and 48.1 µm small,
the coarse voxel bias of D51.

## D63. Switchable pattern light, markers optional (2026-11-03)

Pose comes from the part's geometry by default, with no markers and no preparation. ICP
tracking from geometry closes loops on the virtual bench (D55), and parts with shape in
every direction constrain it fully. Flat, smooth or turned parts constrain it weakly: on
the mat of D55 the weakest direction had about 1e-5 of the strongest's weight. Markers on
a mat around a part stay an option for bench work, never a requirement, and they are
printed, never projected.

The pattern light becomes switchable. One GPIO line on the Pi drives a MOSFET in the
light's supply, so the node can drop the light for chosen frames between exposures. A
frame without the light sees the scene under room light. That serves colour texture
(Phase 11), tracking from texture a part or mat already has where geometry is weak, and
printed markers if a job uses them. We chose this over a ring light at each camera,
since a ring light only helps with retroreflective targets, which are preparation.

Phase 6 registers static views from geometry, a coarse match followed by ICP, with
markers optional, so it no longer waits on hardware. Large smooth panels at a metre are
the weak case, and a virtual bench test of a curved panel with one crease comes next.
Glossy surfaces reflect the dots away from the cameras. Crossed polarizers on the light
and both lenses may replace matte spray there, which only the rig can show.

## D64. Removable dots on long smooth panels (2026-11-03)

A point-frame study tracked a camera sliding 1.1 m along a panel 2 m long at a metre, in
40 frames of about 33,000 points with 30 µm of noise, from geometry alone:

| Panel | Pose drift |
|---|---|
| Cylinder of 1.5 m radius | 1,149 mm |
| Same, with a 4° crease along the slide | 1,165 mm |
| Same, with the crease across the slide | 440 mm |
| Curved both ways, radii of 1.5 m and 3 m | 21 mm |

ICP cannot see motion along a smooth panel, and a crease fixes only motion across it,
only while it is in view. A stationary projector casting texture on the panel was set
aside, since the scanner and its operator shadow it. An inertial sensor drifts by
centimetres within seconds. A third camera tracking the room would add hardware and
software for a precision nobody has measured.

We chose removable dots for long smooth panels, as the least preparation that keeps pose
there. Paper or vinyl dots of 6 to 10 mm go about every 100 to 150 mm along the path, on
featureless spans only, and peel off afterwards. The light-off frames of D63 see them
under room light. Complex shapes such as cavities, mechanical parts and sculptures stay
geometry-only with no preparation. Detecting, triangulating and matching dots between
frames comes next.

## D65. Mesh export from the scene panel (2026-11-03)

A scene panel now has a mesh path, a hole limit, a triangle limit and an Export mesh
button. On export, the depth worker saves its live volume beside that path, and a
thread runs `slscan mesh-volume --json` through the tool command calibration uses, so
the window never waits on meshing. `sls-scan depth --mesh` runs these steps headless,
closing holes up to 10 mm, and the application gate exports a mesh that way. Four
frames of a stand-in sweep export as 30,128 triangles.

## D66. Meshing in Rust (2026-11-03)

`app/src/mesh.rs` ports surface nets, hole filling and quadric decimation, so the
application meshes without Python. Surface nets follow `mesh.surface_nets` step for step.
`sls-scan tsdf-check` meshes the NumPy volume that `slscan tsdf-vectors` writes and
compares it with NumPy's mesh: all 9,501 vertices match exactly, and all 18,228
triangles are equal, in 19.4 ms.

Decimation keeps NumPy's quadrics, boundary planes and flip check, but its heap order and
3 × 3 solve differ, so results are compared, not steps. On the 25-frame sweep's volume at
0.5 mm, 123,752 triangles decimate to 12,000 in 685 ms, where NumPy takes 27.5 s. Both
meshes lie 10.8 µm off the full mesh at the 99th percentile, at most 27.8 µm in Rust and
34.5 µm in NumPy.

The Export mesh button and `sls-scan depth --mesh` now mesh the live volume in the depth
worker and write the file from a thread. `sls-scan mesh-volume` meshes a saved volume as
`slscan mesh-volume` does, and the Python mesher stays as the reference.

## D67. Tracking from reference dots (2026-11-03)

`track.icp` takes matched dot pairs as point-to-point terms in the same least squares as
its point-to-plane terms. In `track.fuse`, each frame's dots, given as positions in its
camera frame, pair with a map of the dots seen so far. A dot pairs with the nearest map
dot within 30 mm after moving by the previous pose. Paired dots refine their map dot by
averaging, and unpaired dots join the map. Dots must lie more than twice that radius
apart, and the scanner must move less than it between frames.

The point-frame study of D64, rerun at 160 frames over the 1.1 m slide, is about 7 mm per
frame, a hand sweep at 30 fps. With dots every 120 mm and 50 µm of noise on each dot:

| Panel | Dots | Pose drift, largest | Last frame |
|---|---|---|---|
| Cylinder of 1.5 m radius | none | 1,091 mm | 1,091 mm |
| Cylinder of 1.5 m radius | every 120 mm | 0.05 mm | 0.03 mm |
| Same, crease along the slide | every 120 mm | 0.08 mm | 0.06 mm |

A test slides 0.24 m along such a panel in 24 frames, where tracking drifts 219 mm
without dots and 0.03 mm with them. Dots enter here as positions. Finding them in
light-off frames, triangulating them, and the Rust port come next.

## D68. Finding reference dots (2026-11-03)

`stereo.find_dots` places dark dots seen with the pattern light off in the left camera's
frame. `stereo.dot_centres` marks regions darker than their 61-pixel local mean by 40% of
it and centres each by its darkness. A region touching the image border or the edge of
the rectified image is dropped, since part of its dot may be missing.

Each rectified left dot pairs with the one right dot on its row whose disparity lies
within 10 mm of the depth measured there by the pattern-lit pair taken next to it. Dots
with no lit depth around them, or with no partner or several, are dropped.

We pair by lit depth because dots sit closer together than the disparity search is wide.
On the near head, depths from 0.4 to 0.6 m span about 730 px of disparity, and dots every
40 mm at 0.5 m are about 370 px apart. Pairing each row in left-to-right order slipped by
one dot whenever each camera saw a different dot at its image edge, which placed dots
about 85 mm too deep. Dots cut by an image edge moved their centres by 1 to 5 mm.

A rendered slide measured the result: near head, 30 frames over 200 mm along a flat
plate at 0.5 m tilted 8 degrees, dots of 2 mm radius every 40 mm, and a 15 mm dot radius
in `track.fuse`. Each frame gave about 10 dots and 7,200 points.

| Tracking | Pose drift, largest |
|---|---|
| Geometry only | 203 mm |
| Dots paired by row order | 86 to 144 mm |
| Dots paired by lit depth | 0.041 mm |

In the test scene, 9 dots lie within 76 µm of their true positions (median 33 µm). The
NumPy reference depth matches coarsely by semi-global matching, which leaves a strip as
wide as its disparity search unmatched: about 730 px of 1,456 for 0.4 to 0.6 m on the
near head. Dots in that strip have no lit depth and are dropped, so the test and the
slide use 0.45 to 0.55 m. Block matching, the GPU path's coarse stage, covers the width.

## D69. Registration of static views from geometry (2026-11-03)

`register.register` finds the motion that takes one view's points onto another's, each
in its own camera frame, with no markers. Both views are thinned to 5 mm voxels and given
normals turned toward their camera. Each point gets a fast point feature histogram (FPFH)
of angles between its normal and those of its neighbours within 25 mm. Points pair when
their histograms are each other's nearest. RANSAC fits a rigid motion to three pairs at a
time, from 20,000 draws whose edge lengths agree within 10%.

We score candidate motions in two stages, because pair counts alone picked wrong motions.
On views 60 degrees apart about 9% of pairs were true, and a wrong motion brought 122
pairs within 7.5 mm against 109 for the true one. The 100 motions with most such pairs
are scored by the share of all thinned points that land within 7.5 mm of the other view.
The best is refit on its pairs, and point-to-plane ICP on all points refines it, with
match distances from 15 mm down to 1.25 mm.

A test object about 0.3 m across, a base plate with two walls and four balls, was
ray-cast from cameras 0.5 m out and 0.4 m up, with 30 µm of noise per coordinate. Each
angle ran five seeds:

| Views apart | Rotation error, largest | Translation error, largest | Failures |
|---|---|---|---|
| 30 degrees | 0.0010 degrees | 6 µm | 0 of 5 |
| 60 degrees | 0.0010 degrees | 9 µm | 0 of 5 |
| 90 degrees | 0.0013 degrees | 13 µm | 0 of 5 |
| 120 degrees | 0.0011 degrees | 9 µm | 0 of 5 |
| 150 degrees | none within 1 degree | none within 1 mm | 5 of 5 |
| 180 degrees, 3 that succeeded | 0.0020 degrees | 19 µm | 2 of 5 |

Each registration took about 1.3 s. At 150 degrees the views mostly see opposite faces of
walls that have no thickness in the test object: 56% of points have normals opposite to
the other view's, so their histograms differ. Ten views around an object are 36 degrees
apart. Merging many views and the scale bar acceptance of Phase 6 come next.

## D70. Many static views around an object (2026-11-03)

`register.register_views` places views taken in order around an object in the first
view's camera frame. Each view registers to its predecessor by D69, and poses chain.
A round of refinement then runs point-to-plane ICP of every view but the first against
all others, on points thinned to a quarter of the coarse voxel. Each view's features
are computed once, and `bincount` builds the histograms, which gives the same features
2.4 times faster.

`slscan view-bench` checks Phase 6 acceptance on synthetic views. A part about a
metre across has a base, a back wall, a slanted side, a raised step and four balls. Two
scale bars of 50.8 mm spheres stand on it: one level and 401 mm long, one rising across
it and 388 mm long. Ten cameras 1.2 m out and 0.9 m up, 36 degrees apart, ray-cast
points every 1.9 mrad. Each coordinate gets 40 µm of noise, matching metre head plate RMS
in D45.

Over three seeds, with one round:

| Voxel | Pose error, largest | Scale bar errors, largest | Time |
|---|---|---|---|
| 10 mm | 0.078 mm, 0.0028 degrees | +0.003 mm, +0.005 mm | 30 s |
| 15 mm | 0.118 mm, 0.0043 degrees | +0.004 mm, +0.007 mm | 18 s |
| 20 mm | 0.241 mm, 0.0083 degrees | +0.009 mm, +0.010 mm | 12 s |

On one seed at 10 mm, largest pose error is 0.077 mm without refinement (24 s), 0.074 mm
after one round (30 s) and 0.060 mm after two (38 s). We chose one round and 10 mm as
command defaults, and its test runs 20 mm.

The level bar first read +0.05 mm on every run, with true poses too. Its spheres rested
on the base, and a 3 mm crop around each sphere took in base points near each contact.
Lifting it 35 mm clear of the base removed that bias.

The views are ray-cast points, not matched stereo pairs, so they lack matching errors at
edges and holes where the pattern misses. A view that fails to register to its
predecessor misplaces every later view. Each pair's record keeps its overlap for that
check.

## D71. Stereo views on the static-view bench (2026-11-03)

`slscan view-bench --stereo` renders each view as a pattern-lit pair for the metre head
of D45 and matches it directly at an 8 px step. Its views have matching errors at edges,
and holes wherever both cameras and the light do not all see a surface. Truth for each
view is its left camera's pose.

The command counts views placed within 10 mm of truth and exits with an error when one
is not. A scale bar check alone misses a misplaced view, because it crops each sphere at
its true position: a run with views misplaced by up to 2.8 m read both bars within
0.11 mm. Wrong matches seen so far land 0.29 m or more from truth, and calibration
offsets stay under 2 mm.

Camera placement decides whether views behind a part register. From 1.1 m out and 0.8 m
up, views 144 to 216 degrees around had 66 to 93% of their points on the flat back of
the wall, and three of nine pairs failed. A flat surface leaves motion along it free,
the smooth-panel case of D63 and D64. From 0.8 m out and 1.0 m up, the command's
default, those views see over the wall.

| Calibration | Points per view | Pose error, largest | Scale bar errors | Run time |
|---|---|---|---|---|
| True | 7,514 | 0.204 mm, 0.0109 degrees | +0.078 and -0.091 mm | 262 s |
| Recovered from twelve board poses | 7,394 | 1.665 mm, 0.0946 degrees | -0.126 and +0.008 mm | 250 s |

Matching takes nearly all of each run, and registration takes 5 s. With a recovered
calibration, pose error rises to 1.7 mm at the view opposite the first and falls to
0.2 mm at the last, and more ICP rounds leave it there. That shape seems likely to come
from a fixed offset between recovered and true camera frames, since merged views stay
consistent and both scale bars read within 0.13 mm. Phase 6 acceptance asks for 0.5 mm.

## D72. Depth frame timings on one desktop (2026-11-03)

One desktop ran each path on its GPU and on lavapipe: a Radeon RX 7900 GRE under RADV,
and llvmpipe on a Ryzen 7 7700X with 16 threads. Both drivers are Mesa 26.2.4, llvmpipe on
LLVM 23.1.1, in a release build. Kernel checks report the fastest of three runs. Stand-in
frames come from `sls-scan depth` against `slscan node-sim` at an 8 px step with 25 px
windows, timed once each, and the first frame of a run includes warm-up. A virtual
machine on the host used about one core throughout.

| Frame | Samples | RX 7900 GRE | llvmpipe, Ryzen 7 7700X |
|---|---|---|---|
| Plate, `depth-check` | 6,374 | 58.4 to 61.2 ms | 1,363 to 1,376 ms |
| Ball bar, `depth-check` | 615 | 23.0 ms | 324 ms |
| Sphere, `depth-check` | 303 | 18.6 ms | 270 ms |
| Stand-in sweep frame | 15,760 to 16,307 | 106.7 to 109.8 ms | 2,632 to 2,778 ms |
| Stand-in preview | 18,486 to 18,638 | 2.9 to 4.2 ms | 27.7 to 29.8 ms |
| Decode, 1280 × 960 | | 6.6 ms | 21.1 ms |

The plate frame keeps D38's 61 ms. Runs taken before a reboot, with the host loaded,
gave 70 to 75 ms, and their selection stage took twice as long.

A stand-in sweep frame has 2.5 times the plate's samples, and refinement takes 90% of
it, about 6 µs per sample. It reaches 9 frames per second, short of Phase 4's 15 at
1.6 MP. Meeting that rate on such frames needs fewer samples or faster refinement.

On the CPU, previews reach about 34 per second, well over Phase 4's 3, where a hosted
runner reached 7 in a debug build (D49). Full frames take 2.6 to 2.8 s, so D50's choice of
previews on software adapters stands.

## D73. Static-view registration in the application (2026-11-03)

`register.rs` ports D69 and D70 to Rust: voxel averages, normals turned toward the
camera, FPFH, mutual feature pairs, RANSAC with its overlap shortlist, ICP refinement and
the view chain. Rigid fits use Horn's unit quaternion method, an eigenvector of a 4 × 4
matrix found by Jacobi rotations, since the application has no linear algebra library.
Its RANSAC draws come from its own seeded generator, so its coarse motions differ from
NumPy's, and ICP takes both to the same poses.

`slscan register-vectors` writes four ray-cast views of D69's test object, 40 degrees
apart, with the poses NumPy finds and the true ones. `sls-scan register-check` registers
them in Rust and accepts 0.001 degrees and 5 µm against NumPy. On the desktop of D72,
Rust poses matched NumPy's to 0.000000 degrees and 0.000 µm, and both lay within
0.00092 degrees and 13.8 µm of truth. A release build took 2.9 s. `sls-scan register`
takes PLY views as `slscan register` does and writes the same merged cloud and poses
file.

Debug builds of the application now optimize its own code at level 1, because the
application gate builds in debug and the unoptimized check took 75 s. At level 1 it
takes 4.9 s, with debug assertions and overflow checks still on.

## D74. Live depth frame settings (2026-11-03)

Live depth frames now sample every 12 px and stop refining a match once a step moves it
less than 10⁻⁴ px. They sampled every 8 px with an exit at 10⁻⁵ px, and D72 measured 9
stand-in sweep frames per second against Phase 4's 15. `slscan depth-params` writes both
settings, and the WGSL kernel and the NumPy reference read the exit. Parameter files
written before keep 10⁻⁵ px. `slscan scan-stereo`, the benches and deferred processing
keep 8 px and 10⁻⁵ px.

With the same exit on both sides, the GPU still matches NumPy on the plate set: a 99th
percentile of 0.436 µm, 2.224 µm largest, none beyond 5 µm. Against NumPy at 10⁻⁵ px, the
looser exit moved that 99th percentile from 0.434 to 0.436 µm.

Stand-in sweep frames ran back to back on the desktop of D72, while a virtual machine
kept about eight of its 16 threads busy:

| Setting | Samples | Frame, mean of frames 2 to 4 | Refinement |
|---|---|---|---|
| 8 px, 10⁻⁵ px | 15,948 | 117.0 to 120.5 ms | 100.7 to 101.1 ms |
| 10 px, 10⁻⁴ px | 10,213 | 81.1 to 86.3 ms | 63.3 to 65.9 ms |
| 12 px, 10⁻⁴ px | 7,079 | 68.2 to 74.9 ms | 51.4 to 54.2 ms |

We chose 12 px, which takes 40% less time than the old setting. Scaled to D72's quiet
107 ms, it seems likely to come to about 64 ms, near 15.6 frames per second. Sweep ICP
residuals stayed at 21 µm. Point spacing grows from 0.86 to 1.29 mm at 0.5 m on the near
head, and the live TSDF of D54 averages many frames over 1 mm voxels.

Refinement kept about 25 ms that did not shrink with fewer samples under this load, so
sparser sampling will not go much further. A run on a quiet machine will settle the rate.

On 2026-11-04 another build kept that desktop's load average near 6. Eight runs at 12 px
took 54.6 to 59.8 ms per frame, 16.7 to 18.3 frames per second, so live frames clear
Phase 4's 15 per second under that load. Runs at 8 px between them took 113.4 to
122.8 ms. Refinement time followed sample count: 102 ms for 15,908 samples at 8 px and
45 ms for 7,064 at 12 px.

## D75. Dot finding in the application (2026-11-03)

`dots.rs` ports D68's dot finding to Rust. The light-off pair goes through D31's GPU
rectification. On the CPU, dark regions are found against a 61 px box mean with OpenCV's
reflect-101 border and labelled 8-connected. Regions cut by the image border or the
rectified image's edge are dropped, and the rest are centred by darkness. The valid area
comes from that undistort-rectify map at its nearest pixel, shrunk by 3 px. Each left
dot pairs with its one right partner on the row whose disparity lies within 10 mm of lit
surface depth, and `depth::triangulate` places it.

`slscan dot-vectors` writes D68's test pair, a dotted plate whose grid runs past both
cameras' views, with the lit surface and the 9 dots NumPy finds. `sls-scan dot-check`
finds those 9 within 0.079 µm on a Radeon RX 7900 GRE and 0.083 µm on llvmpipe, in 138
to 155 ms of a debug build, and accepts 2 µm. Tracking from these dots in the
application's fusion, and light-off frames over the capture protocol, come next.

## D76. Dot tracking in the application (2026-11-03)

`fuse::Fusion::add_with_dots` ports D67 to Rust. A frame's dots pair with the nearest map
dot within 30 mm after the previous pose. Three or more pairs join that frame's ICP as
point-to-point terms, three rows per pair beside its point-to-plane rows. After ICP,
paired dots refine their map dot by averaging, and unpaired dots join the map.
`Fusion::add` does the same with no dots.

`slscan dot-fusion-vectors` writes 12 point frames and dot frames of a 120 mm slide along
D67's cylindrical panel, about 10 mm per frame, with dots every 100 mm and NumPy's
tracking with dots. NumPy's poses lie within 0.019 mm of truth. `sls-scan fusion-check`
tracked them in Rust in 787 ms of a debug build and matched NumPy's poses and its 8,580
model points to 0.000 µm.

A Rust test slides 50 mm along a flat plate, which leaves the slide free without dots,
and tracks it to 1 µm with them. Light-off frames from the capture node and dot tracking
in the live view come next.

## D77. Light-off pairs in the capture protocol (2026-11-03)

A stereo request may set `dark_every`. With N, every Nth pair from the first, pair 0, N,
2N and on, is taken with the pattern light off, D63's MOSFET opening between exposures.
The first pair is dark so that dots are mapped before the head moves. Those pairs are
saved as `dark_000/left.png` and `dark_000/right.png` in place of `frame_`, keeping their
pair index, so lit and dark pairs stay in one numbered sequence.

We chose every 4th pair for the application's reference dots switch, which comes with
live dot tracking and stays off by default, since complex shapes need no dots (D64). At
30 pairs per second that gives a dot fix every 133 ms and leaves 22.5 lit pairs per
second, more than the 15 depth frames per second the GPU keeps (D74).

The stand-in puts reference dots every 30 mm on its sweep's mat when a request has
light-off pairs, and renders those pairs under room light in both its one-shot and its
paced stream. Without light-off pairs its mat stays bare, so earlier sweep figures stand.

## D78. Tracking with dots from light-off pairs (2026-11-03)

Dots are seen only in pairs taken with the pattern light off (D77), so lit frames have no
dot positions. `track.fuse` takes such a pair as a frame of None with its dots. Its pose
is the rigid fit of its dots to their map dots when three or more pair within 30 mm, and
its dots then refine or join the map. Without light-off pairs, fusion is unchanged (D55).

On D67's cylindrical panel, ICP without dots drifted about 9 mm per frame along the
panel, so three lit frames could carry the next light-off pair's dots past their match
radius. Once a light-off pair has been posed, each lit frame's ICP keeps its starting
pose along directions whose normal-equation eigenvalue lies below 10⁻⁴ of the largest,
and solves the others from the surface. On the panel, sliding along it and rolling about
its axis sat at 1.4 × 10⁻⁶ and 3.0 × 10⁻⁶ of the largest, and the next at 7.3 × 10⁻⁴.

Once two light-off pairs have been posed, a lit frame's starting pose continues the
motion between the last two at constant velocity. The first pair of a sequence is dark,
so the map is laid before the head moves. On the panel, every 4th frame light-off, with
30 µm of point noise and 50 µm of dot noise:

| Speed | Frames | Largest before the second fix | Largest after it | Median after it |
|---|---|---|---|---|
| 3 mm per frame | 100 | 9.00 mm | 0.215 mm | 0.063 mm |
| 7 mm per frame | 40 | 21.00 mm | 0.206 mm | 0.073 mm |
| 10 mm per frame | 48 | 70.00 mm | 570 mm | |

Before the second fix, lit frames stay near the first pose along the panel, so their
error is the motion so far. At 10 mm per frame the second light-off pair landed past the
30 mm radius and tracking failed, so a scan with dots should start at hand speed or
slower.

We chose this for smooth panels only. On the stand-in's sweep, a small part on a dotted
mat orbited at 2 degrees per frame, the same rule left lit frames 19 to 84 mm off, where
geometry alone kept them within 0.4 mm. That part's weakest direction lay at 5 × 10⁻⁶
of the largest, as weak as those two panel directions, and the orbit moves along it. The
application's dots switch is for smooth panels, and complex shapes stay geometry-only
(D64).

`sls-scan fusion-check` runs `fuse::Fusion::add_dark` and the lit path on 16 panel
frames and matched NumPy to 0.000 µm. A Rust test slides along a flat plate with every
4th frame light-off and keeps frames after the second fix within 1 µm.

## D79. Reference dots in the live view (2026-11-03)

A "Reference dots (smooth panels)" switch in the configuration panel asks the node for
every 4th pair with the pattern light off (D77). The depth worker keeps each light-off pair
until the next lit pair's depth frame. It then rectifies that pair on the GPU, finds its
dots against those points (D75) and poses it from them (D78) before the lit frame
joins. Lit pairs are still skipped while the worker is busy, and light-off pairs never
are, since the lit pair after one needs it.

`sls-scan depth --dark-every 4` runs the same path without the window. It fails when a
light-off pair after the first finds fewer than 3 dots.

A dot now pairs with a map dot within 0.4 of the map's median dot spacing, at most 30 mm.
The near head sees about 157 × 117 mm at 0.5 m, so its dots need to be about 35 mm apart
for a dozen in each light-off pair. Dots between map dots, such as a grid of 35 mm moved
18 mm, no longer pair with a wrong neighbour.

For light-off requests the stand-in slides 2.5 mm per pair along a flat 0.6 × 0.3 m panel
at 0.5 m with dots about 35 mm apart, in place of D77's dotted mat. Each light-off pair
found 12 to 15 dots, and light-off poses followed the slide, reading 48.4 mm where the
head moved 50 mm.

That 3% shortfall comes from a calibration the application recovers from the
stand-in's board poses, which fits a k2 of 1.77 and 3.65 where the cameras have none.
Two light-off pairs 10 mm apart measured 9.95 mm with the true calibration and 9.73 mm
with the recovered one. A pose from dots alone fits a small patch 0.5 m away, so a
sub-pixel bias across the field becomes about 1 mrad of rotation and 0.5 mm of
translation. Fixing k2 and k3 for narrow lenses would be a change of its own.

## D80. k2 for narrow lenses (2026-11-04)

Calibration fits k2 only for a camera or projector whose half-diagonal field is at least
0.3 in tangent, about 17°. A narrower device keeps k2 and k3 at 0 and fits k1 with its
tangential terms.

A unit of k2 moves an image corner by R⁵ f pixels, for that tangent R and a focal length
f in pixels. Behind a 16 mm lens a Global Shutter camera has R = 0.196, so a unit of k2
moves its corner 1.3 px, and k1 with focal length absorbs all but 0.29 px of it. Board
views stop short of those corners, and it seems likely that they leave k2 to trade against
k1 and focal length.

On twelve stand-in poses the near head fitted k2 of 1.77 and 3.65 where its cameras have
none. Three jittered sessions gave -2.27 to +1.12.

We triangulated noiseless points over each whole image at 0.4 to 0.6 m with each
recovered calibration, then fitted them rigidly to truth. Over four sessions, largest
errors fell from 0.18 to 0.67 mm with k2 fitted to 0.10 to 0.14 mm with k2 fixed. On
6 mm metre-head lenses (R = 0.52, 68 px per unit of k2), a fixed k2 did worse in three
sessions of four, so wide lenses keep fitting it. A 12 mm lens (R = 0.26) falls under
this rule, though k1 leaves 0.9 px per unit of k2 at its corners.

Synthetic cameras have no distortion, so this bench favours a fixed k2. A real 16 mm lens
with a k2 of 0.3 would leave under 0.1 px at its corners.

Through the application on the stand-in, two light-off pairs 10 mm apart now measure
9.83 mm from their dots, against 9.73 mm before and 9.95 mm with true calibration. Over
a 24-pair slide, the last light-off pose reads 49.5 mm for 50 mm, where D79 read
48.4 mm. Sweep tracking with recovered calibration drifts 299 µm where it drifted
344 µm, and repeat runs agree within 0.0001°.

Sphere fits to exact surface points of the near head's ball bar, triangulated with
either calibration, differ by at most 4 µm over six sessions. From rendered pairs,
affine windows now measure that bar 25 µm shorter, 80.379 mm against 80.404 mm for
80.424 mm. Shifting both rectified images by a fraction of a pixel moves this result
between -9 and -38 µm with true calibration. With ours it moves between -31 and -46 µm.
We have not found where matching takes up the rest. Pipeline tests now allow 50 µm on that bar,
where they allowed 20 µm.

Later on 2026-11-04 we rectified that capture with cubic in place of linear
interpolation. True calibration then measured the bar 36.4 µm short, and ours 19.7 µm
short, where linear interpolation gave 11.9 and 42.1 µm. It seems likely that this shift
comes from resampling, not calibration, and that one capture's bar spacing at the near
head moves about 25 µm with how the rectified images are resampled.

## D81. Datum spheres found without seeds (2026-11-04)

Datums now define the object's frame without a picked point. When a datum file gives no
seeds, `slscan report` finds spheres of each datum radius in the cloud.

Points averaged over voxels of a tenth of that radius each vote for the two centres at
that radius along their normal. Cells of a quarter radius with the most votes, summed
over their neighbours, seed sphere fits over shrinking shells. A fit counts when its
radius is within 10% of the datum's, its form RMS is under 5% of it, and its points
spread far enough around the centre that a flat patch cannot pass.

Found spheres are matched to datums by their distances. Three found spheres whose
distances and radii agree with the measured triangle of largest area give a rigid motion,
which must place every datum within 3 mm of a found sphere of its radius. We refuse a
layout that two such motions fit, such as an equilateral triangle, because its frame is
ambiguous. `slscan report --aligned` writes the cloud or mesh in the object's frame.

`slscan view-bench --datums` writes the metre part's four scale bar spheres as a datum
file without seeds, then finds them in the merged views among the part's balls of 50 to
100 mm radius. With ray-cast views their residuals were 2 to 5 µm. With stereo views and a
recovered calibration they were 0.095 to 0.198 mm, where that run's scale bars read
-0.126 and +0.008 mm.

## D82. Phase-shift triangulation in the application (2026-11-04)

Phase 11 starts with static phase-shift scans in the application, for edge-sharp views
with a calibrated projector (D25). `app/src/scan.rs` triangulates decoded columns on the
CPU in double precision, as `slscan/triangulate.py` does. Each camera ray meets its
column's plane, taken from the undistorted projector point at the row where the previous
estimate lands, over three iterations. Undistortion repeats the five fixed iterations of
`cv2.undistortPoints`, so points match NumPy to rounding.

`slscan scan-params` writes a calibration and the decode settings as JSON for it.
`slscan scan-vectors` writes a plate scan by the reference rig with a calibration
recovered from board views, whose camera and projector fit k2 of -0.11 and -0.40.
`sls-scan scan-check` compares the application with NumPy on that scan. From NumPy's
columns, 521,793 points matched within 0.1 nm in 23 to 39 ms on 16 threads. From the
GPU decode they matched within 0.154 µm, with every pixel kept by both, on the Radeon and
on llvmpipe.

## D83. Phase-shift scans from the application (2026-11-04)

`sls-scan scan` asks a node for a scan set at the pitch and steps that
`slscan scan-params` wrote, decodes it on the GPU and triangulates it into a PLY file.
With `--save` it also writes the frames in the layout `slscan scan` reads. On the
stand-in's plate at 1 m, the application and `slscan scan` gave the same 521,793 points
from the same frames, within 0.12 µm, which float vertices in PLY explain. Decode took
21 to 32 ms and triangulation 39 to 43 ms on the Radeon and 16 threads.

That plate's flatness follows the calibration. With six stand-in board poses solved
through the application it measured 0.471 mm RMS, and with twelve 0.170 mm, against
the Phase 2 limit of 0.2 mm. The gate keeps six poses for time and checks this scan
against a 1 mm limit, and `scan-check` compares its points with NumPy.

In the window, a scan set run with a scan parameter file decodes and triangulates on a
thread of its own once its last frame arrives. Its points replace the scene's model, and
"Save points" writes them as PLY. The scene draws at most 20,000 points, every so many
of a larger cloud, since a scan at 1280 × 960 has about half a million. No automated test
drives the window. The functions it calls are those of `sls-scan scan` and its gate run.

## D84. k2 where the board reaches (2026-11-04)

Calibration now fits k2 only when a unit of it moves the board point seen farthest from
the principal point by 10 px or more. That shift is r⁵ f for the point's normalized
radius r, the measure of D80 taken at the board's reach in place of the image corner.
This replaces D80's rule of a 0.3 tangent for the half-diagonal field.

Board views reach 0.45 of the reference rig's half-diagonal at 1 m, where a unit of k2
moves the farthest point 1.2 px in the camera and 0.4 px in the projector. Twelve poses
fitted k2 of about -0.1 and -0.4 there, where both devices have none, and the plate
measured 0.155 to 0.196 mm RMS over six seeds against 0.051 mm with true calibration.
With k2 at 0 it measured 0.085 to 0.092 mm.

Other heads keep their D80 outcome. The metre head's board moves its farthest points
13.9 and 14.7 px per unit of k2, so it still fits k2, and its bench figures are
unchanged. The near head, the small-volume head and the 12 mm lens of D80 stay at
1.12 px or under. An 8 mm lens at 1 m, at 4.9 and 6.8 px, now keeps k2 at 0, which
D80's bench favoured there. A real
calibration whose board reaches a wide lens's corners fits k2.

On a stand-in that had served one scan set and one calibration set, the application's
camera and projector calibration now measures the plate at 0.126 mm RMS with six poses
and 0.092 mm with twelve. The gate takes twelve poses, 13 s of calibration, and checks
the scan against the Phase 2 limit of 0.2 mm in place of D83's 1 mm. In the full gate
run the plate measured 0.088 mm.

## D85. A window scan in the gate (2026-11-04)

The application gate now drives a phase-shift scan through the window. A test in
`app/src/ui.rs` builds the application in `egui_kittest`, the egui project's test harness,
added as a development dependency only. It points the window at the stand-in and at the
gate's scan parameters, clicks "Run sequence" and steps frames until the scan's points
arrive. It then clicks "Save points" and requires the saved points to equal those of
`sls-scan scan` on the same plate. It is skipped unless `SLS_WINDOW_NODE` names a node,
so `cargo test` alone does not need the stand-in.

On the stand-in, the window saved 521,793 points equal to the headless scan's, and the
test took 0.7 s after calibration. This closes the gap D83 left, where no test drove
the window.

## D86. Exposures for phase-shift scans (2026-11-04)

Decode now drops a pixel when any of its phase samples reaches 255, in NumPy and in the
GPU kernel. A clipped sinusoid keeps its modulation and decodes to a wrong column: on a
strip of albedo 2.5 at the nominal exposure every pixel decoded, and its points lay
0.724 mm RMS from their plane. `slscan decode-vectors --exposure 4` renders the plate
saturated, and the gate compares the kernel with NumPy on it. The earlier kernel
disagreed with NumPy on 40% of its pixels there.

A scan request may now list exposures relative to the node's nominal one. The node takes
each pattern at every exposure before the next pattern, so the projector changes pattern
once per pattern, and the frame header's exposure field gives the index.
`decode.decode_exposures` gives each pixel the exposure with the largest phase modulation
among those where it decoded. Modulation grows with exposure, so this is the longest
exposure that does not saturate there, and it needs no exposure values. `slscan scan`
decodes a view of `exposure_` directories this way.

`slscan hdr-bench` scans the acceptance plate as dark, middle and bright strips. One
exposure decodes only the middle strip. Exposures of 0.25, 1, 4 and 16 decode all
three, the dark strip at 68 µm RMS, and the whole plate at 61 µm RMS against 51 µm for
the middle strip alone. The application takes these scans next.

## D87. Exposures in the application (2026-11-04)

The application now takes the scans of D86. `slscan scan-params --exposures` lists
exposures in the scan parameter file, and `sls-scan scan` and the window request them
from the node, as `sls-scan capture --exposures` does. Each exposure decodes on the GPU.
The CPU then gives each pixel the exposure with the largest phase modulation among those
where it decoded, in double precision as NumPy does.

On a strip scan at exposures of 0.25, 1, 4 and 16, `sls-scan scan-check` placed 521,793
points from the GPU decode within 0.154 µm of NumPy's, with every pixel kept by both, on
the Radeon and on llvmpipe. From a stand-in serving the strip plate, `sls-scan scan`
decoded the four exposures in 86 ms and gave the points of `slscan scan` on its saved
frames, within 0.12 µm.

With a twelve-pose calibration from the stand-in, that plate measured 0.098 mm RMS over
all three strips. One exposure gave 172,569 points on the middle strip alone, at
0.060 mm. The gate runs the vector check and the stand-in scan.

## D88. Color from the projector's fields (2026-11-05)

A scan set may end with the camera under the projector's black field and its full red,
green and blue fields. The capture node names them `color/black` onward, the stand-in
renders them at each surface's tint, and `slscan capture --color` asks for them.

A mono camera sees each field as the surface's reflectance in that channel times the
projector's output there, plus room light. Subtracting the black field removes room
light. A white card's median response per channel, taken once per session as
`slscan scan --white`, divides the rest, so a white surface reads 255 in every channel
and a colored one its reflectance against white. Without a card the point colors are the
raw responses in grey levels. `geometry.write_ply` writes the bytes per vertex, so any
PLY reader shows them.

The stand-in's plate has red, green and blue strips, each of albedo 0.8 with a quarter
of the other channels. Its points read within 2 grey levels of the truth, 204 in a
strip's own channel and 51 in the others. A real projector's channels overlap and its
LED output differs per channel, which the white card absorbs. A camera with a color
filter array would see each field through it, so this rests on the mono Global Shutter
camera of D25.

## D89. Colors in the application (2026-11-05)

The application takes the colors of D88. `slscan scan-params --color` sets the scan
parameters to end scans with the projector's fields, and `--white` reads a white card's
fields and writes its response per channel into them. `sls-scan scan` and the window
request the fields, color each point from them as `colors.point_colors` does, rounding
half to even as NumPy does, and write red, green and blue bytes per PLY vertex. The scene
draws a colored scan in its colors.

`slscan scan-vectors --color` writes the fields and NumPy's colors beside a scan, and
`sls-scan scan-check` compares the application's colors byte for byte. On the plate of
red, green and blue strips, none of 521,793 points differed, on the Radeon and on
llvmpipe.

From stand-ins serving that plate and a white card, the application captured the card
with `sls-scan capture --color true` and wrote its balance into the scan parameters. Its
scan then landed within 0.12 µm of `slscan scan --white` on the same frames, with no
color byte differing. Each strip read 204 in its own channel and 51 in the others. The
gate colors its strip scan and checks the color vectors.

## D90. A Haskell reference of the decode (2026-11-05)

`ref/` contains a second implementation of the decode of D16 and D86 in Haskell, as the
reference core of D11 proposed. It decodes each pixel from the Gray code and phase
stacks with NumPy's arithmetic in the same order: sums from the first sample, the
wrapped phase as `np.mod` gives it, `hypot` from the C library, and rounding half to
even. `sls-ref decode-check` compares its columns and masks with the vectors of
`slscan decode-vectors`, which the GPU kernel already matches (D22).

On the plate, the sphere and the strip plate, whose bright strip saturates and whose
dark strip lacks modulation, no mask differed and every shared column was equal to the
last bit of a float. Truncating in place of rounding made 259,279 masks differ, and
dropping the saturation rule made 151,187 differ.

It builds from GHC's own libraries, `base`, `bytestring`, `array` and `filepath`, so any
GHC installation runs it. `ref.do` builds it and runs the check in `verify.do`, and
skips without GHC. Continuous integration runs it in the Tests job, on the GHC of
GitHub's runner image. Triangulation is the next part of the reference.

## D91. Triangulation in the Haskell reference (2026-11-05)

`ref/Triangulate.hs` triangulates as `slscan/triangulate.py` does, with the undistortion
of `cv2.undistortPoints` in its five fixed iterations and the projection of
`cv2.projectPoints`. `sls-ref scan-check` meets each masked pixel's ray with its
projector column's plane and compares the points with those of `slscan scan-vectors`,
whose recovered calibration fits k2 of -0.11 in the camera and -0.40 in the projector.
`ref/Json.hs` reads the nested scan parameters with nothing but `base`.

Of 521,793 points, the pixels kept were the same, and in two runs 26 and 43 points
differed at all, by at most 4.7 × 10⁻⁷ nm. NumPy forms the plane products through BLAS, which seems
likely to sum them in another order, so the check allows 10 nm. Skipping the projector's
undistortion moved points by up to 0.82 mm, and one refinement of the plane in place of
three by up to 0.43 mm. `ref.do` runs this check after the decode checks.

## D92. A third camera against occlusion, measured (2026-11-05)

`slscan occlusion-bench` measures how much of a slot the near head can see in one view.
A pair of cameras counts a surface point when both cameras and the pattern light see it,
inside their images and within 75° of its normal. `slscan/occlusion.py` ray-casts this
against the scene with no matching, so it measures geometry alone. The slot is 20 mm
wide, cut across the baseline into a flat top at 0.5 m.

| Slot depth | Floor, pair alone | Floor, third camera 0.1 m above the light | Floor, third camera 0.05 m beside it |
|---|---|---|---|
| 10 mm | 80.0% | 100.0% | 95.0% |
| 20 mm | 65.0% | 100.0% | 92.5% |
| 40 mm | 30.0% | 100.0% | 85.0% |

The pair loses the floor beside each wall, where one camera cannot see past it. A third
camera above or below the light sees the whole floor with the left or right camera,
over a baseline of 0.141 m against 0.2 m. Depth noise scales with 1/b, so those pairs
would read 1.41 times noisier. Each wall stays visible to one camera only, so all
surfaces gain less: 66.4% to 71.4% at 20 mm deep.

A slot along the baseline loses none of its floor to the pair at these depths. Turning
the head a quarter turn therefore reaches the same floor in a second view, and a
continuous sweep does so as it moves. This is a measurement for the rig of Phase 8, not
a choice of it. A third camera adds a capture channel and matching for two more pairs
per frame.

## D93. A Futhark decode, measured against WGSL (2026-11-06)

`fut/decode.fut` decodes as `slscan/decode.py` does, in a module over the float type, so
one source gives entry points in double and single precision. `slscan/futhark.py` builds
it into a shared library with one Futhark backend and calls it through ctypes, over the
C interface an application would link. `slscan futhark-check` compares its columns and
masks with vectors from `slscan decode-vectors` and times it.

In double precision it matched NumPy to the last bit on plate, sphere and strip plate,
on C, multicore, OpenCL and HIP backends. In single precision no mask differed, and
columns differed by at most 6.1 × 10⁻⁵ px, as WGSL's do. Truncating in place of
rounding made 259,279 masks differ, and dropping the saturation rule 151,187, as in
Haskell (D90).

Times below are for a 1280 × 960 plate scan of 20 Gray code and 4 phase images
(29.5 MB), each the fastest of three runs. We measured on a Radeon RX 7900 GRE over
PCIe 4.0 x16 with Mesa 26.2.4 and ROCm 7.2.4, and a Ryzen 7 7700X. A virtual machine
used about seven of its 16 threads throughout, which slows CPU backends and host work
on every path. `sls-scan decode-check` times WGSL, and `slscan futhark-check` Futhark.

| Decode | Precision | With upload and readback | Kernel alone |
|---|---|---|---|
| WGSL on Vulkan | f32 | 7.7 ms | 0.13 ms |
| Futhark, OpenCL | f32 | 2.2 ms | 0.14 ms |
| Futhark, HIP | f32 | 2.0 ms | 0.14 ms |
| Futhark, OpenCL | f64 | 3.9 ms | 1.74 ms |
| Futhark, HIP | f64 | 3.4 ms | 1.66 ms |
| Futhark, multicore | f32 | 17.7 ms | 13.1 ms |
| Futhark, multicore | f64 | 21.5 ms | 17.6 ms |
| Futhark, C | f64 | 140 ms | 136 ms |
| NumPy | f64 | 108 ms | |

WGSL's and Futhark's single-precision kernels run equally fast. Of WGSL's 7.7 ms, about 7.5 ms is host work in
`app/src/gpu.rs`, which pads both stacks into new 32-bit word buffers and creates
every buffer per call. Futhark keeps device memory between calls and takes bytes as
they are. Double precision costs this card twelve times its kernel time, which seems
likely to follow from RDNA3's low double-precision rate. Futhark's C backend runs one
thread and trails NumPy, whose array operations use vector instructions.

For decode this gives no reason to leave WGSL (D11). Its kernel is as fast, and wgpu
reaches Vulkan on AMD and NVIDIA cards and lavapipe with no toolchain beyond Rust.
Reusing buffers across calls would likely recover most of those 7.5 ms. Futhark stays
as a second check: `fut.do` runs its C and multicore backends in `verify.do` and in
continuous integration, and `slscan futhark-check --backend opencl` checks a card. It
costs a C compiler and a library build per backend, and HIP needs ROCm's include and
library paths when ROCm sits outside the compiler's defaults, as in `/opt/rocm`.

## D94. Decode buffers kept between calls (2026-11-06)

D93 found the WGSL decode spending about 7.5 ms of 7.7 ms on host work, with a kernel
of 0.13 ms. `gpu::Decoder` now keeps its parameter, stack, output and readback buffers
and its bind group from one decode to the next, while stack sizes stay the same, and
allocates them again when sizes change. Stacks go straight into their buffers through
`Queue::write_buffer`, with no padded copy. A scan at several exposures decodes each
exposure into one set of buffers.

On a 1280 × 960 plate scan, with a Radeon RX 7900 GRE and a load average near 5,
`sls-scan decode-check` took 6.6 to 7.2 ms before and 3.6 to 4.6 ms after, over five
alternating runs of each build. Every run matched NumPy, and the kernel stayed at
0.12 ms. Mapping both readback buffers at once and converting from mapped memory made
no difference beyond run-to-run spread, so we left it out. Most of the remaining time
seems likely to be upload: 29.5 MB copied into wgpu's staging memory and then over PCIe.

A decoder of this size keeps 49 MB of buffers between calls: 29.5 MB of stacks and two
pairs of 4.9 MB output and readback buffers.

## D95. Decode time by stage (2026-11-09)

Stage timers in a temporary build split one decode of D93's plate scan, which took 3.6
to 4.6 ms after D94.

| Stage | Time |
|---|---|
| Copy of both stacks into wgpu's staging memory | 1.4 ms |
| GPU: staging copies, kernel, copies to readback buffers | 0.85 ms |
| Copies of both readback buffers into host vectors | 0.4 ms |
| Conversion of those bytes to floats and flags | 1.7 to 2.5 ms |

Conversion ran as a loop that did not vectorize, into new result vectors whose pages
fault on first touch. `gpu::floats` and `gpu::flags` now convert whole words through
`as_chunks`, and `gpu::read_with` runs them on a mapped readback buffer with no host
copy between. Coarse matching, refinement and TSDF readback use these two functions too.

`sls-scan decode-check` took 4.3 to 5.1 ms before and 3.3 to 4.1 ms after, over six
alternating runs on a Radeon RX 7900 GRE with a load average near 6. About 1 ms of each
run is still first touch of new result vectors, since this check keeps one result while
it decodes again. With each result released first, as `scan::decode_scan` does between
exposures, runs took 2.5 to 3.0 ms.

Uploading is what remains on the host. Removing that copy means writing frames into
staging memory as they arrive, which changes the decoder's interface to save about
1.4 ms in a scan whose capture takes seconds, so we left it.

## D96. Sample selection in one pass over rows (2026-11-09)

`depth::select` chose samples through two summed-area tables of a whole image, built on
one thread with two divisions per pixel (D36). It now walks down an image once. It keeps
five rows of 5-pixel sums, marks each pixel that is unlit or unmatched, and combines each
row of marks into one band per sample row. A sample is kept when its band has no mark
across its window. Samples and guesses equal those from summed-area tables on our stereo
vectors, and a test compares them with a scan of every window pixel.

Times are from a Radeon RX 7900 GRE and a Ryzen 7 7700X at a load average of 4 to 6,
with builds alternating.

| Frame | Before | After |
|---|---|---|
| Sphere vectors, `sls-scan depth-check`, six runs | 22.1 to 23.3 ms, selection 9.9 to 10.8 ms | 11.9 to 12.9 ms, selection 0.5 to 0.6 ms |
| Stand-in sweep at live settings, mean of frames 2 to 4, seven runs | 62.9 to 66.0 ms, refinement 47.5 to 50.0 ms | 40.4 to 41.7 ms, refinement 35.3 to 36.3 ms |

Live frames go from about 15 to 24 per second. Refinement did not change, yet it ran
about 13 ms faster. A 10 ms pause in place of old selection, sleeping or spinning,
brought refinement back to 47.8 to 49.9 ms. A host gap of that length before a GPU stage
therefore costs more than its own time. It seems likely that this card drops to a lower
power state while it waits, which we did not confirm from its driver.

Fusion on the CPU sits between frames in that way, at 27 to 43 ms per frame in these
runs. Each frame's first GPU stages may pay for it likewise. We have not measured that.

## D97. Two cameras on the rig (2026-11-09)

Phase 8's rig stays at two cameras and one pattern light for now. D92 measured what a
third camera above that light would add: a whole slot floor across the baseline, where
our pair sees 30% at 40 mm deep. A second view with the head turned a quarter turn
reaches that floor with two cameras, and so does a continuous sweep. A third camera
would add a capture channel, its calibration, and matching for two more pairs per frame.

`slscan occlusion-bench` stays, so a part that a second view cannot reach can be
measured before this is decided again.

## D98. One device for GPU unit tests (2026-11-09)

Unit tests of our decoder, rectifier, block matcher and refiner each opened a device,
and `cargo test` runs them on parallel threads. GitHub moved our runs to its
ubuntu-26.04 image on 2026-11-09, and that test binary then ended with a segmentation
fault in some runs. Its backtrace is in the Vulkan loader, `libvulkan1` 1.4.341, under
`vkSetDebugUtilsObjectNameEXT`, which wgpu calls to name objects in debug builds, while
another thread opens or closes a device.

An Ubuntu 26.04 container on four CPUs, with that image's Mesa 26.0.8 and LLVM 21.1.8
packages, reproduced it.

| Unit tests of main in that container | Runs with a fault |
|---|---|
| Parallel threads | 9 of 40 |
| Parallel threads, object naming off with `WGPU_DEBUG=0` | 0 of 30 |
| Parallel threads, loader 1.4.363 in place of 1.4.341 | 0 of 30 |
| One thread | 0 of 30 |
| Parallel threads, release build | 0 of 40 |
| Parallel threads, release build with `WGPU_DEBUG=1` | 15 of 40 |
| Parallel threads, one shared device | 0 of 40 |

On GitHub's runner, main's tests faulted in 7 of 36 parallel runs and in none of 12 on
one thread. With one shared device they faulted in none of 24.

GPU tests now take one device from `gpu::test_device`, opened once, so no test opens a
device while another names an object. Our application's kernels share a device in that
way already (D37).

A debug build of our application can still meet this fault on that loader, when one
thread opens a device while another uses one, as a scan started during live depth frames
does. Release builds leave object naming off.

