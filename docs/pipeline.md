# Pipeline

Each stage is one module in `slscan/` and one command in `slscan/cli.py`, with files
between stages so any stage can be rerun from its input. All data lives under `data/`,
which is not tracked. Active stereo is the main path (`DESIGN.md` D25). The phase-shift
path with a calibrated projector is the option for edge-sharp static scans.

## Stages

| Stage | Module | Input | Output |
|---|---|---|---|
| Stereo calibration | `calibrate.py` | board images from both cameras | `stereo.yaml`: both cameras' matrices and distortion, left-to-right rotation and translation, board scale |
| Stereo matching | `stereo.py` | left and right images, `stereo.yaml` | a point cloud per frame (PLY) in the left camera's frame |
| Track and fuse | `track.py` | depth frames in order | a pose per frame and the fused model |
| Patterns (phase shift) | `patterns.py` | projector width and height, phase pitch and steps | pattern image stack and decode tables |
| Projector calibration | `calibrate.py` | board images, projected-code images on the board | `calibration.yaml`: camera matrix, distortion, projector matrix, extrinsics, board scale |
| Capture | `capture.py` | pattern stack, exposure list | one directory per view with the images and a manifest |
| Decode (phase shift) | `decode.py` | a view directory | per-pixel projector column, confidence mask |
| Triangulate (phase shift) | `triangulate.py` | decoded view, `calibration.yaml` | a point cloud per view (PLY) |
| Register | `register.py` | view clouds and target detections | view poses, a merged cloud, scale bar report |
| Mesh | `mesh.py` | merged cloud, datum points | a mesh (PLY, OBJ, STL) and the residual report |

## Active stereo

A stereo calibration capture is one directory per board pose (`pose_000` onward), each
with `left.png` and `right.png` taken at the same instant. `slscan calibrate-stereo`
calibrates each camera from every pose in which it sees the board, and the pair from
the corners both see, then writes `stereo.yaml`.

A stereo view is a directory with `left.png` and `right.png` under the pattern light.
`slscan scan-stereo <view>` rectifies both, takes a coarse disparity from semi-global
matching at quarter resolution, and refines a sample every 8 pixels by Lucas-Kanade and
an affine window fitted with ECC. A sample is kept when its window lies on lit, matched
surface, correlates at 0.95 or better, and lands within 3 pixels of its guess.

A second-order window of 25 pixels, fitted by Gauss-Newton, then follows the surface's
curvature (`DESIGN.md` D28). `--order 1` stops at the affine window, which is faster and
reads curved surfaces slightly inside. `--near` and `--far` bound the depth search, and
`--step` and `--window` set density and the trade between noise and curvature bias
(`docs/accuracy.md`). Triangulated points with no other point within three sample
spacings are dropped (D56).

`stereo.direct_matches` matches by Gauss-Newton alone: an affine window from the coarse
disparity, kept on its correlation, then a second-order window (`DESIGN.md` D32). Its
coarse disparity comes from block matching by normalized cross-correlation at quarter
resolution (D34). It is the reference for the GPU path, which has no Lucas-Kanade or ECC
stage.

## Track and fuse

`track.fuse` aligns each depth frame to the model by point-to-plane ICP, starting from
the previous pose, then adds its points to the model as averages over 0.5 mm voxels
(`DESIGN.md` D26, D55). A captured sweep is a directory of `frame_000` onward, each
with `left.png` and `right.png` (D27), and `slscan fuse <sweep>` matches, tracks and fuses
it into one PLY file.

Given reference dots per frame, `track.fuse` pairs them with a map of dots seen so far
and adds point-to-point terms to ICP, which keeps tracking along smooth panels (D67).
This serves smooth panels, and complex shapes stay geometry-only.

A pair taken with the pattern light off is posed from its dots alone. After a quick
turn a search over shifts finds its dots up to two spacings off, given dots off a
regular grid (D110). Lit frames between such pairs keep a start from those poses along
directions the surface leaves nearly free (D78, D107). ICP turns such a frame about its own points, so a tilt it
corrects does not slide that frame along a panel (D108). That start follows capture
time, which keeps it right when a live view leaves lit pairs out (D101).

Each map dot keeps its place along the surface from light-off pairs and takes its
height from the fused model once that reaches it, so light-off poses do not tilt
further along a long panel (D111).

`stereo.find_dots` finds the dots in a pair taken with the pattern light off. It pairs
left and right dots by the depth that the pattern-lit pair next to it measured (D68). The
application's ports, `dots::find_dots` and `fuse::Fusion::add_with_dots`, are checked
against them by `sls-scan dot-check` and `sls-scan fusion-check` (D75, D76).

`fuse::Fusion` aligns each frame to model points near it, and updates its model and
normals only where that frame lands. Results equal those from work on every model point
(D100).

`slscan sweep-bench` runs a synthetic sweep through matching, tracking and fusion and
compares poses and artifacts with the truth, and `--loop` sweeps a closed loop. The application runs a Rust port of both,
checked against them by `sls-scan fusion-check` (D42). `slscan fuse` and `slscan sweep-bench`
match with affine windows of 21 pixels by default, for speed, and take `--order 2` for
second-order windows.

`tsdf.fuse` folds tracked frames into a truncated signed distance volume (D51). Each
frame becomes a depth image at its sample grid, and every voxel averages its distance to
each frame's surface along the camera axis. `slscan sweep-bench --tsdf` measures the
volume's surface in place of the fused points, with `--voxel-mm` and `--truncation-mm`.

## Patterns

Gray code columns: for a projector of width W, ceil(log2 W) bit planes, each shown
with its inverse. Phase shift: three or four sinusoids of pitch P projector pixels,
shifted by 2π/N. Gray code gives the period index and phase gives the position inside
it. Row patterns are optional: triangulation needs only the column, and rows serve
projector calibration.

## Calibration

1. Camera intrinsics: `cv2.aruco` ChArUco detection over 20 to 30 board poses, then
   `cv2.calibrateCamera`. Reprojection error under 0.3 px.
2. Projector as an inverse camera: for each board pose, the Gray code stack projected
   onto the board gives, at every detected corner, the projector pixel that lit it
   (local homography from surrounding decoded pixels). Corners and their projector
   pixels calibrate the projector with `cv2.calibrateCamera`. Reprojection error under
   0.5 px.
3. Stereo: `cv2.stereoCalibrate` on the shared corners gives rotation and translation
   between camera and projector.
4. Scale: board square size, measured with calipers on the print, not taken from the
   print job.

Distortion is radial and tangential with k3 fixed, for both devices. k2 stays at 0
unless the board points reach far enough from the principal point that a unit of k2
moves the farthest of them 10 px or more (D80, D84). Stereo calibration follows the
same rule for each camera. Column and row
Gray codes are both projected, because each corner needs a projector row as well as a
column.

A calibration capture is one directory per board pose (`pose_000`, `pose_001` and so
on). Each contains the camera image of every frame that `slscan calibration-patterns`
writes, under the same file name: `white.png`, `columns_000.png` onward and
`rows_000.png` onward. `slscan calibrate <directory>` reads them and writes
`calibration.yaml`:

| Field | Contents |
|---|---|
| `board` | squares across and down, square and marker size in metres, marker dictionary |
| `camera`, `projector` | image size, 3 × 3 matrix, distortion (k1, k2, p1, p2, k3), reprojection error in pixels |
| `stereo` | rotation and translation in metres from camera frame to projector frame, reprojection error |
| `views` | poses used |

The application runs these steps from its configuration panel (`DESIGN.md` D21).
Capture pose runs a calibration sequence on the node and writes the next `pose_NNN`
directory of a session. `slscan detect` reports the corners found in it, and Solve runs
`slscan calibrate --json` and shows each error against its limit. `sls-scan calibrate`
does the same without a window.

`slscan calibration-diff a.yaml b.yaml` compares two calibrations against the
repeatability limits of Phases 1 and 8: focal length within 0.1 % and baseline within
0.2 mm. It takes two stereo calibrations as well. `slscan calibration-repeat`
calibrates the synthetic stereo head from independent sessions of hand-placed board
poses and checks them against the same limits (`DESIGN.md` D44).

## Capture

The projector is the second display. A fullscreen window shows each pattern, the camera
captures after a settle delay, and a manifest records pattern index, exposure and
timestamp per image. Two or three exposures per pattern cover a view with both dark and
light surfaces. The decode takes the best-exposed sample per pixel.

With the capture node of `DESIGN.md` D13, the application sends a sequence request and
receives frames over the capture protocol (`docs/capture-protocol.md`). `slscan capture`
saves one sequence in the layouts above.

## Decode

`decode.py` binarizes each Gray code plane against its inverse and decodes the bits into
a column index per pixel. Phase shift images give a wrapped phase and a modulation per
pixel. The float column is the phase column nearest the Gray code column. A pixel is
masked when a pair lacks contrast, a phase sample saturates at 255, the sinusoid lacks
modulation, or the two columns disagree by more than a pixel (`DESIGN.md` D16, D86).

A view at several exposures decodes each exposure and gives each pixel the one with the
largest phase modulation among those where it decoded, which is the longest exposure
that does not saturate there (D86).

`ref/` decodes again in Haskell with NumPy's arithmetic in the same order, and
`ref.do` compares its columns and masks with NumPy's on synthetic scans (D90). It also
triangulates the plate scan of `slscan scan-vectors` and compares its points (D91).

`fut/decode.fut` decodes again in Futhark, in double and single precision from one
source. `slscan futhark-check` builds it with a Futhark backend (C, multicore, OpenCL or
HIP), calls it through ctypes and compares its columns and masks with NumPy's. It times
each decode with and without upload and readback, as `sls-scan decode-check` does for
the WGSL kernel. `fut.do` runs it on C and multicore backends (D93).

## Color

`colors.py` colors points from a mono camera and the projector's black, red, green and
blue fields (`DESIGN.md` D88). Each field less the black one is the surface's response
in that channel without room light. A white card's median response per channel is the
white balance, and `slscan scan --white` divides by it so a white surface reads 255 in
every channel. `geometry.write_ply` keeps red, green and blue bytes per vertex, which
`read_ply_colors` reads back.

`slscan scan-params --color --white` puts the fields and the white balance into the
application's scan parameters, and `sls-scan scan` and the window then color their
points the same way (D89).

## Triangulate

Each pixel's ray comes from the camera matrix and distortion, each column's plane from
the projector matrix and extrinsics, and their intersection is the point. Projector
distortion is applied at the row where the previous estimate lands, over three
iterations. Points with
low confidence, grazing angles or a column outside the projector are dropped. Output is
a PLY per view in the camera frame.

## Register

Each view's pose comes from geometry (`DESIGN.md` D63, D69). `register.register` thins
two views to a voxel grid, turns their normals toward each camera, and pairs points whose
fast point feature histograms (FPFH) are each other's nearest. RANSAC on three pairs at a
time shortlists motions, overlap of all points picks one, and point-to-plane ICP refines
it. Optional markers on a mat, when three or more are seen, give a pose in the mat's
frame instead. Scale bars are measured in the merged cloud and reported.

`register.register_views` places views taken in order around an object. Each view
registers to the one before, then rounds of ICP refine every view against all others
(D70). `slscan register` runs it on view clouds saved as PLY files. It writes the merged
cloud, averaged over 0.5 mm voxels, and a JSON file of each view's pose and overlap.
`slscan view-bench` runs this on synthetic views of a part a metre across and reports
pose and scale bar errors against the truth. With `--stereo` its views are matched from
pairs rendered for the metre head (D71).

`sls-scan register` does the same as `slscan register` in the application's Rust port,
and `sls-scan register-check` compares its poses with NumPy's (D73).

## Virtual bench

`slscan simulate` writes a synthetic calibration capture and three scenes (`plate`,
`sphere`, `ball-bar`) in the layouts above, then prints the commands that calibrate,
scan and measure them. `slscan scan <view>` decodes and triangulates one view of
`gray_*.png` and `phase_*.png` captures into a PLY file. `slscan measure` fits a plane,
a sphere or a ball bar to a PLY file. Real captures use the same commands.

## Mesh

`mesh.surface_nets` meshes a TSDF volume: one vertex per cell where the distance changes
sign, joined in quads across each crossing edge (`DESIGN.md` D57). `mesh.write_mesh`
writes it as PLY, OBJ or STL, and `slscan sweep-bench --tsdf --mesh` meshes a sweep.
`mesh.decimate` collapses edges by least quadric error to a face count CAD accepts, and
`--faces` sets it (D58). `mesh.fill_holes` closes holes up to a perimeter limit, which
`--fill-mm` sets, and leaves larger openings such as a part's unseen underside open
(D61).

`slscan report` fits datum spheres in a cloud or mesh, aligns them to their centres
measured on the object, and reports datum residuals and scale bar errors (D60). A datum
file without seeds has its spheres found by radius and matched by their distances, and
`--aligned` writes the cloud or mesh in the object's frame (D81).

## Preview path

Every synchronized frame pair is a depth frame: rectification, coarse disparity, affine
and second-order refinement, and TSDF integration at a coarse voxel size in WGSL kernels
on the host, with triangulation on its CPU in double precision (`DESIGN.md` D48). Poses
come from the frame's geometry against the model, helped by optional markers on a mat
around the part (D63). The scene draws the volume and the live cloud as the scanner
moves, which shows coverage and gaps. In continuous mode the fused model is the result,
and dense matching of selected frames refines it afterwards.

In the phase-shift option a depth frame is a cycle of phase-shifted patterns instead.
Its decode kernel is `app/src/decode.wgsl`, and it matches the NumPy decode (D22). The
decoder keeps its buffers between calls of one stack size (D94).
`app/src/scan.rs` triangulates the decoded columns on the CPU in double precision and
matches `slscan/triangulate.py` (D82). `slscan scan-params` writes the calibration and
decode settings it reads.

`sls-scan scan` captures a scan set from a node, decodes and triangulates it, and writes
the points as PLY (D83). With exposures in its scan parameters it decodes each one and
keeps, per pixel, the one with the largest modulation, as NumPy does (D87). In the
window, a scan set run with a scan parameter file shows its points in the scene, and
"Save points" writes them.

The application gate runs a scan through the window and compares its points with
`sls-scan scan` (D85).

Rectification runs in `app/src/rectify.wgsl` from each camera's rectification parameters
and matches `cv2.remap` within one grey level (D31). Block disparity runs in
`app/src/coarse.wgsl` and matches `block_disparity_small` (D35). Second-order refinement runs in
`app/src/refine.wgsl`, one workgroup per sample after a gradient pass, and matches the
NumPy refinement (D29, D30). It also runs both stages of direct matching (D33).
`slscan stereo-vectors` writes a raw and a rectified view with the rectification
parameters, its affine matches, its coarse samples and their NumPy refinements.
`sls-scan rectify-check`, `sls-scan coarse-check` and `sls-scan refine-check`, each
given that directory, compare the kernels with them.

`app/src/depth.rs` chains the kernels into a depth frame from a raw pair, and `sls-scan
depth-check` compares its points with the NumPy reference (D36). The kernels share one
device, and the rectified pair stays on it through block matching and refinement (D37).
`app/src/tsdf.wgsl` folds depth frames into the TSDF volume of `tsdf.py` with one
invocation per voxel and lists the volume's surface on the device. `sls-scan tsdf-check`
compares both with the NumPy volume and surface that `slscan tsdf-vectors` writes (D52,
D53).

Recorded pairs replay through the same path (D47):

    app/target/debug/sls-scan capture --set stereo --frames 3 --out data/recorded
    slscan depth-params --head stereo --step 16 --out data/replay.json
    slscan replay-reference data/recorded --params data/replay.json --head stereo
    app/target/debug/sls-scan replay --frames data/recorded --params data/replay.json

`slscan replay-reference` writes a NumPy reference beside each pair, and `sls-scan
replay` compares each GPU depth frame with it. It then runs the pairs through the live
pipeline and prints its rate. With `--rate 30` it offers them at 30 per second as the
window does, and a newer pair replaces one still waiting (D102). Recorded light-off
pairs, `dark_000` onward, go in with them (D105).

On the CPU fallback a preview skips refinement (D49). It matches blocks at 1/8
resolution, keeps lit blocks and triangulates each at its centre. Replay compares
previews with `preview.bin`, which `slscan replay-reference` writes when the parameters
have a preview section.

## Application

The Rust application owns the loop: triggered capture of frame pairs from the capture
node, timestamps per pair, WGSL kernels, the fused volume, the deferred processing queue,
and the three panels. Calibration YAML comes from the Python tools and goes back out with
each scan. Selected frame pairs are written for the deferred path, and results as PLY,
OBJ and STL with the report.

A stereo run measures each frame pair on one thread and fuses it on another, so a pair
is measured while the pair before it is fused (`DESIGN.md` D40, D99). One pair waits for
the measuring stage, and a newer pair replaces it, so that stage starts on the newest
pair offered (D102). The scene panel draws the latest frame's points.

Over the fused model, that panel draws the newest frame at its tracked pose. It is green
when its pair was measured as it arrived and amber when it waited for the measuring
stage, so amber means the camera outruns the pipeline (D103).

A stereo run asks its node for pairs as fast as it can, or at the trigger rate set
beside its frame count. Against a stand-in started with `slscan node-sim --paced-poses
61`, a rate of 30 pairs per second drives the window as a camera on a moving head would
(D104).

`slscan depth-params` writes the parameters it needs for a calibration, a sample every 12 px and a refinement
exit at 10⁻⁴ px by default (D74), and `sls-scan depth` runs the same path without the
window. The calibration panel captures board pairs, solves the stereo
calibration through the Python tool and writes those parameters for the next run (D41).

The "Reference dots (smooth panels)" switch takes every 4th stereo pair with the pattern
light off, or every 2nd to 8th by a count beside it (D114). The worker finds that
pair's dots against the next lit depth frame and poses it from them before the lit
frame joins, and `sls-scan depth --dark-every 4` does the same without the window
(D79).

Light-off pairs wait beside the lit pair offered after them, so each is placed in
capture order when newer lit pairs replace older ones, and none is dropped for a newer
one (D105, D112). Finding one's dots takes about 8 ms of the worker's measuring stage
(D109).

With that switch on, a lit frame starts between light-off poses: on a curve through the
two before it and the one after. It is fused once that later pair is posed, so the model
follows the head about 140 ms later than without dots (D107). Lit pairs after a stream's
last light-off pair are measured and left out of the model (D113).

On a software adapter such as lavapipe, the worker and `sls-scan depth` run previews in
place of depth frames and leave them out of the fused model (D50). Each depth frame drops
isolated points (D59) and folds into a coarse TSDF at its tracked pose, and the scene
panel draws that volume's surface (D54). `sls-scan depth --surface` writes it as PLY,
and `--volume` saves the volume for `slscan mesh-volume` to mesh (D62).

The scene panel's Export mesh button meshes the live volume in Rust and writes the file,
as `sls-scan depth --mesh` does (D65, D66). `sls-scan mesh-volume` meshes a saved volume
without Python.
