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
(`docs/accuracy.md`).

`stereo.direct_matches` matches by Gauss-Newton alone: an affine window from the coarse
disparity, kept on its correlation, then a second-order window (`DESIGN.md` D32). Its
coarse disparity comes from block matching by normalized cross-correlation at quarter
resolution (D34). It is the reference for the GPU path, which has no Lucas-Kanade or ECC
stage.

## Track and fuse

`track.fuse` aligns each depth frame to the model by point-to-plane ICP, starting from
the previous pose advanced by the last motion, then adds its points to the model as
averages over 0.5 mm voxels (`DESIGN.md` D26). A captured sweep is a directory of
`frame_000` onward, each with `left.png` and `right.png` (D27), and `slscan fuse <sweep>`
matches, tracks and fuses it into one PLY file.

`slscan sweep-bench` runs a synthetic sweep through matching, tracking and fusion and
compares poses and artifacts with the truth. The application runs a Rust port of both,
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

Distortion is radial and tangential with k3 fixed, for both devices. Column and row
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
masked when a pair lacks contrast, the sinusoid lacks modulation, or the two columns
disagree by more than a pixel (`DESIGN.md` D16).

## Triangulate

Each pixel's ray comes from the camera matrix and distortion, each column's plane from
the projector matrix and extrinsics, and their intersection is the point. Projector
distortion is applied at the row where the previous estimate lands, over three
iterations. Points with
low confidence, grazing angles or a column outside the projector are dropped. Output is
a PLY per view in the camera frame.

## Register

ArUco targets detected in the camera images give each view's pose in the target frame
once three or more targets with known positions are seen. Target positions come from a
bundle adjustment over all views. ICP (Open3D) refines the poses on overlapping clouds.
Scale bars are measured in the merged cloud and reported.

## Virtual bench

`slscan simulate` writes a synthetic calibration capture and three scenes (`plate`,
`sphere`, `ball-bar`) in the layouts above, then prints the commands that calibrate,
scan and measure them. `slscan scan <view>` decodes and triangulates one view of
`gray_*.png` and `phase_*.png` captures into a PLY file. `slscan measure` fits a plane,
a sphere or a ball bar to a PLY file. Real captures use the same commands.

## Mesh

Open3D statistical outlier removal, normal estimation, Poisson or ball-pivot
reconstruction, hole filling, quadric decimation to the face count CAD accepts, and a
rigid alignment to datum points measured on the object. Export as PLY, OBJ and STL with
a report of scale bar lengths and datum residuals.

## Preview path

Every synchronized frame pair is a depth frame: rectification, coarse disparity, affine
and second-order refinement, and TSDF integration at a coarse voxel size in WGSL kernels
on the host, with triangulation on its CPU in double precision (`DESIGN.md` D48). Poses come from the frame's geometry against the model, helped by
markers on the mat around the part. The scene draws the volume and the live cloud as
the scanner moves, which shows coverage and gaps. In continuous mode the fused model is
the result, and dense matching of selected frames refines it afterwards.

In the phase-shift option a depth frame is a cycle of phase-shifted patterns instead.
Its decode kernel is `app/src/decode.wgsl`, and it matches the NumPy decode (D22).

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
invocation per voxel, and `sls-scan tsdf-check` compares it with the NumPy volume that
`slscan tsdf-vectors` writes (D52).

Recorded pairs replay through the same path (D47):

    app/target/debug/sls-scan capture --set stereo --frames 3 --out data/recorded
    slscan depth-params --head stereo --step 16 --out data/replay.json
    slscan replay-reference data/recorded --params data/replay.json --head stereo
    app/target/debug/sls-scan replay --frames data/recorded --params data/replay.json

`slscan replay-reference` writes a NumPy reference beside each pair, and `sls-scan
replay` compares each GPU depth frame with it.

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

A stereo run sends each frame pair through the depth frame on a worker thread, and the
scene panel draws the latest frame's points (`DESIGN.md` D40). `slscan depth-params`
writes the parameters it needs for a calibration, and `sls-scan depth` runs the same
path without the window. The calibration panel captures board pairs, solves the stereo
calibration through the Python tool and writes those parameters for the next run (D41).
On a software adapter such as lavapipe, the worker and `sls-scan depth` run previews in
place of depth frames and leave them out of the fused model (D50).
