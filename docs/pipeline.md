# Pipeline

Each stage is one module in `slscan/` and one command in `slscan/cli.py`, with files
between stages so any stage can be rerun from its input. All data lives under `data/`,
which is not tracked.

## Stages

| Stage | Module | Input | Output |
|---|---|---|---|
| Patterns | `patterns.py` | projector width and height, phase pitch and steps | pattern image stack and decode tables |
| Calibration | `calibrate.py` | board images, projected-code images on the board | `calibration.yaml`: camera matrix, distortion, projector matrix, extrinsics, board scale |
| Capture | `capture.py` | pattern stack, exposure list | one directory per view with the images and a manifest |
| Decode | `decode.py` | a view directory | per-pixel projector column, confidence mask |
| Triangulate | `triangulate.py` | decoded view, `calibration.yaml` | a point cloud per view (PLY) |
| Register | `register.py` | view clouds and target detections | view poses, a merged cloud, scale bar report |
| Mesh | `mesh.py` | merged cloud, datum points | a mesh (PLY, OBJ, STL) and the residual report |

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
0.2 mm.

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

The live path trades resolution and code length for rate. The projector cycles three or
four phase-shifted sinusoids at one pitch, and the global-shutter camera captures each
at the projector rate. Every complete cycle gives a depth frame: phase decode,
unwrapping against the previous frame and the fused model, triangulation, and TSDF
integration at a coarse voxel size, all in WGSL kernels.

Poses come from coded targets seen in the camera frames, refined by ICP against the
model. The scene draws the volume and the live cloud as the rig moves, which shows
coverage and gaps. The preview shows the operator where to point the rig, and the
deliverable comes from the final path.

## Application

The Rust application owns the devices and the loop. It owns projector output as a
fullscreen surface, triggered capture from the preview camera, sequenced capture from
the final camera, timestamps per frame, WGSL kernels, the fused volume, the deferred
processing queue, and the three panels. Calibration YAML comes from the Python tools and goes back
out with each scan. Final views are written as image stacks with a manifest for the
deferred path, and results as PLY, OBJ and STL with the report.
