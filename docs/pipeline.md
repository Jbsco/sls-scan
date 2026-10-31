# Pipeline

Each stage is one module in `slscan/` and one command in `slscan/cli.py`, with files
between stages so any stage can be rerun from its input. All data lives under `data/`,
which is not tracked.

## Stages

| Stage | Module | Input | Output |
|---|---|---|---|
| Patterns | `patterns.py` | projector width and height, phase pitch and steps | the pattern image stack and its decode tables |
| Calibration | `calibrate.py` | board images, projected-code images on the board | `calibration.yaml`: camera matrix, distortion, projector matrix, extrinsics, board scale |
| Capture | `capture.py` | the pattern stack, exposure list | one directory per view with the images and a manifest |
| Decode | `decode.py` | a view directory | per-pixel projector column, confidence mask |
| Triangulate | `triangulate.py` | decoded view, `calibration.yaml` | a point cloud per view (PLY) |
| Register | `register.py` | the views' clouds and target detections | view poses, a merged cloud, scale bar report |
| Mesh | `mesh.py` | the merged cloud, datum points | a mesh (PLY, OBJ, STL) and the residual report |

## Patterns

Gray code columns: for a projector of width W, ceil(log2 W) bit planes, each shown
with its inverse. Phase shift: three or four sinusoids of pitch P projector pixels,
shifted by 2π/N. The Gray code gives the period index and the phase gives the position
inside it. Row patterns are optional: triangulation needs only the column, and rows
serve the projector calibration.

## Calibration

1. Camera intrinsics: `cv2.aruco` ChArUco detection over 20 to 30 board poses, then
   `cv2.calibrateCamera`. Reprojection error under 0.3 px.
2. Projector as an inverse camera: for each board pose, the Gray code stack projected
   onto the board gives, at every detected board corner, the projector pixel that lit
   it (local homography from the surrounding decoded pixels). The corners and their
   projector pixels calibrate the projector with `cv2.calibrateCamera`. Reprojection
   error under 0.5 px.
3. Stereo: `cv2.stereoCalibrate` on the shared corners gives rotation and translation
   between the camera and the projector.
4. Scale: the board's square size, measured with calipers on the print, not taken
   from the print job.

## Capture

The projector is the second display. A fullscreen window shows each pattern, the camera
captures after a settle delay, and a manifest records the pattern index, exposure and
timestamp per image. Two or three exposures per pattern when the view has both dark and
light surfaces; the decode takes the best-exposed sample per pixel.

## Decode

OpenCV's `structured_light.GrayCodePattern` decodes the bit planes with their inverses
into a column index per pixel and a mask of pixels that were lit and unambiguous.
The phase shift images give a wrapped phase per pixel, unwrapped by the Gray code
period. The result is a float column per pixel.

## Triangulate

Each pixel's ray from the camera matrix and distortion, each column's plane from the
projector matrix and the extrinsics; the intersection is the point. Points with low
confidence, grazing angles or a column outside the projector are dropped. Output is a
PLY per view in the camera frame.

## Register

ArUco targets detected in the camera images give each view's pose in the target frame
once three or more targets with known positions are seen. The targets' positions
themselves come from a bundle adjustment over all views. ICP (Open3D) refines the poses
on the overlapping clouds. The scale bars are measured in the merged cloud and reported.

## Mesh

Open3D statistical outlier removal, normal estimation, Poisson or ball-pivot
reconstruction, hole filling, quadric decimation to the face count CAD holds, and a
rigid alignment to the datum points measured on the object. Export as PLY, OBJ and STL
with a report of the scale bar lengths and the datum residuals.

## Preview path

The live path trades resolution and code length for rate. The projector cycles three or
four phase-shifted sinusoids at one pitch, the global-shutter camera captures each
at the projector rate, and every complete cycle gives a depth frame: phase decode,
unwrapping against the previous frame and the fused model, triangulation, and TSDF
integration at a coarse voxel size, all in WGSL kernels. Poses come from the coded
targets seen in the camera frames, refined by ICP against the model. The scene draws
the volume and the live cloud as the rig moves, which shows coverage and gaps. The
preview is never the deliverable: it tells the operator where to point the rig.

## Application

The Rust application owns the devices and the loop: projector output as a fullscreen
surface, triggered capture from the preview camera and sequenced capture from the
final camera, timestamps per frame, the WGSL kernels, the fused volume, the deferred
processing queue, and the three panels. Calibration YAML comes from the Python tools
and goes back out with each scan. The final views are written as image stacks with a
manifest for the deferred path, and the results as PLY, OBJ and STL with the report.
