# Roadmap

Mission: a 3D scanner whose every number is measured, from two cameras and a pattern light
that cost less than a handheld, running on Linux from capture to CAD export.

A phase is complete when its acceptance test passes on the bench and `verify.do` is green.

## Phases

- Phase 0, complete 2026-10-31. Scaffold. License, gates, design ledger, roadmap, docs,
  package skeleton with a tested pattern generator. Acceptance: `verify.do` green,
  locally and in CI (D14).
- Phase 1, in progress. Calibration (Python tooling). Each camera's intrinsics from a
  ChArUco board, the extrinsics of the pair, results saved as YAML with the board's
  square size as the scale. Acceptance: reprojection error under 0.3 px for each camera,
  and repeated calibrations within 0.1 % in focal length. Stereo calibration is in place
  and verified on synthetic views (D25). On the virtual bench, independent sessions with
  a 132 × 96 mm board and 24 poses agree within 0.054 % (D44). Bench acceptance waits on
  the rig.
- Phase 2, in progress. One static view (Python reference pipeline). Synchronized capture
  of both cameras under the pattern light, rectification, matching with affine windows,
  triangulation, filtering. This is the reference the GPU kernels are checked against.
  Acceptance: a flat plate fits a plane under 0.2 mm RMS over 0.5 m, and a scale bar
  measures within 0.2 mm in one view. Matching, triangulation and artifact fits are in
  place and pass on the virtual bench (D25), where the metre head meets both limits with
  50 mm scale bar spheres (D45). Capture and acceptance wait on the rig.
- Phase 3, in progress. Application shell (Rust, wgpu, egui) and the capture node: two
  Global Shutter cameras on the Pi 5 fired by one trigger pulse, frame pairs streamed over
  Ethernet, stereo calibration from the configuration panel. In place: capture protocol
  with stereo sets (D18, D27), stand-in node (D17), application shell (D20) and
  calibration runs from the panel for both heads (D21, D41). Acceptance: frame pairs
  captured in step at 30 fps with no drops over a minute, and a calibration run end to
  end from the panel. A minute of paced pairs from the stand-in passes (D46).
- Phase 4, in progress. GPU matching and preview. WGSL kernels for rectification, coarse
  disparity, refinement and a coarse TSDF, with lavapipe as the CPU fallback. Acceptance:
  15 depth frames per second at 1.6 MP on either machine's GPU, 3 or more on the CPU
  fallback, and kernel outputs within tolerance of the Phase 2 reference on recorded
  frames. Its kernels (D29, D31, D35) and CPU triangulation (D48) give a 61 ms plate
  frame on a desktop GPU (D38), live (D40). Recorded pairs (D47) and TSDF kernels (D52,
  D53) match NumPy. Unrefined previews reach 7 per second on llvmpipe (D49).
- Phase 5, in progress. Continuous scanning. Pose tracking from geometry and from markers
  on the mat around the part, TSDF fusion of the depth stream, loop closure on revisited
  areas. Acceptance: a sweep around a part fuses to a model whose ball bar is within 0.05
  mm and shows no doubling at revisited surfaces. ICP tracking with point fusion (D26)
  and a TSDF (D51) keep a synthetic sweep's ball bar within 0.03 mm. The application
  ports both (D42, D52) and draws a live coarse TSDF (D54). A closed loop returns within
  15 µm with no doubling (D55).
- Phase 6, open. Many static views. Marker detection, view poses from the markers, ICP
  refinement, keyed markers defining the object's frame, a merged cloud. Acceptance: two
  scale bars in different orientations measure within 0.5 mm after a 10-view scan of an
  object a metre across.
- Phase 7, open. Deferred processing and export. Dense matching of selected frames,
  meshing, hole filling, decimation, alignment to datums, PLY, OBJ and STL export, the
  residual report. Acceptance: a decimated mesh imports into CAD and its datum points land
  within 1 mm of their measured positions over a metre.
- Phase 8, open. The rig. Stiff bar, camera mounts, the pattern light, trigger wiring,
  handle or tripod adapter, a case for the board and markers, drawings in CAD.
  Acceptance: a calibration before and after carrying the rig to a site agrees within
  0.1 % in focal length and 0.2 mm in baseline.
- Phase 9, in progress. Near-range accuracy (D23, D25). 16 mm lenses at 0.5 m, a
  low-expansion bar, a glass target, and matching windows that follow curvature.
  Acceptance on the bench: an 80 mm flat fits a plane under 0.01 mm RMS, a 25 mm
  sphere's radius lands within 0.01 mm, and a ball bar's spacing within 0.02 mm.
  Second-order matching windows are in place and remove the curvature bias on the
  virtual bench (D28). Direct matching from block disparity measures the ball bar within
  4 µm there (D34).
- Phase 10, open. Applications. An interior a few metres across, a pattern and the mold
  taken from it, small parts on a turntable with a known axis. Acceptance per job: scale
  bars within the Phase 6 figure and datum residuals within the Phase 7 figure.
- Phase 11, candidates. Phase-shift static scans with a calibrated projector (D23, D24),
  HDR capture, color texture, a third camera for occlusion, the Haskell reference core,
  Futhark kernels.

## Open questions

- Frame rate over Ethernet: full-resolution pairs fit gigabit at 30 fps. 60 fps needs
  2 × 2 binning or a 2.5 GbE link.
- Pattern light: dot size and brightness for each working distance, measured on the
  first rig.
- CAD import size for a scan a few metres across: decimation target in faces, measured on
  the first import.
