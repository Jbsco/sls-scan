# Roadmap

Mission: a structured-light scanner whose every number is measured, from a camera and a
projector that cost less than a handheld, running on Linux from capture to CAD export.

A phase is complete when its acceptance test passes on the bench and `verify.do` is green.

## Phases

- Phase 0, complete 2026-10-31. Scaffold. License, gates, design ledger, roadmap, docs, the
  package skeleton with a tested pattern generator (Gray code encode and decode, phase
  shift images). Acceptance: `verify.do` green, locally and in CI on the self-hosted
  runner (D10).
- Phase 1, open. Calibration (Python tooling). Camera intrinsics from a ChArUco board,
  the projector as an inverse camera through projected Gray codes, stereo extrinsics,
  results saved as YAML with the board's square size as the scale. Acceptance:
  reprojection error under 0.3 px for the camera and under 0.5 px for the projector,
  repeated calibrations within 0.1 % in focal length.
- Phase 2, open. One static view (Python reference pipeline). Pattern display,
  synchronized capture at one or more exposures, Gray code decode, phase shift
  refinement, triangulation, filtering. This is the reference implementation whose
  vectors the kernels are later checked against. Acceptance: a flat plate fits a plane
  with an RMS under 0.2 mm over 0.5 m, and a scale bar measures within 0.2 mm in one
  view.
- Phase 3, open. The application shell (Rust, wgpu, egui) and the capture node. The
  three panels, calibration through the frontend (D12), the capture interface with its
  two implementations (the Pi node of D13 streaming over Ethernet, USB cameras on the
  host), pattern display flipped on the vertical blank, the trigger line, frame
  timestamps. Acceptance: pattern frames captured in order at the projector rate with
  no drops over a minute, and a calibration run end to end from the panel.
- Phase 4, open. GPU decode and preview. WGSL kernels for phase decode, unwrapping,
  triangulation and the coarse TSDF, the live point cloud and volume in the scene,
  the CPU fallback through lavapipe. Acceptance: 15 depth frames per second at 1 MP
  on either machine's GPU, 3 or more on the CPU fallback, and kernel outputs within
  tolerance of the Phase 2 reference on recorded frames.
- Phase 5, open. Many views and markers. Coded target detection in the application,
  view poses from the targets, ICP refinement, keyed targets defining the object's
  frame, a merged cloud. Acceptance: two scale bars in different orientations measure
  within 0.5 mm after a 10-view scan, and a sphere scanned from four sides fits its
  diameter within 0.3 mm.
- Phase 6, open. Continuous scanning. Pose tracking from targets and ICP while the rig
  moves, TSDF fusion of the preview stream, loop closure on revisited targets.
  Acceptance: a slow sweep of a 1 m object produces a fused preview whose scale bars
  are within 1 mm, with no visible doubling at revisited surfaces.
- Phase 7, open. Deferred full-resolution processing and export. The full sequence
  per view queued and processed on the GPU or CPU, meshing, hole filling,
  decimation, alignment to datums, PLY, OBJ and STL export, the residual report.
  Acceptance: a decimated cab mesh imports into CAD and its datum points land within
  1 mm of their measured positions.
- Phase 8, open. The rig. Extrusion bar, mounts for both cameras and the projector,
  the trigger wiring, tripod adapter, a case for the board and targets, drawings in
  CAD. Acceptance: a calibration before and after carrying the rig to the vehicle
  agrees within 0.1 % in focal length and 0.2 mm in baseline.
- Phase 9, open. Applications. A bare cab, a composite plug, a mold against its plug,
  small parts on a turntable with a known axis. Acceptance per job: the scale bars
  within the Phase 5 figure and the datum residuals within the Phase 7 figure.
- Phase 10, candidates. HDR capture, color texture, a second camera for occlusion, a
  kilohertz DLP for faster preview, a one-shot coverage pattern, the Haskell reference
  core, Futhark kernels, a tethered mirrorless camera for the final path.

## Open questions

- The projector refresh the preview can rely on: 60 Hz is the floor for a used office
  DLP, 120 Hz doubles the preview rate if the unit accepts it at 1080p.
- The projector's gamma: measured per unit and linearized in the phase shift, or
  avoided with a binary-only decode at lower precision.
- Which CAD import size holds the cab: decimation target in faces, measured on the
  first import.
