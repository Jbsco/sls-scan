# Roadmap

Mission: a structured-light scanner whose every number is measured, from a camera and a
projector that cost less than a handheld, running on Linux from capture to CAD export.

A phase is complete when its acceptance test passes on the bench and `verify.do` is green.

## Phases

- Phase 0, in progress. Scaffold. License, gates, design ledger, roadmap, docs, the
  package skeleton with a tested pattern generator (Gray code encode and decode, phase
  shift images). Acceptance: `verify.do` green.
- Phase 1, open. Calibration. Camera intrinsics from a ChArUco board, the projector as an
  inverse camera through projected Gray codes, stereo extrinsics, results saved as YAML
  with the board's square size as the scale. Acceptance: reprojection error under 0.3 px
  for the camera and under 0.5 px for the projector, repeated calibrations within 0.1 %
  in focal length.
- Phase 2, open. One view. Pattern display on the projector output, synchronized camera
  capture at one or more exposures, Gray code decode, phase shift refinement,
  triangulation to a point cloud, filtering of unlit and ambiguous pixels. Acceptance:
  a flat plate fits a plane with an RMS under 0.2 mm over 0.5 m, and a scale bar
  measures within 0.2 mm in one view.
- Phase 3, open. Many views. Coded target detection, view poses from the targets, ICP
  refinement, a merged cloud. Acceptance: two scale bars in different orientations
  measure within 0.5 mm after a 10-view scan, and a sphere scanned from four sides
  fits its diameter within 0.3 mm.
- Phase 4, open. Mesh and export. Outlier removal, meshing, hole filling, decimation,
  alignment to measured datum points, PLY, OBJ and STL export, a report of the scale
  bar and datum residuals per scan. Acceptance: a decimated cab mesh imports into CAD
  and its datum points land within 1 mm of their measured positions.
- Phase 5, open. The rig. Extrusion bar, camera and projector mounts, tripod adapter,
  cable routing, a case for the calibration board and targets, drawings in CAD.
  Acceptance: a calibration taken before and after carrying the rig to the vehicle agrees
  within 0.1 % in focal length and 0.2 mm in baseline.
- Phase 6, open. Applications. A bare cab, a composite plug, a mold against its plug,
  small parts on a turntable with a known axis. Acceptance per job: the scale bars within
  the Phase 3 figure and the datum residuals within the Phase 4 figure.
- Phase 7, candidates. HDR capture, color texture from a second exposure set, capture
  automation on a Raspberry Pi with a one-button view, a live preview, a tethered
  mirrorless camera, a second camera for occlusion, and a port of the pure core to a
  language with proof.

## Open questions

- Whether a 0.2 m baseline for recesses is worth a second calibration set, or a second
  camera position on the same bar covers it.
- The projector's gamma: measured per unit and linearized in the phase shift, or
  avoided with a binary-only (Gray code plus line shift) decode at lower precision.
- Which CAD import size holds the cab: decimation target in faces, measured on the
  first import.
