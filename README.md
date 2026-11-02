# sls-scan

[![Tests](https://github.com/Jbsco/sls-scan/actions/workflows/test.yml/badge.svg)](https://github.com/Jbsco/sls-scan/actions/workflows/test.yml)
[![Style](https://github.com/Jbsco/sls-scan/actions/workflows/style.yml/badge.svg)](https://github.com/Jbsco/sls-scan/actions/workflows/style.yml)

A structured-light 3D scanner built from two synchronized cameras and a pattern light on a
stiff bar, with calibration, capture, matching, registration and meshing scripted on
Linux. The light throws a fixed random pattern, so parts scan without any coating, and
every depth frame is one exposure, so the scanner can sweep around a part continuously.
It measures objects from a few centimetres to a few metres across, as meshes for CAD.

Handheld scanners sold for this work capture only through their makers' Windows and
macOS software. This project applies the same principle, under the operator's control
from calibration to export.

Nothing in this repository has run on hardware yet. Until the rig is built and the
acceptance tests in `docs/accuracy.md` pass on it, use anything here at your own risk.

## Contents

- [Status](#status)
- [Applications](#applications)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Hardware](#hardware)
- [Accuracy](#accuracy)
- [Repository layout](#repository-layout)
- [Development](#development)
- [License](#license)

## Status

Phase 0 of `ROADMAP.md`, the scaffold, is complete. Calibration (Phase 1) and the static
reference pipeline (Phase 2) are implemented and verified on a virtual bench of
synthetic captures, for active stereo (`DESIGN.md` D25) and for the phase-shift option
(D15 and D17). Phase 3 is in progress: the capture protocol, a stand-in node, and the
application shell with its capture client, three panels and calibration runs for both
heads (D21, D41).

Phase 4 has GPU kernels for the phase-shift decode (D22), stereo rectification (D31),
block disparity (D35) and second-order stereo refinement (D29). They chain into a depth
frame from a raw pair at 15 frames per second on a desktop GPU (D36, D38). The
application runs that path live on stereo captures and draws each frame's points (D40). Phase 5 has ICP tracking and point fusion, verified on a synthetic
sweep (D26). Phase 9 has second-order matching windows, which follow curved surfaces
(D28).

Bench acceptance of each phase waits on the hardware in `docs/hardware.md`.

## Applications

Structured-light scanning serves any job that needs the shape of a real object as
measured geometry. Among them:

- **Automotive.** Interiors, body panels and the space between chassis components, for
  parts designed to fit them.
- **Molds and patterns.** A pattern before its mold is taken, and the mold or the cast
  part against it.
- **Small parts.** Brackets, housings and fittings for reproduction by printing or
  machining, on a turntable in a later phase.
- **Interiors and architecture.** Rooms, fixtures and built-in equipment a few metres
  across.
- **Archival.** Objects recorded as meshes for reference or reproduction.

## How it works

1. **Calibrate.** Each camera's intrinsics and the pair's extrinsics come from a printed
   ChArUco board in 12 to 30 poses seen by both cameras. The board's measured square size
   sets the absolute scale.
2. **Capture.** One trigger pulse exposes both cameras at the same instant while the
   pattern light throws its fixed random dots on the part. Each pair of frames is a depth
   frame, up to 30 per second over gigabit Ethernet.
3. **Match.** Both images are rectified so that matching points share a row. A coarse
   disparity comes from semi-global matching, and each sample is refined to a small
   fraction of a pixel by an affine window, which follows the different foreshortening
   each camera sees on a slanted surface. A second-order window then follows the
   surface's curvature.
4. **Triangulate.** Each matched pair of pixels gives a point in the left camera's frame.
5. **Track and fuse.** While the scanner moves, each depth frame is aligned to the model
   from its geometry and from markers on the mat around the part, and fused into a
   volume.
6. **Mesh and export.** Outlier removal, meshing, hole filling, decimation to a size CAD
   can hold, alignment to datum points measured on the object, and export as PLY, OBJ or
   STL.

Matching and triangulation are NumPy and OpenCV in `slscan/stereo.py`, written as the
reference the GPU kernels are checked against. A phase-shift path with a calibrated
projector (Gray codes and phase shifts) is also implemented, for edge-sharp static scans.

## Architecture

The application is Rust with wgpu and egui, so one binary runs on an AMD or an NVIDIA
card through Vulkan, and on the CPU through Mesa's software Vulkan when no card is
present. It has three panels: configuration (devices, calibration, exposure), preview
and processing (live view, coverage, capture, the deferred queue), and the scene (the
fused model with its datums, scale bars, residual report and measurement tools).

The capture node is a Raspberry Pi 5 with both cameras and the trigger line, streaming
frame pairs to the host over Ethernet. Matching, triangulation and fusion run on the
host's GPU. Scan modes are continuous (sweep and fuse), static (view, reorient, view),
and keyed (named markers define the object's frame). Python tools do the calibration
solve and mesh post-processing. `DESIGN.md` D11 and D25 have the reasoning.

## Hardware

| Part | Choice | Reason |
|---|---|---|
| Cameras | Two Raspberry Pi Global Shutter cameras, IMX296, 1456 × 1088, on the Pi capture node | One trigger pulse exposes both at once, so a moving scanner is no problem |
| Lenses | 16 mm C-mount for near range (0.5 m), 6 mm CS-mount for a metre | Field of 157 mm or 0.84 m across at those distances |
| Capture node | Raspberry Pi 5: both cameras, a GPIO trigger line, gigabit Ethernet to the host | Trigger and capture on one clock |
| Pattern light | Any LED projector showing one random-dot image, or an LED slide projector | Puts texture on plain surfaces. It is neither calibrated nor synchronized |
| GPU | Any Vulkan-capable card (AMD or NVIDIA on Linux), or Mesa lavapipe on the CPU | One set of WGSL kernels for matching, fusion and the scene |
| Bar | Steel or carbon fibre, cameras 0.2 m apart for 0.5 m or 0.35 m apart for a metre, toed in | Calibration stays valid only while the cameras do not move relative to each other |
| Calibration board | ChArUco, printed flat for a metre, glass or ceramic for near range | One board calibrates both cameras and the pair |
| Markers and scale bars | Printed ArUco markers on the mat around the part, bars at a caliper-measured spacing | Tracking during sweeps and an independent check of scale |

Glossy, black and transparent surfaces still need a matte spray. `docs/hardware.md` has
the full list, trigger wiring, rates and build notes.

## Accuracy

Depth resolution follows δz = z²·δd/(f·b), with δd the matching precision in pixels. On
the virtual bench, two Global Shutter cameras with 16 mm lenses on a 0.2 m baseline at
0.5 m match uncoated parts to about 0.01 px. An 80 mm plate measures 8 to 10 µm RMS and
a ball bar's spacing lands within 0.02 mm. Affine matching windows read a 25 mm sphere
about 40 µm inside its surface. Second-order windows follow its curvature and bring that
offset under 1 µm. `docs/accuracy.md` has the numbers, the acceptance tests and the
error budget.

## Repository layout

| Path | Contents |
|---|---|
| `slscan/` | Python package: calibration, stereo matching, the phase-shift path, artifact fits, synthetic captures, capture protocol with a stand-in node, and the commands in `cli.py` |
| `app/` | Rust application: capture client, three panels, and a headless `capture` mode |
| `tests/` | pytest tests of the pure functions, and the virtual bench on synthetic captures |
| `docs/hardware.md` | Bill of materials, rig geometry and build notes |
| `docs/accuracy.md` | Error model and acceptance tests |
| `docs/pipeline.md` | Each stage with its inputs, outputs and file formats |
| `docs/capture-protocol.md` | Messages between the application and a capture node |
| `tools/` | Markdown rendering check, prose check |
| `style.do`, `test.do`, `app.do`, `verify.do`, `node.do` | The gates, runnable with `bash` or `redo` |
| `Dockerfile` | Stand-in capture node image |
| `DESIGN.md` | Numbered design decisions with their reasons |
| `ROADMAP.md` | Phases with acceptance criteria, and open questions |

## Development

Python 3.11 or later with the packages in `pyproject.toml` (OpenCV with the contrib
modules, NumPy, Open3D, PyYAML, pytest). The application needs Rust 1.93, pinned in
`app/rust-toolchain.toml`, and a Vulkan driver (or Mesa's lavapipe for the CPU path).

    python3 -m venv .venv && . .venv/bin/activate
    pip install -e '.[dev]'
    bash test.do                              # the pytest suite
    bash style.do                             # markdown, sanitization and prose
    bash app.do                               # Rust format, lints, tests, protocol check
    bash verify.do                            # every gate, cheapest first
    bash node.do                              # virtual bench over the capture protocol, needs Docker

The virtual bench runs the whole pipeline on synthetic captures, with no hardware:

    slscan simulate --head stereo --out data/simulated    # prints calibrate, scan and measure commands
    slscan node-sim --scene sphere            # stand-in capture node on port 5600
    cargo run --release --manifest-path app/Cargo.toml    # application window

A change is finished when `verify.do` is green, and CI runs the same gates on every push
and pull request. Captures, calibrations and scan outputs live under `data/`, which is
not tracked.

## License

GPL-3.0-or-later. See `LICENSE`.
