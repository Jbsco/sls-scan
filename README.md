# sls-scan

[![Tests](https://github.com/Jbsco/sls-scan/actions/workflows/test.yml/badge.svg)](https://github.com/Jbsco/sls-scan/actions/workflows/test.yml)
[![Style](https://github.com/Jbsco/sls-scan/actions/workflows/style.yml/badge.svg)](https://github.com/Jbsco/sls-scan/actions/workflows/style.yml)

A structured-light 3D scanner built from a camera and a projector on a rigid bar, with
calibration, capture, decoding, triangulation, registration and meshing scripted on
Linux. It scans objects from a few centimetres to a few metres across, at 0.2 to 0.5 mm
within one view and 0.5 to 1 mm over a few metres, as meshes for CAD.
Handheld scanners sold for this work capture only through their makers' Windows and
macOS software. This project applies the same measurement principle from a static rig,
under the operator's control from calibration to export.

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

Phase 0 of `ROADMAP.md`, the scaffold, is complete. Phase 1 calibration tooling is
implemented and verified against synthetic views with known parameters (`DESIGN.md`
D15). The static reference pipeline (Phase 2, Python) is next, then the application
with its GPU preview (Phases 3 and 4, Rust). Bench acceptance of each phase waits on
the hardware in `docs/hardware.md`.

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

1. **Calibrate.** Camera intrinsics come from a printed ChArUco board in 20 to 30
   poses. The projector is calibrated as an inverse camera: Gray codes projected onto the
   same board map its corners to projector pixels. A stereo calibration between the two
   gives the baseline. The board's printed square size sets the absolute scale.
2. **Capture.** The projector shows about 11 Gray code patterns and their inverses, then
   three or four phase-shifted sinusoids, and the camera photographs each one, at two or
   three exposures when the surface has both dark and light areas.
3. **Decode.** The Gray codes give every camera pixel the projector column it sees, with
   no ambiguity. The phase shifts refine that column to about a tenth of a pixel.
4. **Triangulate.** Each camera pixel's ray meets its projector column's plane at a
   point. One view is 10 to 12 million points before filtering.
5. **Register.** Printed coded targets across the object, each seen in three or more
   views, give the view poses. ICP on the overlaps refines them. Scale bars of measured
   length in two orientations check the result.
6. **Mesh and export.** Outlier removal, Poisson or ball-pivot meshing, hole filling,
   quadric decimation to a size CAD can hold, alignment to datum points measured on the
   object, and export as PLY, OBJ or STL.

Decoding uses OpenCV's `structured_light` module (Gray code and sinusoidal phase shift),
the point cloud work uses Open3D, and everything else is NumPy.

## Architecture

Two paths share one calibration. The **preview path** runs live at 10 to 30 depth
frames per second. A short phase-shift cycle from a 1 MP global-shutter camera
synchronized to the projector is decoded and triangulated in WGSL compute shaders
through wgpu, then fused into a coarse volume and drawn in the scene as the rig moves.
The **final path** captures the full Gray code and phase-shift sequence per view at
full resolution. It processes the sequence after the scan, in seconds per view on the
GPU or tens of seconds on the CPU, then registers and meshes the result.

The application is Rust with wgpu and egui, so one binary runs on an AMD or an NVIDIA
card through Vulkan, and on the CPU through Mesa's software Vulkan when no card is
present. It has three panels: configuration (devices, calibration, pattern set,
exposure), preview and processing (live view, coverage, capture, the deferred queue),
and the scene (the final mesh with its datums, scale bars, residual report and
measurement tools).

Scan modes are static (view, reorient, view), continuous (sweep and fuse), and keyed
(named targets define the object's frame). Python tools do the calibration solve and
the mesh post-processing with OpenCV and Open3D, exchanging YAML and PLY files with the
application. `DESIGN.md` D11 has the reasoning.

## Hardware

| Part | Choice | Reason |
|---|---|---|
| Final camera | Raspberry Pi HQ camera, IMX477, 12 MP, 6 mm lens, on the Pi capture node | Full-resolution static views. f ≈ 3,870 px, a 1.05 × 0.78 m field at 1 m |
| Preview camera | Raspberry Pi Global Shutter camera, IMX296, 1456 × 1088, triggered per pattern, on the Pi capture node | Live path at the projector rate, with a global shutter that freezes each pattern |
| Capture node | Raspberry Pi 5: both cameras, HDMI to the projector, a GPIO trigger line, gigabit Ethernet to the host | Display timing, trigger and capture on one clock. USB cameras on the host are the alternative |
| GPU | Any Vulkan-capable card (AMD or NVIDIA on Linux), or Mesa lavapipe on the CPU | One set of WGSL kernels for preview, final processing and the scene |
| Projector | Used DLP with a native 1920 × 1080 DMD, 2,000 ANSI lumens or more, manual focus to 1 m, temporal processing off | DLP keeps binary patterns sharp. `docs/hardware.md` lists the requirements |
| Bar | 40 × 40 aluminum extrusion, camera and projector at a 0.35 m baseline toed in to cross at 1 m, on a tripod | Calibration stays valid only while the pair does not move relative to each other |
| Calibration board | ChArUco, printed flat on foam board | One board calibrates the camera, the projector and the pair |
| Targets and scale bars | Printed ArUco targets, two bars with targets at a caliper-measured spacing | Registration across views and an independent check of scale |
| Surface preparation | Matte scanning spray or dry developer | Gloss and black surfaces return no usable pattern without it |

`docs/hardware.md` has the full list with prices, the rig geometry and the build notes.

## Accuracy

At a 1 m working distance with a 0.35 m baseline, the depth resolution is
δz = z²·δp/(f·b). That is 0.07 mm with phase shift at δp ≈ 0.1 px, 0.2 mm at a
realistic 0.3 px, and 0.7 mm with Gray code alone. Accuracy within one view, after calibration
residuals and rig drift, is 0.2 to 0.5 mm over the 1 m field. Over a few metres, with
targets and scale bars, 0.5 to 1 mm. `docs/accuracy.md` derives these and sets the
acceptance tests: a flat plate, a sphere and the scale bars.

## Repository layout

| Path | Contents |
|---|---|
| `slscan/` | Python package: pattern codes, calibration, synthetic test views and the commands in `cli.py`, with the reference pipeline next |
| `app/` | Rust application, from Phase 3 |
| `tests/` | pytest tests of the pure functions and of calibration on synthetic views |
| `docs/hardware.md` | Bill of materials, rig geometry and build notes |
| `docs/accuracy.md` | Error model and acceptance tests |
| `docs/pipeline.md` | Each stage with its inputs, outputs and file formats |
| `tools/` | Markdown rendering check, prose check |
| `style.do`, `test.do`, `verify.do` | The gates, runnable with `bash` or `redo` |
| `DESIGN.md` | Numbered design decisions with their reasons |
| `ROADMAP.md` | Phases with acceptance criteria, and open questions |

## Development

Python 3.11 or later with the packages in `pyproject.toml` (OpenCV with the contrib
modules, NumPy, Open3D, PyYAML, pytest). The application, from Phase 3, needs a stable
Rust toolchain and a Vulkan driver (or Mesa's lavapipe for the CPU path).

    python3 -m venv .venv && . .venv/bin/activate
    pip install -e '.[dev]'
    bash test.do                              # the pytest suite
    bash style.do                             # markdown, sanitization and prose
    bash verify.do                            # every gate, cheapest first

A change is finished when `verify.do` is green, and CI runs the same gates on every push
and pull request. Captures, calibrations and scan outputs live under `data/`, which is
not tracked.

## License

GPL-3.0-or-later. See `LICENSE`.
