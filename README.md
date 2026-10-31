# sls-scan

A structured-light 3D scanner built from a camera and a projector on a rigid bar, with
its calibration, capture, decoding, triangulation, registration and meshing scripted
on Linux. It scans objects from a few centimetres to a few metres across at an accuracy
that reaches 0.2 to 0.5 mm within one view and 0.5 to 1 mm across a vehicle cab, for
reverse engineering into CAD. The handheld scanners sold for this work capture only
through their own Windows and macOS software; this project is the same principle,
static instead of swept, in the open, and under the operator's control end to end.

## Contents

- [Status](#status)
- [Applications](#applications)
- [How it works](#how-it-works)
- [Hardware](#hardware)
- [Accuracy](#accuracy)
- [Repository layout](#repository-layout)
- [Development](#development)
- [License](#license)

## Status

Phase 0 of `ROADMAP.md`, the scaffold, is complete: the pattern generator and its Gray
code encode and decode are implemented and tested, and `verify.do` is green. The
calibration tooling and the static reference pipeline (Phases 1 and 2, Python) come
first, then the application with its GPU preview (Phases 3 and 4, Rust), once the
hardware in `docs/hardware.md` is on the bench.

## Applications

- **A bare vehicle cab.** With the dashboard, carpet and trim out, one scan gives the
  floor, firewall and pillar geometry for a dashboard frame, seat adapters, cable runs
  and crash-space checks, as reference geometry in CAD.
- **A plug for a composite part.** A repaired original part, finished to gloss, scanned
  before the mold is taken, so the mold and the shell can be checked against it.
- **Molds and shells.** The cured part scanned against its plug for shrink and spring.
- **Small parts.** Trim tabs, clip bosses, brackets and housings for reproduction by
  printing or machining, on a turntable in a later phase.
- **Underbody packaging.** The space around a transfer case, a crossmember or an exhaust
  run, measured once for parts that have to fit between them.

## How it works

1. **Calibrate.** The camera's intrinsics come from a printed ChArUco board in 20 to 30
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

The decoding uses OpenCV's `structured_light` module (Gray code and sinusoidal phase
shift), the point cloud work uses Open3D, and everything else is NumPy.

## Architecture

Two paths share one calibration. The **preview path** runs live at 10 to 30 depth
frames per second: a short phase-shift cycle from a 1 MP global-shutter camera
synchronized to the projector, decoded and triangulated in WGSL compute shaders
through wgpu, fused into a coarse volume and drawn in the scene as the rig moves. The
**final path** captures the full Gray code and phase-shift sequence per view at full
resolution and processes it after the scan, on the GPU in seconds per view or on the
CPU in tens of seconds, then registers and meshes it.

The application is Rust with wgpu and egui, so one binary runs on an AMD or an NVIDIA
card through Vulkan, and on the CPU through Mesa's software Vulkan when no card is
present. It has three panels: configuration (devices, calibration, pattern set,
exposure), preview and processing (live view, coverage, capture, the deferred queue),
and the scene (the final mesh with its datums, scale bars, residual report and
measurement tools). Scan modes are static (view, reorient, view), continuous (sweep
and fuse), and keyed (named targets define the object's frame). Python tools do the
calibration solve and the mesh post-processing with OpenCV and Open3D, exchanging
YAML and PLY files with the application. `DESIGN.md` D11 has the reasoning.

## Hardware

| Part | Choice | Why |
|---|---|---|
| Final camera | 12 MP, IMX477 class (Raspberry Pi HQ camera on a Pi capture node, or a USB module), 6 mm lens | Full-resolution static views; f ≈ 3,870 px, a 1.05 × 0.78 m field at 1 m |
| Preview camera | 1 MP monochrome global shutter, OV9281 class, USB, hardware trigger | The live path at the projector rate; global shutter freezes each pattern |
| GPU | Any Vulkan-capable card (AMD or NVIDIA on Linux), or Mesa lavapipe on the CPU | One set of WGSL kernels for preview, final processing and the scene |
| Projector | Used DLP with a native 1920 × 1080 DMD, 2,000 ANSI lumens or more, manual focus to 1 m, temporal processing off | DLP keeps binary patterns sharp; `docs/hardware.md` lists the requirements |
| Bar | 40 × 40 aluminum extrusion, camera and projector at a 0.35 m baseline toed in to cross at 1 m, on a tripod | The calibration holds only while the pair does not move relative to each other |
| Calibration board | ChArUco, printed flat on foam board | One board calibrates the camera, the projector and the pair |
| Targets and scale bars | Printed ArUco targets, two bars with targets at a caliper-measured spacing | Registration across views and an independent check of scale |
| Surface preparation | Matte scanning spray or dry developer | Gloss and black surfaces return no usable pattern without it |

`docs/hardware.md` has the full list with prices, the rig geometry and the build notes.

## Accuracy

At a 1 m working distance with a 0.35 m baseline, the depth resolution is
δz = z²·δp/(f·b): 0.07 mm with phase shift at δp ≈ 0.1 px, 0.2 mm at a realistic
0.3 px, 0.7 mm with Gray code alone. Accuracy within one view, after calibration
residuals and rig drift, is 0.2 to 0.5 mm over the 1 m field. Across a cab, with
targets and scale bars, 0.5 to 1 mm. `docs/accuracy.md` derives these and sets the
acceptance tests: a flat plate, a sphere and the scale bars.

## Repository layout

- `slscan/`: the Python package. The pattern codes now, the calibration and the
  reference pipeline next, `cli.py` for the commands.
- `app/`: the Rust application, from Phase 3.
- `tests/`: pytest tests of the pure functions, the pattern codes first.
- `docs/hardware.md`: the bill of materials, rig geometry and build notes.
- `docs/accuracy.md`: the error model and the acceptance tests.
- `docs/pipeline.md`: each stage, its inputs, outputs and file formats.
- `DESIGN.md`: the design decision ledger, dated and numbered.
- `ROADMAP.md`: phases with acceptance criteria.
- `tools/mdlint.py`: the markdown gate.
- `style.do`, `test.do`, `verify.do`: the gates.

## Development

Python 3.11 or later with the packages in `pyproject.toml` (OpenCV with the contrib
modules, NumPy, Open3D, PyYAML, pytest). The application, from Phase 3, needs a stable
Rust toolchain and a Vulkan driver (or Mesa's lavapipe for the CPU path).

    python3 -m venv .venv && . .venv/bin/activate
    pip install -e '.[dev]'
    bash test.do                              # the pytest suite
    bash style.do                             # markdown and sanitization
    bash verify.do                            # every gate, cheapest first

A change is finished when `verify.do` is green. Captures, calibrations and scan outputs
live under `data/`, which is not tracked.

## License

GPL-3.0-or-later. See `LICENSE`.
