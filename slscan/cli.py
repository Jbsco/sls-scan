"""The command line: one command per pipeline stage."""

import argparse
import glob
import os
import sys

import numpy as np

from . import calibrate, decode, geometry, patterns, synthetic, triangulate

CAMERA_LIMIT_PX = 0.3
PROJECTOR_LIMIT_PX = 0.5


def _write_stack(stack: np.ndarray, prefix: str) -> None:
    import cv2

    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    for i, image in enumerate(stack):
        cv2.imwrite(f"{prefix}{i:03d}.png", image)


def _spec(args: argparse.Namespace) -> calibrate.BoardSpec:
    return calibrate.BoardSpec(args.squares_x, args.squares_y, args.square_mm / 1000.0,
                               args.marker_mm / 1000.0, args.dictionary)


def read_views(directory: str):
    """Yield the calibration views of a capture directory, one pose directory at a time."""
    import cv2

    poses = sorted(glob.glob(os.path.join(directory, "pose_*")))
    if not poses:
        raise FileNotFoundError(f"no pose_* directories in {directory}")
    for pose in poses:
        def stack(name):
            files = sorted(glob.glob(os.path.join(pose, f"{name}_*.png")))
            return np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])
        yield {"white": cv2.imread(os.path.join(pose, "white.png"), cv2.IMREAD_GRAYSCALE),
               "columns": stack("columns"), "rows": stack("rows")}


def cmd_patterns(args: argparse.Namespace) -> int:
    gray = patterns.gray_code_images(args.width, args.height)
    _write_stack(gray, f"{args.out}/gray_")
    phase = patterns.phase_shift_images(args.width, args.height, args.pitch, args.steps)
    _write_stack(phase, f"{args.out}/phase_")
    print(f"{gray.shape[0]} Gray code images and {phase.shape[0]} phase images written to {args.out}")
    return 0


PLATE_LIMIT_M = 0.0002


def read_scan(directory: str) -> dict:
    """Read a view written with the file names of slscan patterns: gray_*.png and phase_*.png."""
    import cv2

    def stack(name):
        files = sorted(glob.glob(os.path.join(directory, f"{name}_*.png")))
        if not files:
            raise FileNotFoundError(f"no {name}_*.png in {directory}")
        return np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files])
    return {"gray": stack("gray"), "phase": stack("phase")}


def write_scan(directory: str, captures: dict) -> None:
    _write_stack(captures["gray"], os.path.join(directory, "gray_"))
    _write_stack(captures["phase"], os.path.join(directory, "phase_"))


def cmd_scan(args: argparse.Namespace) -> int:
    cal = calibrate.load(args.calibration)
    view = read_scan(args.view)
    columns, mask = decode.decode_columns(view["gray"], view["phase"], cal.projector.width,
                                          args.pitch)
    points, _ = triangulate.triangulate(columns, mask, cal)
    geometry.write_ply(args.out, points)
    print(f"{len(points)} points written to {args.out}")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    points = geometry.read_ply(args.cloud)
    if args.shape == "plane":
        _, _, rms = geometry.fit_plane(points)
        print(f"plane RMS residual {1000.0 * rms:.3f} mm (limit {1000.0 * PLATE_LIMIT_M:.1f} mm)")
        return 0 if rms < PLATE_LIMIT_M else 1
    if args.shape == "sphere":
        centre, radius, rms = geometry.fit_sphere(points)
        print(f"sphere radius {1000.0 * radius:.3f} mm, form RMS {1000.0 * rms:.3f} mm, "
              f"centre {np.round(1000.0 * centre, 3)} mm")
        return 0
    labels = geometry.split_two(points)
    centres = [geometry.fit_sphere(points[labels == i])[0] for i in (0, 1)]
    print(f"ball bar spacing {1000.0 * np.linalg.norm(centres[0] - centres[1]):.3f} mm")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    rig = synthetic.reference_rig()
    spec = calibrate.BoardSpec(9, 6, 0.04, 0.03)
    import cv2

    for i, view in enumerate(synthetic.calibration_views(rig, spec, args.seed)):
        pose = os.path.join(args.out, "calibration", f"pose_{i:03d}")
        os.makedirs(pose, exist_ok=True)
        cv2.imwrite(os.path.join(pose, "white.png"), view["white"])
        _write_stack(view["columns"], os.path.join(pose, "columns_"))
        _write_stack(view["rows"], os.path.join(pose, "rows_"))
    scenes = synthetic.acceptance_scenes()
    for name, scene in scenes.items():
        captures, _ = synthetic.scan_captures(rig, scene, pitch=args.pitch, seed=args.seed)
        write_scan(os.path.join(args.out, name), captures)
    sphere = scenes["sphere"][0]
    bar = scenes["ball-bar"]
    print(f"synthetic captures written to {args.out}")
    print(f"true sphere radius {1000.0 * sphere.radius:.3f} mm, "
          f"true ball bar spacing {1000.0 * np.linalg.norm(bar[0].centre - bar[1].centre):.3f} mm")
    print("next:")
    print(f"  slscan calibrate {args.out}/calibration --squares-x 9 --squares-y 6 --square-mm 40 "
          f"--marker-mm 30 --projector-width {rig.projector.width} "
          f"--projector-height {rig.projector.height} --out {args.out}/calibration.yaml")
    for name, shape in (("plate", "plane"), ("sphere", "sphere"), ("ball-bar", "ball-bar")):
        print(f"  slscan scan {args.out}/{name} --calibration {args.out}/calibration.yaml "
              f"--pitch {args.pitch} --out {args.out}/{name}.ply && "
              f"slscan measure {args.out}/{name}.ply --shape {shape}")
    return 0


def cmd_board(args: argparse.Namespace) -> int:
    import cv2

    spec = _spec(args)
    px = round(args.square_mm / 25.4 * args.dpi)
    cv2.imwrite(args.out, calibrate.board_image(spec, px, px))
    print(f"board of {spec.squares_x} x {spec.squares_y} squares at {args.dpi} dpi written to "
          f"{args.out}; print it at 100 % scale and measure a square with calipers")
    return 0


def cmd_calibration_patterns(args: argparse.Namespace) -> int:
    import cv2

    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(os.path.join(args.out, "white.png"),
                np.full((args.height, args.width), 255, dtype=np.uint8))
    _write_stack(patterns.gray_code_images(args.width, args.height, "columns"),
                 os.path.join(args.out, "columns_"))
    _write_stack(patterns.gray_code_images(args.width, args.height, "rows"),
                 os.path.join(args.out, "rows_"))
    print(f"calibration patterns written to {args.out}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    cal = calibrate.calibrate(read_views(args.captures), _spec(args),
                              (args.projector_width, args.projector_height))
    calibrate.save(args.out, cal)
    print(f"{cal.views} views, calibration written to {args.out}")
    print(f"camera reprojection {cal.camera.rms_px:.3f} px (limit {CAMERA_LIMIT_PX})")
    print(f"projector reprojection {cal.projector.rms_px:.3f} px (limit {PROJECTOR_LIMIT_PX})")
    print(f"baseline {1000.0 * np.linalg.norm(cal.stereo.translation):.2f} mm")
    ok = cal.camera.rms_px < CAMERA_LIMIT_PX and cal.projector.rms_px < PROJECTOR_LIMIT_PX
    return 0 if ok else 1


def cmd_calibration_diff(args: argparse.Namespace) -> int:
    d = calibrate.compare(calibrate.load(args.first), calibrate.load(args.second))
    ok = True
    for key in ("camera_fx", "camera_fy", "projector_fx", "projector_fy"):
        within = d[key] <= args.focal_limit
        ok &= within
        print(f"{key} change {100.0 * d[key]:.3f} % (limit {100.0 * args.focal_limit:.3f} %)")
    within = d["baseline_m"] * 1000.0 <= args.baseline_limit_mm
    ok &= within
    print(f"baseline change {1000.0 * d['baseline_m']:.3f} mm (limit {args.baseline_limit_mm} mm)")
    return 0 if ok else 1


def _board_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--squares-x", type=int, default=11)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=50.0, help="square size, measured on the print")
    p.add_argument("--marker-mm", type=float, default=37.0)
    p.add_argument("--dictionary", default="DICT_5X5_100")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="slscan", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("patterns", help="write the projector pattern stack as PNG files")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--pitch", type=int, default=32, help="phase shift pitch in projector pixels")
    p.add_argument("--steps", type=int, default=4, help="phase shift steps")
    p.add_argument("--out", default="data/patterns", help="output directory")
    p.set_defaults(func=cmd_patterns)

    p = sub.add_parser("board", help="write the printable ChArUco board as a PNG file")
    _board_arguments(p)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--out", default="data/board.png")
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("calibration-patterns", help="write the white frame and both Gray code stacks")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--out", default="data/calibration-patterns")
    p.set_defaults(func=cmd_calibration_patterns)

    p = sub.add_parser("calibrate", help="calibrate camera, projector and pair from captured poses")
    p.add_argument("captures", help="directory of pose_* directories")
    _board_arguments(p)
    p.add_argument("--projector-width", type=int, default=1920)
    p.add_argument("--projector-height", type=int, default=1080)
    p.add_argument("--out", default="data/calibration.yaml")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("calibration-diff", help="compare two calibrations against repeatability limits")
    p.add_argument("first")
    p.add_argument("second")
    p.add_argument("--focal-limit", type=float, default=0.001, help="relative focal length change")
    p.add_argument("--baseline-limit-mm", type=float, default=0.2)
    p.set_defaults(func=cmd_calibration_diff)

    p = sub.add_parser("scan", help="decode and triangulate one view into a PLY point cloud")
    p.add_argument("view", help="directory of gray_*.png and phase_*.png captures")
    p.add_argument("--calibration", default="data/calibration.yaml")
    p.add_argument("--pitch", type=int, default=32, help="phase shift pitch in projector pixels")
    p.add_argument("--out", default="data/view.ply")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("measure", help="fit an acceptance artifact to a point cloud")
    p.add_argument("cloud")
    p.add_argument("--shape", choices=("plane", "sphere", "ball-bar"), required=True)
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("simulate", help="write synthetic calibration and scan captures")
    p.add_argument("--out", default="data/simulated")
    p.add_argument("--pitch", type=int, default=16, help="phase shift pitch for the 1024-column virtual projector")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
