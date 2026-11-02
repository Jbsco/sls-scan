"""The command line: one command per pipeline stage."""

import argparse
import glob
import os
import sys

import numpy as np

from . import calibrate, decode, geometry, node, patterns, synthetic, triangulate

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
        limit = args.limit_mm / 1000.0
        print(f"plane RMS residual {1000.0 * rms:.3f} mm (limit {args.limit_mm:g} mm)")
        return 0 if rms < limit else 1
    if args.shape == "sphere":
        centre, radius, rms = geometry.fit_sphere(points)
        print(f"sphere radius {1000.0 * radius:.3f} mm, form RMS {1000.0 * rms:.3f} mm, "
              f"centre {np.round(1000.0 * centre, 3)} mm")
        return 0
    labels = geometry.split_two(points)
    centres = [geometry.fit_sphere(points[labels == i])[0] for i in (0, 1)]
    print(f"ball bar spacing {1000.0 * np.linalg.norm(centres[0] - centres[1]):.3f} mm")
    return 0


HEADS = {
    "reference": (synthetic.reference_rig, synthetic.acceptance_scenes,
                  calibrate.BoardSpec(9, 6, 0.04, 0.03), 1.0),
    "small": (synthetic.small_volume_rig, synthetic.small_volume_scenes, synthetic.SMALL_BOARD, 0.5),
}


def cmd_simulate(args: argparse.Namespace) -> int:
    make_rig, make_scenes, spec, crossing = HEADS[args.head]
    rig = make_rig()
    import cv2

    for i, view in enumerate(synthetic.calibration_views(rig, spec, args.seed, crossing)):
        pose = os.path.join(args.out, "calibration", f"pose_{i:03d}")
        os.makedirs(pose, exist_ok=True)
        cv2.imwrite(os.path.join(pose, "white.png"), view["white"])
        _write_stack(view["columns"], os.path.join(pose, "columns_"))
        _write_stack(view["rows"], os.path.join(pose, "rows_"))
    scenes = make_scenes()
    for name, scene in scenes.items():
        captures, _ = synthetic.scan_captures(rig, scene, pitch=args.pitch, seed=args.seed)
        write_scan(os.path.join(args.out, name), captures)
    sphere = scenes["sphere"][0]
    bar = scenes["ball-bar"]
    print(f"synthetic captures written to {args.out}")
    print(f"true sphere radius {1000.0 * sphere.radius:.3f} mm, "
          f"true ball bar spacing {1000.0 * np.linalg.norm(bar[0].centre - bar[1].centre):.3f} mm")
    print("next:")
    print(f"  slscan calibrate {args.out}/calibration --squares-x {spec.squares_x} "
          f"--squares-y {spec.squares_y} --square-mm {1000.0 * spec.square_m:g} "
          f"--marker-mm {1000.0 * spec.marker_m:g} --projector-width {rig.projector.width} "
          f"--projector-height {rig.projector.height} --out {args.out}/calibration.yaml")
    for name, shape in (("plate", "plane"), ("sphere", "sphere"), ("ball-bar", "ball-bar")):
        print(f"  slscan scan {args.out}/{name} --calibration {args.out}/calibration.yaml "
              f"--pitch {args.pitch} --out {args.out}/{name}.ply && "
              f"slscan measure {args.out}/{name}.ply --shape {shape}")
    return 0


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def cmd_node_sim(args: argparse.Namespace) -> int:
    import signal

    signal.signal(signal.SIGTERM, _interrupt)
    server = node.StandIn((args.host, args.port), scene=args.scene, seed=args.seed)
    print(f"stand-in node on {args.host}:{server.server_address[1]}, scene {args.scene}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    host, port = args.node.rsplit(":", 1)
    result = node.capture(host, int(port), args.set, args.out, args.pitch, args.steps,
                          wait=args.wait)
    print(f"{result['frames']} frames from {result['node']['node']} written to {args.out}")
    if result["missing"] or not result["in_order"]:
        print(f"missing frames {result['missing']}, in order: {result['in_order']}")
        return 1
    return 0


def cmd_decode_vectors(args: argparse.Namespace) -> int:
    """Write a synthetic scan and its NumPy decode as raw files for the GPU kernels."""
    import json

    rig = synthetic.reference_rig()
    captures, _ = synthetic.scan_captures(rig, synthetic.acceptance_scenes()[args.scene],
                                          pitch=args.pitch, seed=args.seed)
    columns, mask = decode.decode_columns(captures["gray"], captures["phase"],
                                          rig.projector.width, args.pitch)
    os.makedirs(args.out, exist_ok=True)
    captures["gray"].tofile(os.path.join(args.out, "gray.bin"))
    captures["phase"].tofile(os.path.join(args.out, "phase.bin"))
    columns.astype("<f4").tofile(os.path.join(args.out, "columns.bin"))
    mask.astype(np.uint8).tofile(os.path.join(args.out, "mask.bin"))
    meta = {"width": rig.camera.width, "height": rig.camera.height,
            "planes": int(captures["gray"].shape[0] // 2), "steps": int(captures["phase"].shape[0]),
            "projector_width": rig.projector.width, "pitch": args.pitch, "min_contrast": 10.0,
            "min_modulation": 10.0, "max_disagreement": 1.0}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"decode vectors of the {args.scene} scene written to {args.out}")
    return 0


def cmd_motion_error(args: argparse.Namespace) -> int:
    make_rig, make_scenes, spec, _ = HEADS[args.head]
    rig = synthetic.small_volume_preview_rig() if args.head == "small" else make_rig()
    cal = synthetic.true_calibration(rig, spec)
    plate = make_scenes()["plate"][0]
    steps_ms = 1000.0 * args.steps / args.rate
    print(f"{args.head} head, {args.rate:g} Hz, {args.steps} steps ({steps_ms:.0f} ms cycle), "
          f"pitch {args.pitch}: RMS depth error of one cycle")
    for speed in (float(s) for s in args.speeds.split(",")):
        errors = [synthetic.moving_plate_error(rig, cal, plate, speed / 1000.0 * np.array(d),
                                               args.rate, args.steps, args.pitch)
                  for d in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
        print(f"  {speed:g} mm/s: across the axis {1e6 * errors[0]:.1f} µm, "
              f"along the axis {1e6 * errors[1]:.1f} µm")
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
    import json

    try:
        cal = calibrate.calibrate(read_views(args.captures), _spec(args),
                                  (args.projector_width, args.projector_height))
    except (ValueError, FileNotFoundError) as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"calibration failed: {e}")
        return 1
    calibrate.save(args.out, cal)
    ok = cal.camera.rms_px < CAMERA_LIMIT_PX and cal.projector.rms_px < PROJECTOR_LIMIT_PX
    baseline = float(np.linalg.norm(cal.stereo.translation))
    if args.json:
        print(json.dumps({"views": cal.views, "camera_rms_px": cal.camera.rms_px,
                          "camera_limit_px": CAMERA_LIMIT_PX,
                          "projector_rms_px": cal.projector.rms_px,
                          "projector_limit_px": PROJECTOR_LIMIT_PX, "baseline_m": baseline,
                          "out": args.out, "ok": ok}))
        return 0 if ok else 1
    print(f"{cal.views} views, calibration written to {args.out}")
    print(f"camera reprojection {cal.camera.rms_px:.3f} px (limit {CAMERA_LIMIT_PX})")
    print(f"projector reprojection {cal.projector.rms_px:.3f} px (limit {PROJECTOR_LIMIT_PX})")
    print(f"baseline {1000.0 * baseline:.2f} mm")
    return 0 if ok else 1


def cmd_detect(args: argparse.Namespace) -> int:
    import json

    import cv2

    image = cv2.imread(os.path.join(args.pose, "white.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(json.dumps({"error": f"no white.png in {args.pose}"}) if args.json
              else f"no white.png in {args.pose}")
        return 1
    ids, _ = calibrate.detect_corners(image, _spec(args))
    total = (args.squares_x - 1) * (args.squares_y - 1)
    print(json.dumps({"corners": len(ids), "of": total}) if args.json
          else f"{len(ids)} of {total} corners")
    return 0


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
    p.add_argument("--json", action="store_true", help="print the result as one JSON object")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("detect", help="count the board corners seen in one calibration pose")
    p.add_argument("pose", help="pose directory with white.png")
    _board_arguments(p)
    p.add_argument("--json", action="store_true", help="print the result as one JSON object")
    p.set_defaults(func=cmd_detect)

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
    p.add_argument("--limit-mm", type=float, default=1000.0 * PLATE_LIMIT_M,
                   help="plane RMS limit in mm")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("simulate", help="write synthetic calibration and scan captures")
    p.add_argument("--head", choices=sorted(HEADS), default="reference")
    p.add_argument("--out", default="data/simulated")
    p.add_argument("--pitch", type=int, default=16, help="phase shift pitch for the 1024-column virtual projector")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("motion-error", help="depth error of one phase cycle on a moving plate")
    p.add_argument("--head", choices=sorted(HEADS), default="small")
    p.add_argument("--rate", type=float, default=60.0, help="projector frame rate in Hz")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--pitch", type=int, default=16)
    p.add_argument("--speeds", default="0,1,2,5", help="sweep speeds in mm/s, comma separated")
    p.set_defaults(func=cmd_motion_error)

    p = sub.add_parser("decode-vectors", help="write a synthetic scan and its reference decode")
    p.add_argument("--out", default="data/decode-vectors")
    p.add_argument("--scene", choices=sorted(synthetic.acceptance_scenes()), default="plate")
    p.add_argument("--pitch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_decode_vectors)

    p = sub.add_parser("node-sim", help="run a stand-in capture node serving synthetic frames")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5600)
    p.add_argument("--scene", choices=sorted(synthetic.acceptance_scenes()), default="plate")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_node_sim)

    p = sub.add_parser("capture", help="run one pattern sequence on a capture node and save it")
    p.add_argument("--node", default="127.0.0.1:5600", help="host:port of the capture node")
    p.add_argument("--set", choices=("scan", "calibration"), default="scan")
    p.add_argument("--pitch", type=int, default=32)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--wait", type=float, default=0.0, help="seconds to retry while the node starts")
    p.add_argument("--out", default="data/view")
    p.set_defaults(func=cmd_capture)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
