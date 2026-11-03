"""The command line: one command per pipeline stage."""

import argparse
import glob
import os
import sys

import numpy as np

from . import calibrate, decode, geometry, node, patterns, stereo, synthetic, triangulate

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


def read_stereo_views(directory: str):
    """Yield left and right board images from each pose_* directory of a capture."""
    import cv2

    poses = sorted(glob.glob(os.path.join(directory, "pose_*")))
    if not poses:
        raise FileNotFoundError(f"no pose_* directories in {directory}")
    for pose in poses:
        yield {side: cv2.imread(os.path.join(pose, f"{side}.png"), cv2.IMREAD_GRAYSCALE)
               for side in ("left", "right")}


def simulate_stereo(args: argparse.Namespace) -> int:
    import cv2

    rig = synthetic.stereo_head()
    spec = synthetic.SMALL_BOARD
    for i, view in enumerate(synthetic.stereo_calibration_views(rig, spec, args.seed)):
        pose = os.path.join(args.out, "calibration", f"pose_{i:03d}")
        os.makedirs(pose, exist_ok=True)
        for side in ("left", "right"):
            cv2.imwrite(os.path.join(pose, f"{side}.png"), view[side])
    rng = np.random.default_rng(args.seed)
    pattern = synthetic.random_dots(rig.light.width, rig.light.height, 3.0, 1.0, rng)
    scenes = synthetic.stereo_scenes()
    for name, scene in scenes.items():
        images = synthetic.stereo_capture(rig, scene, pattern, rng)
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        for side in ("left", "right"):
            cv2.imwrite(os.path.join(args.out, name, f"{side}.png"), images[side])
    bar = scenes["ball-bar"]
    print(f"synthetic stereo captures written to {args.out}")
    print(f"true sphere radius {1000.0 * scenes['sphere'][0].radius:.3f} mm, "
          f"true ball bar spacing {1000.0 * np.linalg.norm(bar[0].centre - bar[1].centre):.3f} mm")
    print("next:")
    print(f"  slscan calibrate-stereo {args.out}/calibration --squares-x {spec.squares_x} "
          f"--squares-y {spec.squares_y} --square-mm {1000.0 * spec.square_m:g} "
          f"--marker-mm {1000.0 * spec.marker_m:g} --out {args.out}/stereo.yaml")
    for name, shape in (("plate", "plane"), ("sphere", "sphere"), ("ball-bar", "ball-bar")):
        print(f"  slscan scan-stereo {args.out}/{name} --calibration {args.out}/stereo.yaml "
              f"--out {args.out}/{name}.ply && slscan measure {args.out}/{name}.ply --shape {shape}")
    return 0


def cmd_calibrate_stereo(args: argparse.Namespace) -> int:
    import json

    try:
        cal = calibrate.calibrate_stereo(read_stereo_views(args.captures), _spec(args))
    except (ValueError, FileNotFoundError) as e:
        print(json.dumps({"error": str(e)}) if args.json else f"calibration failed: {e}")
        return 1
    calibrate.save_stereo(args.out, cal)
    ok = cal.left.rms_px < CAMERA_LIMIT_PX and cal.right.rms_px < CAMERA_LIMIT_PX
    baseline = float(np.linalg.norm(cal.stereo.translation))
    if args.json:
        print(json.dumps({"views": cal.views, "left_rms_px": cal.left.rms_px,
                          "right_rms_px": cal.right.rms_px, "stereo_rms_px": cal.stereo.rms_px,
                          "camera_limit_px": CAMERA_LIMIT_PX, "baseline_m": baseline,
                          "out": args.out, "ok": ok}))
        return 0 if ok else 1
    print(f"{cal.views} views, stereo calibration written to {args.out}")
    print(f"left reprojection {cal.left.rms_px:.3f} px, right {cal.right.rms_px:.3f} px "
          f"(limit {CAMERA_LIMIT_PX}), pair {cal.stereo.rms_px:.3f} px")
    print(f"baseline {1000.0 * baseline:.2f} mm")
    return 0 if ok else 1


def cmd_scan_stereo(args: argparse.Namespace) -> int:
    import cv2

    cal = calibrate.load_stereo(args.calibration)
    images = [cv2.imread(os.path.join(args.view, f"{side}.png"), cv2.IMREAD_GRAYSCALE)
              for side in ("left", "right")]
    if any(i is None for i in images):
        print(f"left.png and right.png are needed in {args.view}")
        return 1
    points, _ = stereo.depth_points(cal, images[0], images[1], args.near, args.far,
                                    args.step, args.window, order=args.order)
    geometry.write_ply(args.out, points)
    print(f"{len(points)} points written to {args.out}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    if args.head == "stereo":
        return simulate_stereo(args)
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
                          wait=args.wait, frames=args.frames)
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


def cmd_stereo_vectors(args: argparse.Namespace) -> int:
    """Write a stereo view with its rectification, its affine matches and their NumPy
    second-order refinement as raw files for the GPU kernels."""
    import json

    head = synthetic.stereo_head()
    cal = synthetic.true_stereo_calibration(head, synthetic.SMALL_BOARD)
    pattern = synthetic.random_dots(head.light.width, head.light.height, 3.0, 1.0,
                                    np.random.default_rng(0))
    images = synthetic.stereo_capture(head, synthetic.stereo_scenes()[args.scene], pattern,
                                      np.random.default_rng(args.seed))
    rect, rl, rr, points, guesses = stereo.coarse_samples(
        cal, images["left"], images["right"], 0.4, 0.6, args.step, args.window)
    left_pts, right_pts, warps = stereo.refine(rl, rr, points, guesses, args.window,
                                               return_warps=True)
    refined, kept = stereo.second_order_matches(rl, rr, left_pts, right_pts, warps, args.window,
                                                args.iterations)
    _, _, _, block_points, block_guesses = stereo.coarse_samples(
        cal, images["left"], images["right"], 0.4, 0.6, args.step, args.window, coarse="block")
    small, lo, count = stereo.block_disparity_small(rl, rr, stereo.disparity_at(rect, 0.6),
                                                    stereo.disparity_at(rect, 0.4))
    direct, direct_kept = stereo.direct_refine(rl, rr, block_points, block_guesses, args.window)
    os.makedirs(args.out, exist_ok=True)
    images["left"].tofile(os.path.join(args.out, "left-raw.bin"))
    images["right"].tofile(os.path.join(args.out, "right-raw.bin"))
    with open(os.path.join(args.out, "rectify.json"), "w", encoding="utf-8") as f:
        json.dump(stereo.rectification_params(cal, rect), f)
    rl.tofile(os.path.join(args.out, "left.bin"))
    rr.tofile(os.path.join(args.out, "right.bin"))
    samples = np.column_stack([left_pts, right_pts, warps.reshape(-1, 4)])
    samples.astype("<f4").tofile(os.path.join(args.out, "samples.bin"))
    refined.astype("<f4").tofile(os.path.join(args.out, "refined.bin"))
    kept.astype(np.uint8).tofile(os.path.join(args.out, "kept.bin"))
    start = np.column_stack([block_points, block_guesses,
                             np.tile([1.0, 0.0, 0.0, 1.0], (len(block_points), 1))])
    start.astype("<f4").tofile(os.path.join(args.out, "direct-samples.bin"))
    direct.astype("<f4").tofile(os.path.join(args.out, "direct-refined.bin"))
    direct_kept.astype(np.uint8).tofile(os.path.join(args.out, "direct-kept.bin"))
    small.astype("<f4").tofile(os.path.join(args.out, "coarse.bin"))
    matched = block_points[direct_kept]
    cloud, in_range = stereo.triangulate(rect, matched, direct[direct_kept], 0.4, 0.6,
                                         return_kept=True)
    np.column_stack([matched[in_range], cloud]).astype("<f8").tofile(
        os.path.join(args.out, "depth.bin"))
    meta = {"width": int(rl.shape[1]), "height": int(rl.shape[0]), "window": args.window,
            "iterations": args.iterations, "samples": int(len(left_pts)),
            "direct_samples": int(len(block_points)), "max_move": 3.0,
            "min_correlation": 0.95,
            "coarse": {"width": int(small.shape[1]), "height": int(small.shape[0]), "scale": 4,
                       "block": 7, "lo": lo, "count": count, "min_correlation": 0.6,
                       "margin": 0.05, "fill": 2},
            "depth": {"near": 0.4, "far": 0.6, "step": args.step, "min_level": 6.0,
                      "points": int(len(cloud))}}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"stereo vectors of the {args.scene} scene written to {args.out}: {len(left_pts)} "
          f"affine matches, {int(kept.sum())} kept by second-order refinement, "
          f"{int(direct_kept.sum())} of {len(block_points)} kept by direct refinement")
    return 0


def cmd_depth_params(args: argparse.Namespace) -> int:
    """Write the depth frame parameters of a stereo calibration for the application."""
    import json

    if args.head:
        head = synthetic.stereo_head()
        cal = synthetic.true_stereo_calibration(head, synthetic.SMALL_BOARD)
    else:
        cal = calibrate.load_stereo(args.calibration)
    params = stereo.depth_parameters(cal, args.near, args.far, args.step, args.window)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(params, f)
    coarse = params["coarse"]
    if args.json:
        print(json.dumps({"out": args.out, "width": params["width"], "height": params["height"],
                          "lo": coarse["lo"], "count": coarse["count"]}))
        return 0
    print(f"depth parameters written to {args.out}: {params['width']} x {params['height']}, "
          f"reduced disparities {coarse['lo']} to {coarse['lo'] + coarse['count'] - 1}")
    return 0


def cmd_fusion_vectors(args: argparse.Namespace) -> int:
    """Write the depth frames of a short synthetic sweep and their NumPy tracking and fusion
    as raw files for the application."""
    import json

    from . import track

    rig = synthetic.stereo_head()
    cal = synthetic.true_stereo_calibration(rig, synthetic.SMALL_BOARD)
    pattern = synthetic.random_dots(rig.light.width, rig.light.height, 3.0, 1.0,
                                    np.random.default_rng(args.seed))
    poses = synthetic.sweep_poses(args.frames, args.yaw, args.pitch)
    frames = [stereo.depth_points(cal, im["left"], im["right"], 0.4, 0.6, args.step)[0]
              for im in synthetic.sweep_captures(rig, synthetic.sweep_scene(), poses, pattern,
                                                 args.seed)]
    voxel = args.voxel_mm / 1000.0
    tracked, model = track.fuse(frames, voxel)
    os.makedirs(args.out, exist_ok=True)
    np.concatenate(frames).astype("<f8").tofile(os.path.join(args.out, "frames.bin"))
    np.array([np.concatenate([r.ravel(), t]) for r, t in tracked]).astype("<f8").tofile(
        os.path.join(args.out, "poses.bin"))
    model.astype("<f8").tofile(os.path.join(args.out, "model.bin"))
    meta = {"counts": [int(len(f)) for f in frames], "voxel": voxel, "model": int(len(model))}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"fusion vectors written to {args.out}: {len(frames)} frames of "
          f"{min(meta['counts'])} to {max(meta['counts'])} points, {len(model)} fused points")
    return 0


def cmd_stereo_bench(args: argparse.Namespace) -> int:
    """Calibrate the synthetic metre head from board views, scan its Phase 2 artifacts in one
    view each by direct matching, and check them against the Phase 2 limits."""
    head = synthetic.metre_head()
    spec = synthetic.METRE_BOARD
    cal = calibrate.calibrate_stereo(
        synthetic.stereo_calibration_views(head, spec, args.seed, crossing_m=1.0), spec)
    print(f"metre head: {cal.views} views, left {cal.left.rms_px:.3f} px, right "
          f"{cal.right.rms_px:.3f} px, baseline "
          f"{1000.0 * np.linalg.norm(cal.stereo.translation):.2f} mm")
    pattern = synthetic.random_dots(head.light.width, head.light.height, 3.0, 1.0,
                                    np.random.default_rng(args.seed))
    scenes = synthetic.metre_scenes()

    def scan(scene, step):
        images = synthetic.stereo_capture(head, scene, pattern,
                                          np.random.default_rng(args.seed + 1))
        rect, _, _, points, refined, kept = stereo.direct_matches(
            cal, images["left"], images["right"], 0.8, 1.2, step, args.window)
        return stereo.triangulate(rect, points[kept], refined[kept], 0.8, 1.2)

    plate = scan(scenes["plate"], args.plate_step)
    _, _, rms = geometry.fit_plane(plate)
    span = np.ptp(plate[:, :2], axis=0)
    bar = scan(scenes["scale-bar"], args.bar_step)
    labels = geometry.split_two(bar)
    centres = [geometry.fit_sphere(bar[labels == i])[0] for i in (0, 1)]
    balls = scenes["scale-bar"]
    error = np.linalg.norm(centres[0] - centres[1]) - np.linalg.norm(balls[0].centre
                                                                     - balls[1].centre)
    limit = 0.0002
    ok = rms < limit and abs(error) < limit
    print(f"plate {len(plate)} points over {1000.0 * span[0]:.0f} × {1000.0 * span[1]:.0f} mm, "
          f"RMS {1000.0 * rms:.3f} mm (limit {1000.0 * limit:g} mm)")
    print(f"scale bar {len(bar)} points on two {2000.0 * balls[0].radius:.1f} mm spheres, "
          f"spacing error {1000.0 * error:+.3f} mm (limit {1000.0 * limit:g} mm)")
    return 0 if ok else 1


def cmd_replay_reference(args: argparse.Namespace) -> int:
    """Write the NumPy direct-matching points of every recorded stereo pair under a
    directory, and its preview points when the parameters have a preview, as the
    application's replay compares them."""
    import json

    import cv2

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)
    if args.head:
        cal = synthetic.true_stereo_calibration(synthetic.stereo_head(), synthetic.SMALL_BOARD)
    else:
        cal = calibrate.load_stereo(args.calibration)
    depth = params["depth"]
    frames = sorted(glob.glob(os.path.join(args.recorded, "frame_*")))
    if not frames:
        print(f"no frame_* directories in {args.recorded}")
        return 1
    for frame in frames:
        left, right = (cv2.imread(os.path.join(frame, f"{side}.png"), cv2.IMREAD_GRAYSCALE)
                       for side in ("left", "right"))
        rect, _, _, points, refined, kept = stereo.direct_matches(
            cal, left, right, depth["near"], depth["far"], depth["step"], params["window"],
            depth["min_level"], params["max_move"], params["min_correlation"])
        matched = points[kept]
        cloud, in_range = stereo.triangulate(rect, matched, refined[kept], depth["near"],
                                             depth["far"], return_kept=True)
        np.column_stack([matched[in_range], cloud]).astype("<f8").tofile(
            os.path.join(frame, "reference.bin"))
        line = f"{os.path.basename(frame)}: {len(cloud)} reference points"
        if "preview" in params:
            coarse = params["coarse"]
            rect, left_pts, right_pts = stereo.preview_matches(
                cal, left, right, depth["near"], depth["far"], params["preview"]["scale"],
                depth["min_level"], coarse["block"], coarse["min_correlation"], coarse["margin"],
                coarse["fill"])
            preview, in_range = stereo.triangulate(rect, left_pts, right_pts, depth["near"],
                                                   depth["far"], return_kept=True)
            np.column_stack([left_pts[in_range], preview]).astype("<f8").tofile(
                os.path.join(frame, "preview.bin"))
            line += f", {len(preview)} preview points"
        print(line)
    return 0


def cmd_calibration_repeat(args: argparse.Namespace) -> int:
    """Calibrate the synthetic stereo head from independent sessions of jittered board
    poses and report how far their focal lengths and baselines spread."""
    from itertools import combinations

    head = synthetic.stereo_head()
    spec = _spec(args)
    print(f"board {args.squares_x} x {args.squares_y} squares of {args.square_mm:g} mm, "
          f"{args.passes} passes of {len(synthetic.board_poses(spec, 0.5))} poses")
    sessions = []
    for seed in range(args.sessions):
        views = synthetic.stereo_calibration_views(head, spec, seed, jitter_deg=args.jitter_deg,
                                                   jitter_m=args.jitter_mm / 1000.0,
                                                   passes=args.passes)
        cal = calibrate.calibrate_stereo(views, spec)
        sessions.append(cal)
        print(f"session {seed}: {cal.views} views, left {cal.left.rms_px:.3f} px, right "
              f"{cal.right.rms_px:.3f} px, left fx {cal.left.matrix[0, 0]:.2f} px, baseline "
              f"{1000.0 * np.linalg.norm(cal.stereo.translation):.3f} mm")
    focal, baseline = 0.0, 0.0
    for a, b in combinations(sessions, 2):
        d = calibrate.compare_stereo(a, b)
        focal = max(focal, d["left_fx"], d["left_fy"], d["right_fx"], d["right_fy"])
        baseline = max(baseline, d["baseline_m"])
    reprojection = max(max(c.left.rms_px, c.right.rms_px) for c in sessions)
    ok = (focal <= args.focal_limit and 1000.0 * baseline <= args.baseline_limit_mm
          and reprojection < CAMERA_LIMIT_PX)
    print(f"largest focal length change {100.0 * focal:.3f} % (limit "
          f"{100.0 * args.focal_limit:.3f} %), largest baseline change "
          f"{1000.0 * baseline:.3f} mm (limit {args.baseline_limit_mm} mm), largest "
          f"reprojection {reprojection:.3f} px (limit {CAMERA_LIMIT_PX})")
    return 0 if ok else 1


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


def cmd_sweep_bench(args: argparse.Namespace) -> int:
    from . import track

    rig = synthetic.stereo_head()
    spec = synthetic.SMALL_BOARD
    if args.recovered:
        cal = calibrate.calibrate_stereo(synthetic.stereo_calibration_views(rig, spec), spec)
    else:
        cal = synthetic.true_stereo_calibration(rig, spec)
    rng = np.random.default_rng(args.seed)
    pattern = synthetic.random_dots(rig.light.width, rig.light.height, 3.0, 1.0, rng)
    scene = synthetic.sweep_scene()
    poses = synthetic.sweep_poses(args.frames, args.yaw, args.pitch)
    window = 25 if args.order == 2 else 21
    frames = [stereo.depth_points(cal, im["left"], im["right"], 0.4, 0.6, args.step, window,
                                  order=args.order)[0]
              for im in synthetic.sweep_captures(rig, scene, poses, pattern, args.seed)]
    result = track.sweep_bench(rig, cal, scene, poses, frames)
    print(f"{result['frames']} frames, sweep ±{args.yaw:g}° yaw and ±{args.pitch:g}° pitch, "
          f"{'recovered' if args.recovered else 'true'} calibration, order {args.order} windows, "
          f"{result['points']} fused points")
    print(f"pose error: translation up to {1e6 * result['max_translation_error_m']:.1f} µm, "
          f"rotation up to {result['max_rotation_error_deg']:.4f}°")
    print(f"ball bar spacing error {1e6 * result['ball_bar_error_m']:+.1f} µm, sphere radius errors "
          + ", ".join(f"{1e6 * e:+.1f} µm" for e in result["ball_radius_errors_m"]))
    print(f"mat flatness {1e6 * result['mat_flatness_m']:.1f} µm RMS")
    return 0


def cmd_fuse(args: argparse.Namespace) -> int:
    import cv2

    from . import track

    cal = calibrate.load_stereo(args.calibration)
    frames_dirs = sorted(glob.glob(os.path.join(args.sweep, "frame_*")))
    if not frames_dirs:
        print(f"no frame_* directories in {args.sweep}")
        return 1
    clouds = []
    for d in frames_dirs:
        left, right = (cv2.imread(os.path.join(d, f"{s}.png"), cv2.IMREAD_GRAYSCALE)
                       for s in ("left", "right"))
        clouds.append(stereo.depth_points(cal, left, right, args.near, args.far, args.step,
                                          args.window, order=args.order)[0])
    poses, model = track.fuse(clouds, args.voxel_mm / 1000.0)
    geometry.write_ply(args.out, model)
    travel = np.linalg.norm(poses[-1][1] - poses[0][1])
    print(f"{len(clouds)} frames, {sum(len(c) for c in clouds)} matched points, "
          f"{len(model)} fused points written to {args.out}; last pose {1000.0 * travel:.1f} mm "
          f"from the first")
    if len(clouds) > 1 and len(model) >= sum(len(c) for c in clouds):
        print("no frames overlapped: the sweep moved too far between frames to track")
        return 1
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

    image = cv2.imread(os.path.join(args.pose, args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(json.dumps({"error": f"no {args.image} in {args.pose}"}) if args.json
              else f"no {args.image} in {args.pose}")
        return 1
    ids, _ = calibrate.detect_corners(image, _spec(args))
    total = (args.squares_x - 1) * (args.squares_y - 1)
    print(json.dumps({"corners": len(ids), "of": total}) if args.json
          else f"{len(ids)} of {total} corners")
    return 0


def _is_stereo(path: str) -> bool:
    import yaml

    with open(path, encoding="utf-8") as f:
        return "left" in yaml.safe_load(f)


def cmd_calibration_diff(args: argparse.Namespace) -> int:
    if _is_stereo(args.first):
        d = calibrate.compare_stereo(calibrate.load_stereo(args.first),
                                     calibrate.load_stereo(args.second))
        keys = ("left_fx", "left_fy", "right_fx", "right_fy")
    else:
        d = calibrate.compare(calibrate.load(args.first), calibrate.load(args.second))
        keys = ("camera_fx", "camera_fy", "projector_fx", "projector_fy")
    ok = True
    for key in keys:
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
    p.add_argument("--image", default="white.png", help="image file in the pose directory")
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

    p = sub.add_parser("calibrate-stereo", help="calibrate two cameras from captured board poses")
    p.add_argument("captures", help="directory of pose_* directories with left.png and right.png")
    _board_arguments(p)
    p.add_argument("--out", default="data/stereo.yaml")
    p.add_argument("--json", action="store_true", help="print the result as one JSON object")
    p.set_defaults(func=cmd_calibrate_stereo)

    p = sub.add_parser("scan-stereo", help="match one stereo view into a PLY point cloud")
    p.add_argument("view", help="directory with left.png and right.png")
    p.add_argument("--calibration", default="data/stereo.yaml")
    p.add_argument("--near", type=float, default=0.4, help="nearest surface in metres")
    p.add_argument("--far", type=float, default=0.6, help="farthest surface in metres")
    p.add_argument("--step", type=int, default=8, help="sample spacing in pixels")
    p.add_argument("--window", type=int, default=25, help="matching window in pixels")
    p.add_argument("--order", type=int, choices=(1, 2), default=2,
                   help="1 for affine windows, 2 for second-order windows that follow curvature")
    p.add_argument("--out", default="data/view.ply")
    p.set_defaults(func=cmd_scan_stereo)

    p = sub.add_parser("fuse", help="match, track and fuse a captured stereo sweep")
    p.add_argument("sweep", help="directory of frame_* directories with left.png and right.png")
    p.add_argument("--calibration", default="data/stereo.yaml")
    p.add_argument("--near", type=float, default=0.4)
    p.add_argument("--far", type=float, default=0.6)
    p.add_argument("--step", type=int, default=12)
    p.add_argument("--window", type=int, default=21)
    p.add_argument("--order", type=int, choices=(1, 2), default=1,
                   help="1 for affine windows, 2 for second-order windows that follow curvature")
    p.add_argument("--voxel-mm", type=float, default=0.5)
    p.add_argument("--out", default="data/model.ply")
    p.set_defaults(func=cmd_fuse)

    p = sub.add_parser("sweep-bench", help="track and fuse a synthetic continuous sweep")
    p.add_argument("--frames", type=int, default=25)
    p.add_argument("--yaw", type=float, default=25.0, help="yaw half-range in degrees")
    p.add_argument("--pitch", type=float, default=8.0, help="pitch amplitude in degrees")
    p.add_argument("--step", type=int, default=12, help="sample spacing in pixels")
    p.add_argument("--recovered", action="store_true", help="use a calibration recovered from board views")
    p.add_argument("--order", type=int, choices=(1, 2), default=1,
                   help="1 for affine windows, 2 for second-order windows that follow curvature")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_sweep_bench)

    p = sub.add_parser("measure", help="fit an acceptance artifact to a point cloud")
    p.add_argument("cloud")
    p.add_argument("--shape", choices=("plane", "sphere", "ball-bar"), required=True)
    p.add_argument("--limit-mm", type=float, default=1000.0 * PLATE_LIMIT_M,
                   help="plane RMS limit in mm")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("simulate", help="write synthetic calibration and scan captures")
    p.add_argument("--head", choices=sorted(HEADS) + ["stereo"], default="reference")
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

    p = sub.add_parser("depth-params",
                       help="write the depth frame parameters of a stereo calibration for "
                            "the application")
    p.add_argument("--calibration", default="data/stereo.yaml")
    p.add_argument("--head", choices=("stereo",),
                   help="use the synthetic head's true calibration instead")
    p.add_argument("--near", type=float, default=0.4, help="nearest surface in metres")
    p.add_argument("--far", type=float, default=0.6, help="farthest surface in metres")
    p.add_argument("--step", type=int, default=8, help="sample spacing in pixels")
    p.add_argument("--window", type=int, default=25, help="matching window in pixels")
    p.add_argument("--out", default="data/depth.json")
    p.add_argument("--json", action="store_true", help="print the result as one JSON object")
    p.set_defaults(func=cmd_depth_params)

    p = sub.add_parser("stereo-bench",
                       help="check the synthetic metre head against the Phase 2 limits in "
                            "one view")
    p.add_argument("--window", type=int, default=21, help="matching window in pixels")
    p.add_argument("--plate-step", type=int, default=8, help="sample spacing on the plate")
    p.add_argument("--bar-step", type=int, default=4, help="sample spacing on the scale bar")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_stereo_bench)

    p = sub.add_parser("replay-reference",
                       help="write the reference points of recorded stereo pairs for the "
                            "application's replay")
    p.add_argument("recorded", help="directory of frame_* directories with left.png and right.png")
    p.add_argument("--params", default="data/depth.json", help="depth frame parameters")
    p.add_argument("--calibration", default="data/stereo.yaml")
    p.add_argument("--head", choices=("stereo",),
                   help="use the synthetic head's true calibration instead")
    p.set_defaults(func=cmd_replay_reference)

    p = sub.add_parser("calibration-repeat",
                       help="calibrate the synthetic stereo head from independent sessions "
                            "and report their spread")
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--squares-x", type=int, default=11)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-mm", type=float, default=12.0)
    p.add_argument("--marker-mm", type=float, default=9.0)
    p.add_argument("--dictionary", default="DICT_5X5_100")
    p.add_argument("--passes", type=int, default=2, help="times through the pose table")
    p.add_argument("--jitter-deg", type=float, default=5.0,
                   help="largest change of each board angle per session")
    p.add_argument("--jitter-mm", type=float, default=10.0,
                   help="largest change of each board position per session")
    p.add_argument("--focal-limit", type=float, default=0.001, help="relative focal length change")
    p.add_argument("--baseline-limit-mm", type=float, default=0.2)
    p.set_defaults(func=cmd_calibration_repeat)

    p = sub.add_parser("fusion-vectors",
                       help="write the depth frames of a short synthetic sweep and their "
                            "reference tracking and fusion")
    p.add_argument("--out", default="data/fusion-vectors")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--yaw", type=float, default=5.0, help="yaw amplitude in degrees")
    p.add_argument("--pitch", type=float, default=2.0, help="pitch amplitude in degrees")
    p.add_argument("--step", type=int, default=16, help="sample spacing in pixels")
    p.add_argument("--voxel-mm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fusion_vectors)

    p = sub.add_parser("stereo-vectors",
                       help="write a stereo view with its rectification, its affine matches "
                            "and their reference second-order refinement")
    p.add_argument("--out", default="data/stereo-vectors")
    p.add_argument("--scene", choices=("plate", "sphere", "ball-bar"), default="sphere")
    p.add_argument("--step", type=int, default=8, help="sample spacing in pixels")
    p.add_argument("--window", type=int, default=25, help="matching window in pixels")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_stereo_vectors)

    p = sub.add_parser("node-sim", help="run a stand-in capture node serving synthetic frames")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5600)
    p.add_argument("--scene", choices=sorted(synthetic.acceptance_scenes()), default="plate")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_node_sim)

    p = sub.add_parser("capture", help="run one pattern sequence on a capture node and save it")
    p.add_argument("--node", default="127.0.0.1:5600", help="host:port of the capture node")
    p.add_argument("--set", choices=("scan", "calibration", "stereo", "stereo-calibration"),
                   default="scan")
    p.add_argument("--frames", type=int, default=1, help="frame pairs in a stereo set")
    p.add_argument("--pitch", type=int, default=32)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--wait", type=float, default=0.0, help="seconds to retry while the node starts")
    p.add_argument("--out", default="data/view")
    p.set_defaults(func=cmd_capture)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
