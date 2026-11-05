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


def cmd_mesh_volume(args: argparse.Namespace) -> int:
    """Mesh a TSDF volume saved as meta.json, tsdf.bin and weight.bin, as the application
    and tsdf-vectors write it."""
    import json

    from . import mesh, tsdf

    with open(os.path.join(args.volume, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    volume = tsdf.Volume.from_shape(meta["lower"], meta["shape"], meta["voxel"],
                                    meta["truncation"])
    shape = volume.shape
    volume.tsdf[:] = np.fromfile(os.path.join(args.volume, "tsdf.bin"), "<f4").reshape(shape)
    volume.weight[:] = np.fromfile(os.path.join(args.volume, "weight.bin"),
                                   "<f4").reshape(shape)
    vertices, faces = mesh.surface_nets(volume)
    closed = 0
    if args.fill_mm:
        vertices, faces, closed = mesh.fill_holes(vertices, faces, args.fill_mm / 1000.0)
        print(f"{closed} holes up to {args.fill_mm:g} mm around filled")
    if args.faces:
        vertices, faces = mesh.decimate(vertices, faces, args.faces)
    if not len(faces):
        if args.json:
            print(json.dumps({"error": f"{args.volume}: the volume has no surface"}))
        else:
            print(f"{args.volume}: the volume has no surface")
        return 1
    mesh.write_mesh(args.out, vertices, faces)
    print(f"{len(vertices)} vertices and {len(faces)} triangles written to {args.out}")
    if args.json:
        print(json.dumps({"out": args.out, "vertices": int(len(vertices)),
                          "triangles": int(len(faces)), "holes": int(closed)}))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Fit datum spheres in a PLY cloud or mesh, align them to their measured centres and
    report residuals and scale bar errors."""
    import json

    from . import report

    points = geometry.read_ply(args.cloud)
    result = report.measure(points, report.load_datums(args.datums))
    for name, datum in result["datums"].items():
        print(f"datum {name}: residual {1e6 * datum['residual_m']:.1f} µm, "
              f"radius {1e6 * datum['radius_error_m']:+.1f} µm")
    for name, error in result["scale_bars_m"].items():
        print(f"scale bar {name}: {1e6 * error:+.1f} µm")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        print(f"report written to {args.out}")
    if args.aligned:
        from . import mesh

        moved = points @ np.array(result["rotation"]).T + np.array(result["translation_m"])
        faces = geometry.read_ply_faces(args.cloud)
        if len(faces):
            mesh.write_mesh(args.aligned, moved, faces)
        else:
            geometry.write_ply(args.aligned, moved)
        print(f"{'mesh' if len(faces) else 'points'} in the object's frame written to {args.aligned}")
    worst = max(d["residual_m"] for d in result["datums"].values())
    return 0 if worst <= args.limit_mm / 1000.0 else 1


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
                          wait=args.wait, frames=args.frames, dark_every=args.dark_every)
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


def cmd_scan_params(args: argparse.Namespace) -> int:
    """Write a calibration and decode settings as JSON for the application's scans."""
    import json

    params = triangulate.scan_params(calibrate.load(args.calibration), args.pitch, args.steps)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=1)
    print(f"scan parameters written to {args.out}")
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
                                                args.iterations, exit_px=args.exit_px)
    _, _, _, block_points, block_guesses = stereo.coarse_samples(
        cal, images["left"], images["right"], 0.4, 0.6, args.step, args.window, coarse="block")
    small, lo, count = stereo.block_disparity_small(rl, rr, stereo.disparity_at(rect, 0.6),
                                                    stereo.disparity_at(rect, 0.4))
    direct, direct_kept = stereo.direct_refine(rl, rr, block_points, block_guesses, args.window,
                                               exit_px=args.exit_px)
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
            "iterations": args.iterations, "exit_px": args.exit_px,
            "samples": int(len(left_pts)),
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
    params = stereo.depth_parameters(cal, args.near, args.far, args.step, args.window,
                                     exit_px=args.exit_px)
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


def cmd_dot_fusion_vectors(args: argparse.Namespace) -> int:
    """Write point frames and reference dot frames of a slide along a smooth panel, with
    their NumPy tracking and fusion, for the application's fusion-check. With dark_every,
    every that many-th frame from the first is a light-off pair with dots and no points,
    and lit frames have no dots."""
    import json

    from . import track

    clouds, dot_frames, _ = synthetic.panel_frames(np.random.default_rng(args.seed), args.frames,
                                                   args.slide_mm / 1000.0,
                                                   args.dot_pitch_mm / 1000.0)
    dark = [args.dark_every > 0 and k % args.dark_every == 0 for k in range(len(clouds))]
    if args.dark_every > 0:
        clouds = [np.empty((0, 3)) if d else c for c, d in zip(clouds, dark)]
        dot_frames = [f if d else np.empty((0, 3)) for f, d in zip(dot_frames, dark)]
    voxel = args.voxel_mm / 1000.0
    tracked, model = track.fuse([None if d else c for c, d in zip(clouds, dark)], voxel,
                                dots=dot_frames)
    os.makedirs(args.out, exist_ok=True)
    np.concatenate(clouds).astype("<f8").tofile(os.path.join(args.out, "frames.bin"))
    np.concatenate(dot_frames).astype("<f8").tofile(os.path.join(args.out, "dots.bin"))
    np.array([np.concatenate([r.ravel(), t]) for r, t in tracked]).astype("<f8").tofile(
        os.path.join(args.out, "poses.bin"))
    model.astype("<f8").tofile(os.path.join(args.out, "model.bin"))
    meta = {"counts": [int(len(c)) for c in clouds],
            "dot_counts": [int(len(d)) for d in dot_frames], "dark": dark, "voxel": voxel,
            "model": int(len(model))}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"dot fusion vectors written to {args.out}: {len(clouds)} frames, {sum(dark)} light-off, "
          f"up to {max(meta['dot_counts'])} dots each, {len(model)} fused points")
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


def cmd_dot_vectors(args: argparse.Namespace) -> int:
    """Write a light-off stereo pair of a dotted plate, its rectification, the points of the
    pattern-lit pair beside it and the dots NumPy finds, for the application's dot-check."""
    import json

    head = synthetic.stereo_head()
    cal = synthetic.true_stereo_calibration(head, synthetic.SMALL_BOARD)
    pattern = synthetic.random_dots(head.light.width, head.light.height, 3.0, 1.0,
                                    np.random.default_rng(0))
    us, vs = np.meshgrid(np.arange(-0.14, 0.141, 0.04), np.arange(-0.09, 0.091, 0.04))
    plate = synthetic.Plate(np.array([0.013, 0.007, 0.5]), synthetic.rotation(8.0, 0.0), 0.16,
                            0.11, 0.8, tuple(zip(us.ravel(), vs.ravel())), 0.002)
    lit = synthetic.stereo_capture(head, [plate], pattern, np.random.default_rng(args.seed))
    surface, rect = stereo.depth_points(cal, lit["left"], lit["right"], args.near, args.far,
                                        step=12)
    room = synthetic.stereo_room_view(head, [plate], np.random.default_rng(args.seed + 1))
    dots = stereo.find_dots(cal, room["left"], room["right"], surface, args.near, args.far)
    os.makedirs(args.out, exist_ok=True)
    room["left"].tofile(os.path.join(args.out, "left-raw.bin"))
    room["right"].tofile(os.path.join(args.out, "right-raw.bin"))
    surface.astype("<f8").tofile(os.path.join(args.out, "surface.bin"))
    dots.astype("<f8").tofile(os.path.join(args.out, "dots.bin"))
    with open(os.path.join(args.out, "rectify.json"), "w", encoding="utf-8") as f:
        json.dump(stereo.rectification_params(cal, rect), f)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"width": cal.left.width, "height": cal.left.height, "near": args.near,
                   "far": args.far, "surface": int(len(surface)), "dots": int(len(dots))}, f)
    print(f"dot vectors written to {args.out}: {len(dots)} dots, {len(surface)} surface points")
    return 0


def cmd_register_vectors(args: argparse.Namespace) -> int:
    """Write ray-cast views of the static test object as PLY files, with the poses the NumPy
    registration finds and the true poses, for the application's register-check."""
    import json

    from . import register

    scene = synthetic.static_object()
    poses = synthetic.orbit_poses(args.azimuths)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    names, views = [], []
    for k, (r, t) in enumerate(poses):
        names.append(f"view_{k}.ply")
        path = os.path.join(args.out, names[-1])
        geometry.write_ply(path, synthetic.point_view(scene, r, t, rng))
        views.append(geometry.read_ply(path))
    voxel = args.voxel_mm / 1000.0
    found, records = register.register_views(views, voxel, args.rounds)
    r0, t0 = poses[0]
    truth = []
    for rk, tk in poses:
        true_r = r0 @ rk.T
        truth.append({"rotation": true_r.tolist(), "translation": (t0 - true_r @ tk).tolist()})
    with open(os.path.join(args.out, "register.json"), "w", encoding="utf-8") as f:
        json.dump({"voxel": voxel, "rounds": args.rounds, "views": names,
                   "numpy": [{"rotation": r.tolist(), "translation": t.tolist()}
                             for r, t in found],
                   "truth": truth, "overlaps": [rec["overlap"] for rec in records]},
                  f, indent=1)
    print(f"{len(views)} views written to {args.out}")
    return 0


def cmd_tsdf_vectors(args: argparse.Namespace) -> int:
    """Write the depth frames of a short synthetic sweep, their tracked poses, the left
    camera's rectification and the NumPy TSDF volume they fuse into, for the application."""
    import json

    from . import mesh, track, tsdf

    rig = synthetic.stereo_head()
    cal = synthetic.true_stereo_calibration(rig, synthetic.SMALL_BOARD)
    pattern = synthetic.random_dots(rig.light.width, rig.light.height, 3.0, 1.0,
                                    np.random.default_rng(args.seed))
    poses = synthetic.sweep_poses(args.frames, args.yaw, args.pitch)
    window = 21
    frames, rect = [], None
    for im in synthetic.sweep_captures(rig, synthetic.sweep_scene(), poses, pattern, args.seed):
        points, rect = stereo.depth_points(cal, im["left"], im["right"], 0.4, 0.6, args.step,
                                           window)
        frames.append(points)
    tracked, _ = track.fuse(frames)
    volume, _ = tsdf.fuse(frames, tracked, rect, args.step, window, args.voxel_mm / 1000.0,
                          args.truncation_mm / 1000.0)
    os.makedirs(args.out, exist_ok=True)
    np.concatenate(frames).astype("<f8").tofile(os.path.join(args.out, "frames.bin"))
    np.array([np.concatenate([r.ravel(), t]) for r, t in tracked]).astype("<f8").tofile(
        os.path.join(args.out, "poses.bin"))
    volume.tsdf.astype("<f4").tofile(os.path.join(args.out, "tsdf.bin"))
    volume.weight.astype("<f4").tofile(os.path.join(args.out, "weight.bin"))
    surface, keys = tsdf.surface(volume, return_keys=True)
    surface.astype("<f8").tofile(os.path.join(args.out, "surface.bin"))
    keys.astype("<u4").tofile(os.path.join(args.out, "surface-keys.bin"))
    vertices, faces = mesh.surface_nets(volume)
    vertices.astype("<f8").tofile(os.path.join(args.out, "mesh-vertices.bin"))
    faces.astype("<u4").tofile(os.path.join(args.out, "mesh-faces.bin"))
    meta = {"counts": [int(len(f)) for f in frames], "step": args.step, "window": window,
            "width": rect.size[0], "height": rect.size[1],
            "rotation": rect.left_rotation.ravel().tolist(),
            "projection": rect.left_projection.ravel().tolist(),
            "lower": volume.lower.tolist(), "shape": list(volume.shape),
            "voxel": volume.voxel, "truncation": volume.truncation, "surface": int(len(surface)),
            "mesh_vertices": int(len(vertices)), "mesh_faces": int(len(faces))}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"TSDF vectors written to {args.out}: {len(frames)} frames into "
          f"{' x '.join(str(n) for n in volume.shape)} voxels, {len(surface)} surface points")
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
            depth["min_level"], params["max_move"], params["min_correlation"],
            exit_px=params.get("exit_px", 1e-5))
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
    if args.loop:
        poses = synthetic.loop_poses(args.frames, args.yaw, args.pitch)
    else:
        poses = synthetic.sweep_poses(args.frames, args.yaw, args.pitch)
    window = 25 if args.order == 2 else 21
    frames, rect = [], None
    for im in synthetic.sweep_captures(rig, scene, poses, pattern, args.seed):
        points, rect = stereo.depth_points(cal, im["left"], im["right"], 0.4, 0.6, args.step,
                                           window, order=args.order)
        frames.append(points)
    voxel = args.voxel_mm / 1000.0
    surface = None
    if args.tsdf:
        from . import mesh, tsdf

        def surface(tracked):
            volume, points = tsdf.fuse(frames, tracked, rect, args.step, window, voxel,
                                       args.truncation_mm / 1000.0)
            if not args.mesh:
                return points
            vertices, faces = mesh.surface_nets(volume)
            if args.fill_mm:
                vertices, faces, closed = mesh.fill_holes(vertices, faces,
                                                          args.fill_mm / 1000.0)
                print(f"{closed} holes up to {args.fill_mm:g} mm around filled")
            if args.faces:
                vertices, faces = mesh.decimate(vertices, faces, args.faces)
            mesh.write_mesh(args.mesh, vertices, faces)
            print(f"{len(vertices)} vertices and {len(faces)} triangles written to {args.mesh}")
            return vertices
    result = track.sweep_bench(rig, cal, scene, poses, frames, voxel, surface)
    if args.datums:
        import json

        r0, t0 = track.left_camera_pose(rig, *poses[0])
        named = {"ball-a": scene["ball-bar"][0], "ball-b": scene["ball-bar"][1],
                 "sphere": scene["sphere"]}
        datums = {"spheres": {name: {"centre": ball.centre.tolist(), "radius": ball.radius,
                                     "seed": (r0 @ ball.centre + t0).tolist()}
                              for name, ball in named.items()},
                  "scale_bars": [["ball-a", "ball-b"]]}
        with open(args.datums, "w", encoding="utf-8") as f:
            json.dump(datums, f, indent=1)
        print(f"datums written to {args.datums}")
    print(f"{result['frames']} frames, sweep ±{args.yaw:g}° yaw and ±{args.pitch:g}° pitch, "
          f"{'recovered' if args.recovered else 'true'} calibration, order {args.order} windows, "
          f"{result['points']} {'mesh vertices' if args.tsdf and args.mesh else 'TSDF surface points' if args.tsdf else 'fused points'}")
    print(f"pose error: translation up to {1e6 * result['max_translation_error_m']:.1f} µm, "
          f"rotation up to {result['max_rotation_error_deg']:.4f}°; last frame "
          f"{1e6 * result['last_translation_error_m']:.1f} µm, "
          f"{result['last_rotation_error_deg']:.4f}°")
    print(f"ball bar spacing error {1e6 * result['ball_bar_error_m']:+.1f} µm, sphere radius errors "
          + ", ".join(f"{1e6 * e:+.1f} µm" for e in result["ball_radius_errors_m"]))
    print(f"mat flatness {1e6 * result['mat_flatness_m']:.1f} µm RMS")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    import json

    from . import register, track

    views = [geometry.read_ply(path) for path in args.views]
    poses, records = register.register_views(views, args.voxel_mm / 1000.0, args.rounds)
    merged = np.vstack([v @ r.T + t for v, (r, t) in zip(views, poses)])
    if args.merge_voxel_mm > 0:
        merged = track.voxel_average(merged, args.merge_voxel_mm / 1000.0)
    geometry.write_ply(args.out, merged)
    overlaps = [None] + [record["overlap"] for record in records]
    with open(args.poses, "w", encoding="utf-8") as f:
        json.dump({"frame": args.views[0],
                   "views": [{"file": path, "rotation": r.tolist(), "translation": t.tolist(),
                              "overlap_with_previous": overlap}
                             for path, (r, t), overlap in zip(args.views, poses, overlaps)]},
                  f, indent=1)
    for path, overlap in zip(args.views[1:], overlaps[1:]):
        print(f"{path}: overlap with the view before {overlap:.2f}")
    print(f"{len(merged)} points written to {args.out}, poses in the frame of {args.views[0]} "
          f"written to {args.poses}")
    return 0


def cmd_view_bench(args: argparse.Namespace) -> int:
    import time

    from . import register, track

    scene = synthetic.metre_object()
    shapes = scene["part"] + [ball for bar in scene["scale-bars"] for ball in bar]
    distance = args.distance or (0.8 if args.stereo else 1.2)
    height = args.height or (1.0 if args.stereo else 0.9)
    poses = synthetic.orbit_poses(np.arange(args.views) * 360.0 / args.views, distance, height,
                                  (0.0, 0.0, 0.1))
    rng = np.random.default_rng(args.seed)
    if args.stereo:
        head = synthetic.metre_head()
        spec = synthetic.METRE_BOARD
        if args.recovered:
            cal = calibrate.calibrate_stereo(
                synthetic.stereo_calibration_views(head, spec, args.seed, crossing_m=1.0), spec)
        else:
            cal = synthetic.true_stereo_calibration(head, spec)
        pattern = synthetic.random_dots(head.light.width, head.light.height, 3.0, 1.0, rng)
        views = []
        for r, t in poses:
            images = synthetic.stereo_capture(head, synthetic.transform_scene(shapes, r, t),
                                              pattern, rng)
            rect, _, _, points, refined, kept = stereo.direct_matches(
                cal, images["left"], images["right"], args.near, args.far, args.step, 21)
            cloud = stereo.triangulate(rect, points[kept], refined[kept], args.near, args.far)
            views.append(stereo.drop_isolated(cloud, args.step, rect.left_projection[0, 0]))
        poses = [track.left_camera_pose(head, r, t) for r, t in poses]
        source = (f"metre head stereo views, step {args.step}, "
                  f"{'recovered' if args.recovered else 'true'} calibration")
    else:
        views = [synthetic.point_view(shapes, r, t, rng, 0.42, 440, args.noise_um * 1e-6)
                 for r, t in poses]
        source = f"ray-cast views, {args.noise_um:g} µm noise"
    datums = None
    if args.datums:
        import json

        names = [f"bar-{i + 1}{side}" for i in range(len(scene["scale-bars"])) for side in "ab"]
        balls = [ball for bar in scene["scale-bars"] for ball in bar]
        datums = {"spheres": {name: {"centre": ball.centre.tolist(), "radius": ball.radius}
                              for name, ball in zip(names, balls)},
                  "scale_bars": [names[i:i + 2] for i in range(0, len(names), 2)]}
        with open(args.datums, "w", encoding="utf-8") as f:
            json.dump(datums, f, indent=1)
        print(f"datums written to {args.datums}")
    start = time.time()
    result = register.view_bench(scene, poses, views, args.voxel_mm / 1000.0, args.rounds, datums)
    print(f"{result['views']} {source}, {distance:g} m out and {height:g} m up, "
          f"{np.mean([len(v) for v in views]):.0f} points each, {args.voxel_mm:g} mm voxels, "
          f"{args.rounds} rounds, registered in {time.time() - start:.0f} s")
    print("overlap with the view before: "
          + ", ".join(f"{o:.2f}" for o in result["overlaps"]))
    print(f"pose error: translation up to {1e3 * result['max_translation_error_m']:.3f} mm, "
          f"rotation up to {result['max_rotation_error_deg']:.4f}°")
    placed = sum(e < 0.01 for e in result["translation_errors_m"])
    print(f"{placed} of {result['views']} views placed within 10 mm")
    print("scale bar errors "
          + ", ".join(f"{1e3 * e:+.3f} mm" for e in result["scale_bar_errors_m"])
          + " (limit 0.5 mm)")
    if datums:
        print("datums found without seeds, residuals "
              + ", ".join(f"{n} {1e3 * d['residual_m']:.3f} mm"
                          for n, d in result["datums"]["datums"].items()))
    ok = placed == result["views"] and max(abs(e) for e in result["scale_bar_errors_m"]) < 0.0005
    return 0 if ok else 1


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
    p.add_argument("--loop", action="store_true",
                   help="sweep a closed loop that ends at the first pose")
    p.add_argument("--voxel-mm", type=float, default=0.5, help="fusion voxel size")
    p.add_argument("--tsdf", action="store_true",
                   help="measure a TSDF surface in place of the fused points")
    p.add_argument("--truncation-mm", type=float, default=1.0, help="TSDF truncation distance")
    p.add_argument("--mesh", help="with --tsdf, write the surface as a mesh (.ply, .obj or .stl) "
                                  "and measure its vertices")
    p.add_argument("--datums", help="write the scene's spheres as a datum file for report")
    p.add_argument("--fill-mm", type=float, default=0.0,
                   help="with --mesh, close holes up to this perimeter")
    p.add_argument("--faces", type=int, default=0,
                   help="with --mesh, decimate the mesh to at most this many triangles")
    p.set_defaults(func=cmd_sweep_bench)

    p = sub.add_parser("register", help="register view clouds from geometry and merge them")
    p.add_argument("views", nargs="+", help="view clouds (PLY) in order around the object, "
                                          "each in its own camera frame")
    p.add_argument("--voxel-mm", type=float, default=10.0,
                   help="coarse match voxel size, about 1 to 2%% of the object's size")
    p.add_argument("--rounds", type=int, default=1, help="ICP rounds against all other views")
    p.add_argument("--merge-voxel-mm", type=float, default=0.5,
                   help="average the merged cloud over voxels of this size, 0 to keep all points")
    p.add_argument("--out", default="data/merged.ply")
    p.add_argument("--poses", default="data/poses.json")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("view-bench", help="register synthetic static views of a metre object")
    p.add_argument("--views", type=int, default=10, help="views evenly spaced around the object")
    p.add_argument("--distance", type=float, default=0.0,
                   help="camera distance out in metres, 1.2 or with --stereo 0.8 by default")
    p.add_argument("--height", type=float, default=0.0,
                   help="camera height in metres, 0.9 or with --stereo 1.0 by default")
    p.add_argument("--stereo", action="store_true",
                   help="match stereo pairs rendered for the metre head in place of ray-cast "
                        "points")
    p.add_argument("--recovered", action="store_true",
                   help="with --stereo, use a calibration recovered from board views")
    p.add_argument("--step", type=int, default=8, help="with --stereo, sample spacing in pixels")
    p.add_argument("--near", type=float, default=0.8, help="with --stereo, nearest depth in metres")
    p.add_argument("--far", type=float, default=1.8, help="with --stereo, farthest depth in metres")
    p.add_argument("--noise-um", type=float, default=40.0, help="point noise per coordinate")
    p.add_argument("--voxel-mm", type=float, default=10.0, help="coarse match voxel size")
    p.add_argument("--rounds", type=int, default=1, help="ICP rounds against all other views")
    p.add_argument("--datums", help="write the scale bar spheres as a datum file without seeds, "
                                    "then find and align them in the merged views")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_view_bench)

    p = sub.add_parser("mesh-volume", help="mesh a saved TSDF volume")
    p.add_argument("volume", help="directory with meta.json, tsdf.bin and weight.bin")
    p.add_argument("--out", required=True, help="mesh file, .ply, .obj or .stl")
    p.add_argument("--fill-mm", type=float, default=0.0, help="close holes up to this perimeter")
    p.add_argument("--faces", type=int, default=0, help="decimate to at most this many triangles")
    p.add_argument("--json", action="store_true", help="end with one JSON line for the application")
    p.set_defaults(func=cmd_mesh_volume)

    p = sub.add_parser("report", help="align datum spheres in a cloud or mesh to their "
                                      "measured centres and report residuals")
    p.add_argument("cloud", help="PLY points or mesh")
    p.add_argument("--datums", required=True, help="datum file of spheres and scale bars")
    p.add_argument("--out", help="write the report as JSON")
    p.add_argument("--aligned", help="write the cloud or mesh in the object's frame")
    p.add_argument("--limit-mm", type=float, default=1.0, help="largest datum residual allowed")
    p.set_defaults(func=cmd_report)

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

    p = sub.add_parser("scan-params", help="write a calibration and decode settings as JSON "
                                           "for the application's scans")
    p.add_argument("--calibration", required=True)
    p.add_argument("--pitch", type=int, default=16, help="phase shift pitch in projector pixels")
    p.add_argument("--steps", type=int, default=4, help="phase steps")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_scan_params)

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
    p.add_argument("--step", type=int, default=12, help="sample spacing in pixels")
    p.add_argument("--window", type=int, default=25, help="matching window in pixels")
    p.add_argument("--exit-px", type=float, default=1e-4,
                   help="refinement stops when a step moves the match less than this")
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

    p = sub.add_parser("dot-vectors",
                       help="write a light-off pair of a dotted plate with the dots NumPy finds")
    p.add_argument("--out", default="data/dot-vectors")
    p.add_argument("--near", type=float, default=0.45, help="nearest surface in metres")
    p.add_argument("--far", type=float, default=0.55, help="farthest surface in metres")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_dot_vectors)

    p = sub.add_parser("register-vectors",
                       help="write views of the static test object with NumPy and true poses")
    p.add_argument("--out", default="data/register-vectors")
    p.add_argument("--azimuths", type=float, nargs="+", default=[0.0, 40.0, 80.0, 120.0],
                   help="camera azimuths in degrees around the object")
    p.add_argument("--voxel-mm", type=float, default=5.0, help="coarse match voxel size")
    p.add_argument("--rounds", type=int, default=1, help="ICP rounds against all other views")
    p.add_argument("--seed", type=int, default=3)
    p.set_defaults(func=cmd_register_vectors)

    p = sub.add_parser("tsdf-vectors",
                       help="write the depth frames of a short synthetic sweep, their poses "
                            "and the reference TSDF volume")
    p.add_argument("--out", default="data/tsdf-vectors")
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--yaw", type=float, default=4.0, help="yaw amplitude in degrees")
    p.add_argument("--pitch", type=float, default=2.0, help="pitch amplitude in degrees")
    p.add_argument("--step", type=int, default=8, help="sample spacing in pixels")
    p.add_argument("--voxel-mm", type=float, default=1.0)
    p.add_argument("--truncation-mm", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_tsdf_vectors)

    p = sub.add_parser("dot-fusion-vectors",
                       help="write point and dot frames of a slide along a smooth panel and "
                            "their reference tracking with dots")
    p.add_argument("--out", default="data/dot-fusion-vectors")
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--slide-mm", type=float, default=120.0, help="slide length")
    p.add_argument("--dot-pitch-mm", type=float, default=100.0, help="dot spacing")
    p.add_argument("--voxel-mm", type=float, default=6.0)
    p.add_argument("--dark-every", type=int, default=0,
                   help="make every this many-th frame from the first a light-off pair")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_dot_fusion_vectors)

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
    p.add_argument("--exit-px", type=float, default=1e-4,
                   help="refinement stops when a step moves the match less than this")
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
    p.add_argument("--dark-every", type=int, default=0,
                   help="take every this many-th stereo pair with the pattern light off")
    p.add_argument("--pitch", type=int, default=32)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--wait", type=float, default=0.0, help="seconds to retry while the node starts")
    p.add_argument("--out", default="data/view")
    p.set_defaults(func=cmd_capture)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
