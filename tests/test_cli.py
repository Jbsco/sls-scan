import json
import os

import cv2
import numpy as np

from slscan import calibrate, cli, geometry, patterns, stereo, synthetic


def test_calibration_patterns_command_writes_both_axes(tmp_path):
    out = str(tmp_path / "patterns")
    assert cli.main(["calibration-patterns", "--width", "64", "--height", "16", "--out", out]) == 0
    names = sorted(os.listdir(out))
    assert "white.png" in names
    assert sum(n.startswith("columns_") for n in names) == 2 * patterns.bit_planes(64)
    assert sum(n.startswith("rows_") for n in names) == 2 * patterns.bit_planes(16)
    row = cv2.imread(os.path.join(out, "rows_000.png"), cv2.IMREAD_GRAYSCALE)
    assert row.shape == (16, 64) and row[0, 0] == 0 and row[15, 0] == 255


def test_board_command_writes_detectable_board(tmp_path):
    out = str(tmp_path / "board.png")
    args = ["board", "--squares-x", "5", "--squares-y", "4", "--square-mm", "20",
            "--marker-mm", "15", "--dpi", "200", "--out", out]
    assert cli.main(args) == 0
    ids, _ = calibrate.detect_corners(cv2.imread(out, cv2.IMREAD_GRAYSCALE),
                                      calibrate.BoardSpec(5, 4, 0.02, 0.015))
    assert len(ids) == 4 * 3


def test_read_views_orders_pose_files(tmp_path):
    for pose in ("pose_001", "pose_000"):
        d = tmp_path / pose
        d.mkdir()
        cv2.imwrite(str(d / "white.png"), np.full((4, 4), 200, dtype=np.uint8))
        for axis in ("columns", "rows"):
            for i in range(4):
                cv2.imwrite(str(d / f"{axis}_{i:03d}.png"), np.full((4, 4), i, dtype=np.uint8))
    views = list(cli.read_views(str(tmp_path)))
    assert len(views) == 2
    assert views[0]["columns"].shape == (4, 4, 4)
    assert [int(v[0, 0]) for v in views[1]["rows"]] == [0, 1, 2, 3]


def test_calibration_diff_command_flags_focal_change(tmp_path):
    k = np.array([[1500.0, 0, 640], [0, 1500.0, 480], [0, 0, 1]])
    intr = calibrate.Intrinsics(1280, 960, k, np.zeros(5), 0.1)
    stereo = calibrate.Stereo(np.eye(3), np.array([-0.35, 0.0, 0.0]), 0.1)
    first = calibrate.Calibration(calibrate.BoardSpec(), intr, intr, stereo, 12)
    calibrate.save(str(tmp_path / "a.yaml"), first)
    k2 = k.copy()
    k2[0, 0] *= 1.002
    second = calibrate.Calibration(calibrate.BoardSpec(), calibrate.Intrinsics(1280, 960, k2, np.zeros(5), 0.1),
                                   intr, stereo, 12)
    calibrate.save(str(tmp_path / "b.yaml"), second)
    assert cli.main(["calibration-diff", str(tmp_path / "a.yaml"), str(tmp_path / "a.yaml")]) == 0
    assert cli.main(["calibration-diff", str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")]) == 1


def test_scan_and_measure_commands_on_a_synthetic_plate(tmp_path, rig, spec):
    captures, _ = synthetic.scan_captures(rig, synthetic.acceptance_scenes()["plate"], pitch=16)
    cli.write_scan(str(tmp_path / "plate"), captures)
    calibrate.save(str(tmp_path / "true.yaml"), synthetic.true_calibration(rig, spec))
    cloud = str(tmp_path / "plate.ply")
    assert cli.main(["scan", str(tmp_path / "plate"), "--calibration", str(tmp_path / "true.yaml"),
                     "--pitch", "16", "--out", cloud]) == 0
    assert geometry.fit_plane(geometry.read_ply(cloud))[2] < 0.0001
    assert cli.main(["measure", cloud, "--shape", "plane"]) == 0


def test_detect_command_counts_corners(tmp_path):
    pose = tmp_path / "pose_000"
    pose.mkdir()
    board = ["--squares-x", "5", "--squares-y", "4", "--square-mm", "20", "--marker-mm", "15"]
    assert cli.main(["board", *board, "--dpi", "200", "--out", str(pose / "white.png")]) == 0
    assert cli.main(["detect", str(pose), *board]) == 0


def test_detect_command_prints_json(tmp_path, capsys):
    pose = tmp_path / "pose_000"
    pose.mkdir()
    board = ["--squares-x", "5", "--squares-y", "4", "--square-mm", "20", "--marker-mm", "15"]
    cli.main(["board", *board, "--dpi", "200", "--out", str(pose / "white.png")])
    capsys.readouterr()
    assert cli.main(["detect", str(pose), *board, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"corners": 12, "of": 12}


def test_calibrate_command_prints_json(tmp_path, capsys, monkeypatch, calibration):
    monkeypatch.setattr(cli, "read_views", lambda directory: iter(()))
    monkeypatch.setattr(calibrate, "calibrate", lambda *args, **kwargs: calibration)
    out = str(tmp_path / "calibration.yaml")
    assert cli.main(["calibrate", str(tmp_path), "--json", "--out", out]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["views"] == calibration.views
    assert result["camera_rms_px"] < result["camera_limit_px"]
    assert calibrate.load(out).views == calibration.views


def test_calibrate_command_reports_too_few_views(tmp_path, capsys):
    (tmp_path / "pose_000").mkdir()
    cv2.imwrite(str(tmp_path / "pose_000" / "white.png"), np.zeros((8, 8), dtype=np.uint8))
    for axis in ("columns", "rows"):
        cv2.imwrite(str(tmp_path / "pose_000" / f"{axis}_000.png"), np.zeros((8, 8), dtype=np.uint8))
        cv2.imwrite(str(tmp_path / "pose_000" / f"{axis}_001.png"), np.zeros((8, 8), dtype=np.uint8))
    assert cli.main(["calibrate", str(tmp_path), "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_decode_vectors_match_the_reference_decode(tmp_path, rig):
    out = str(tmp_path / "vectors")
    assert cli.main(["decode-vectors", "--out", out]) == 0
    meta = json.load(open(os.path.join(out, "meta.json")))
    n = meta["width"] * meta["height"]
    gray = np.fromfile(os.path.join(out, "gray.bin"), dtype=np.uint8)
    phase = np.fromfile(os.path.join(out, "phase.bin"), dtype=np.uint8)
    assert gray.size == 2 * meta["planes"] * n and phase.size == meta["steps"] * n
    columns = np.fromfile(os.path.join(out, "columns.bin"), dtype="<f4")
    mask = np.fromfile(os.path.join(out, "mask.bin"), dtype=np.uint8).astype(bool)
    from slscan import decode
    again, again_mask = decode.decode_columns(gray.reshape(-1, meta["height"], meta["width"]),
                                              phase.reshape(-1, meta["height"], meta["width"]),
                                              meta["projector_width"], meta["pitch"])
    assert np.array_equal(again_mask.ravel(), mask)
    assert np.allclose(again.ravel()[mask], columns[mask], atol=1e-4)


def test_stereo_vectors_command_writes_consistent_files(tmp_path):
    out = tmp_path / "stereo"
    assert cli.main(["stereo-vectors", "--out", str(out), "--step", "16"]) == 0
    meta = json.loads((out / "meta.json").read_text())
    pixels, samples = meta["width"] * meta["height"], meta["samples"]
    assert samples > 0
    assert (out / "left.bin").stat().st_size == pixels
    assert (out / "right.bin").stat().st_size == pixels
    assert (out / "samples.bin").stat().st_size == 32 * samples
    assert (out / "refined.bin").stat().st_size == 8 * samples
    assert (out / "kept.bin").stat().st_size == samples
    assert (out / "left-raw.bin").stat().st_size == pixels
    assert (out / "right-raw.bin").stat().st_size == pixels
    assert set(json.loads((out / "rectify.json").read_text())) == {"left", "right"}
    direct = meta["direct_samples"]
    assert direct > 0
    assert (out / "direct-samples.bin").stat().st_size == 32 * direct
    assert (out / "direct-refined.bin").stat().st_size == 8 * direct
    assert (out / "direct-kept.bin").stat().st_size == direct
    coarse = meta["coarse"]
    assert (out / "coarse.bin").stat().st_size == 4 * coarse["width"] * coarse["height"]
    assert (out / "depth.bin").stat().st_size == 40 * meta["depth"]["points"]


def test_depth_params_command_matches_block_matching_search(tmp_path):
    out = tmp_path / "depth.json"
    assert cli.main(["depth-params", "--head", "stereo", "--out", str(out)]) == 0
    params = json.loads(out.read_text())
    assert set(params) == {"width", "height", "window", "iterations", "max_move",
                           "min_correlation", "coarse", "depth", "cameras", "preview"}
    assert set(params["cameras"]) == {"left", "right"}
    head = synthetic.stereo_head()
    cal = synthetic.true_stereo_calibration(head, synthetic.SMALL_BOARD)
    rect = stereo.rectify(cal)
    lo, count = stereo.search_range(stereo.disparity_at(rect, 0.6), stereo.disparity_at(rect, 0.4))
    assert (params["coarse"]["lo"], params["coarse"]["count"]) == (lo, count)
    lo, count = stereo.search_range(stereo.disparity_at(rect, 0.6), stereo.disparity_at(rect, 0.4),
                                    8)
    assert params["preview"] == {"scale": 8, "lo": lo, "count": count}
    assert (params["width"], params["height"]) == (cal.left.width, cal.left.height)
    assert cli.main(["depth-params", "--head", "stereo", "--out", str(out), "--json"]) == 0


def test_detect_command_reads_a_named_image(tmp_path, capsys):
    head = synthetic.stereo_head()
    r, t = synthetic.board_poses(synthetic.SMALL_BOARD, 0.5)[0]
    pair = synthetic.stereo_board_view(head, synthetic.SMALL_BOARD, r, t, np.random.default_rng(0))
    pose = tmp_path / "pose_000"
    pose.mkdir()
    for side in ("left", "right"):
        cv2.imwrite(str(pose / f"{side}.png"), pair[side])
    args = ["--squares-x", "9", "--squares-y", "6", "--square-mm", "10", "--marker-mm", "7.5"]
    assert cli.main(["detect", str(pose), *args, "--image", "right.png", "--json"]) == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["of"] == 40 and result["corners"] > 20
    assert cli.main(["detect", str(pose), *args, "--json"]) == 1


def test_fusion_vectors_command_writes_consistent_files(tmp_path):
    out = tmp_path / "fusion"
    assert cli.main(["fusion-vectors", "--out", str(out), "--frames", "3", "--step", "24"]) == 0
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["counts"]) == 3 and min(meta["counts"]) > 0
    assert (out / "frames.bin").stat().st_size == 24 * sum(meta["counts"])
    assert (out / "poses.bin").stat().st_size == 8 * 12 * 3
    assert (out / "model.bin").stat().st_size == 24 * meta["model"]


def test_tsdf_vectors_command_writes_consistent_files(tmp_path):
    out = tmp_path / "tsdf"
    assert cli.main(["tsdf-vectors", "--out", str(out), "--frames", "2", "--step", "16"]) == 0
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["counts"]) == 2 and min(meta["counts"]) > 0 and meta["surface"] > 0
    assert (out / "frames.bin").stat().st_size == 24 * sum(meta["counts"])
    assert (out / "poses.bin").stat().st_size == 8 * 12 * 2
    voxels = int(np.prod(meta["shape"]))
    assert (out / "tsdf.bin").stat().st_size == 4 * voxels
    assert (out / "weight.bin").stat().st_size == 4 * voxels
    assert (out / "surface.bin").stat().st_size == 24 * meta["surface"]
    assert (out / "surface-keys.bin").stat().st_size == 4 * meta["surface"]
    assert (out / "mesh-vertices.bin").stat().st_size == 24 * meta["mesh_vertices"]
    assert (out / "mesh-faces.bin").stat().st_size == 12 * meta["mesh_faces"]


def test_mesh_volume_command_meshes_tsdf_vectors(tmp_path, capsys):
    out = tmp_path / "tsdf"
    assert cli.main(["tsdf-vectors", "--out", str(out), "--frames", "2", "--step", "16"]) == 0
    mesh_path = tmp_path / "volume.stl"
    assert cli.main(["mesh-volume", str(out), "--out", str(mesh_path)]) == 0
    triangles = int(np.frombuffer(mesh_path.read_bytes()[80:84], "<u4")[0])
    assert triangles > 1000
    capsys.readouterr()
    assert cli.main(["mesh-volume", str(out), "--out", str(mesh_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["triangles"] == triangles and result["out"] == str(mesh_path)


def test_calibration_repeat_passes_with_a_board_filling_the_view(capsys):
    assert cli.main(["calibration-repeat", "--sessions", "3"]) == 0
    assert "largest focal length change" in capsys.readouterr().out


def test_stereo_calibration_diff_compares_both_cameras(tmp_path, capsys):
    head = synthetic.stereo_head()
    spec = synthetic.SMALL_BOARD
    first = calibrate.calibrate_stereo(synthetic.stereo_calibration_views(head, spec), spec)
    calibrate.save_stereo(str(tmp_path / "a.yaml"), first)
    assert cli.main(["calibration-diff", str(tmp_path / "a.yaml"), str(tmp_path / "a.yaml")]) == 0
    assert "right_fx change 0.000 %" in capsys.readouterr().out


def test_stereo_bench_meets_the_phase_2_limits(capsys):
    assert cli.main(["stereo-bench", "--plate-step", "24"]) == 0
    out = capsys.readouterr().out
    assert "plate" in out and "scale bar" in out


def test_replay_reference_writes_points_per_recorded_pair(tmp_path):
    head = synthetic.stereo_head()
    pattern = synthetic.random_dots(head.light.width, head.light.height, 3.0, 1.0,
                                    np.random.default_rng(0))
    images = synthetic.stereo_capture(head, synthetic.stereo_scenes()["sphere"], pattern,
                                      np.random.default_rng(1))
    frame = tmp_path / "recorded" / "frame_000"
    frame.mkdir(parents=True)
    for side in ("left", "right"):
        cv2.imwrite(str(frame / f"{side}.png"), images[side])
    params = tmp_path / "depth.json"
    assert cli.main(["depth-params", "--head", "stereo", "--step", "16", "--out", str(params)]) == 0
    assert cli.main(["replay-reference", str(tmp_path / "recorded"), "--params", str(params),
                     "--head", "stereo"]) == 0
    rows = np.fromfile(frame / "reference.bin", "<f8").reshape(-1, 5)
    assert len(rows) > 50 and np.all((rows[:, 4] > 0.4) & (rows[:, 4] < 0.6))
    preview = np.fromfile(frame / "preview.bin", "<f8").reshape(-1, 5)
    assert len(preview) > 50 and np.all((preview[:, 4] > 0.4) & (preview[:, 4] < 0.6))
    assert np.all(preview[:, :2] % 8 == 3.5)


def test_register_command_places_views_of_an_object(tmp_path):
    scene = synthetic.static_object()
    poses = synthetic.orbit_poses([0.0, 60.0, 120.0])
    rng = np.random.default_rng(3)
    paths = []
    for k, (r, t) in enumerate(poses):
        paths.append(str(tmp_path / f"view_{k}.ply"))
        geometry.write_ply(paths[-1], synthetic.point_view(scene, r, t, rng))
    out, pose_file = str(tmp_path / "merged.ply"), str(tmp_path / "poses.json")
    assert cli.main(["register", *paths, "--voxel-mm", "5", "--out", out, "--poses", pose_file]) == 0
    with open(pose_file, encoding="utf-8") as f:
        found = json.load(f)["views"]
    r0, t0 = poses[0]
    for view, (rk, tk) in zip(found, poses):
        true_r = r0 @ rk.T
        assert np.allclose(view["rotation"], true_r, atol=1e-4)
        assert np.linalg.norm(np.array(view["translation"]) - (t0 - true_r @ tk)) < 0.0001
    assert len(geometry.read_ply(out)) > 10000
