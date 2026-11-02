import json
import os

import cv2
import numpy as np

from slscan import calibrate, cli, geometry, patterns, synthetic


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
