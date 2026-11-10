import json
import socket
import threading

import cv2
import numpy as np
import pytest

from slscan import cli, decode, geometry, node, synthetic, triangulate


@pytest.fixture(scope="module")
def stand_in():
    server = node.StandIn(("127.0.0.1", 0), scene="plate")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address
    server.shutdown()
    server.server_close()


def test_frame_payload_round_trip():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    header, back = node.parse_frame(node.frame_payload(7, 2, 1, 123456789, image))
    assert header == {"sequence": 7, "index": 2, "exposure": 1, "timestamp_ns": 123456789}
    assert np.array_equal(back, image)


def test_pattern_names_follow_capture_layout():
    names = node.pattern_names("scan", 1024, 768, 4)
    assert names[0] == "gray_000" and names[-1] == "phase_003" and len(names) == 24
    names = node.pattern_names("calibration", 1024, 768, 4)
    assert names[0] == "white" and names[1] == "columns_000" and names[-1] == "rows_019"


def test_scan_sequence_over_the_protocol(stand_in, tmp_path, rig, spec):
    host, port = stand_in
    result = node.capture(host, port, "scan", str(tmp_path / "plate"), pitch=16)
    assert result["missing"] == [] and result["in_order"]
    assert np.all(np.diff(result["timestamps_ns"]) == node.FRAME_PERIOD_NS)
    view = cli.read_scan(str(tmp_path / "plate"))
    columns, mask = decode.decode_columns(view["gray"], view["phase"], rig.projector.width, 16)
    points, _ = triangulate.triangulate(columns, mask, synthetic.true_calibration(rig, spec))
    assert geometry.fit_plane(points)[2] < 0.0001


def test_pattern_names_take_each_pattern_at_every_exposure():
    names = node.pattern_names("scan", 1024, 768, 4, exposures=3)
    assert names[:4] == ["exposure_000/gray_000", "exposure_001/gray_000",
                         "exposure_002/gray_000", "exposure_000/gray_001"]
    assert names[-1] == "exposure_002/phase_003" and len(names) == 72


def test_scan_at_several_exposures_over_the_protocol(tmp_path, rig, spec):
    server = node.StandIn(("127.0.0.1", 0), scene="strips")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address
        result = node.capture(host, port, "scan", str(tmp_path / "strips"), pitch=16,
                              exposures=[0.25, 1.0, 4.0, 16.0])
    finally:
        server.shutdown()
        server.server_close()
    assert result["missing"] == [] and result["in_order"]
    view = cli.read_scan(str(tmp_path / "strips"))
    assert len(view["exposures"]) == 4
    columns, mask = decode.decode_exposures(view["exposures"], rig.projector.width, 16)
    single, single_mask = decode.decode_columns(view["exposures"][1]["gray"],
                                                view["exposures"][1]["phase"],
                                                rig.projector.width, 16)
    assert np.count_nonzero(mask) > 1.4 * np.count_nonzero(single_mask)
    points, _ = triangulate.triangulate(columns, mask, synthetic.true_calibration(rig, spec))
    assert geometry.fit_plane(points)[2] < 0.0001


def test_calibration_sequence_writes_a_pose(stand_in, tmp_path):
    host, port = stand_in
    result = node.capture(host, port, "calibration", str(tmp_path / "pose_000"))
    assert result["missing"] == []
    views = list(cli.read_views(str(tmp_path)))
    assert views[0]["columns"].shape[0] == 20 and views[0]["rows"].shape[0] == 20


def test_unknown_pattern_set_returns_error(stand_in):
    host, port = stand_in
    with socket.create_connection((host, port), timeout=10) as sock:
        assert node.receive(sock)[0] == node.HELLO
        node.send(sock, node.SEQUENCE, json.dumps({"set": "none", "pitch": 16, "steps": 4}).encode())
        kind, payload = node.receive(sock)
        assert kind == node.ERROR and "none" in json.loads(payload)["message"]


def test_capture_command_saves_a_scan(stand_in, tmp_path):
    host, port = stand_in
    out = str(tmp_path / "view")
    assert cli.main(["capture", "--node", f"{host}:{port}", "--set", "scan", "--pitch", "16",
                     "--out", out]) == 0
    assert cli.read_scan(out)["phase"].shape[0] == 4


def test_connect_waits_for_a_late_node():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    server = node.StandIn(("127.0.0.1", port))
    timer = threading.Timer(0.5, server.serve_forever)
    timer.daemon = True
    timer.start()
    try:
        sock, hello = node.connect("127.0.0.1", port, wait=10.0, timeout=10.0)
        sock.close()
        assert hello["node"] == "stand-in"
    finally:
        server.shutdown()
        server.server_close()


def test_connect_without_wait_fails_fast():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(OSError):
        node.connect("127.0.0.1", port)


def test_stereo_sequence_pairs_share_timestamps(stand_in, tmp_path):
    host, port = stand_in
    result = node.capture(host, port, "stereo", str(tmp_path / "sweep"), frames=3)
    assert result["missing"] == [] and result["in_order"]
    stamps = result["timestamps_ns"]
    assert stamps[0::2] == stamps[1::2]
    assert stamps[2] - stamps[0] == node.STEREO_PERIOD_NS
    for i in range(3):
        for side in ("left", "right"):
            assert (tmp_path / "sweep" / f"frame_{i:03d}" / f"{side}.png").exists()


def test_stereo_calibration_and_fuse_over_the_protocol(stand_in, tmp_path):
    host, port = stand_in
    for i in range(12):
        node.capture(host, port, "stereo-calibration", str(tmp_path / "cal" / f"pose_{i:03d}"))
    board = ["--squares-x", "9", "--squares-y", "6", "--square-mm", "10", "--marker-mm", "7.5"]
    cal = str(tmp_path / "stereo.yaml")
    assert cli.main(["calibrate-stereo", str(tmp_path / "cal"), *board, "--out", cal]) == 0
    node.capture(host, port, "stereo", str(tmp_path / "sweep"), frames=3)
    model = str(tmp_path / "model.ply")
    assert cli.main(["fuse", str(tmp_path / "sweep"), "--calibration", cal, "--step", "16",
                     "--out", model]) == 0
    assert len(geometry.read_ply(model)) > 3000


def test_paced_stereo_stream_keeps_pairs_in_step(stand_in, tmp_path):
    host, port = stand_in
    result = node.capture(host, port, "stereo", str(tmp_path / "paced"), frames=6, rate_hz=5.0)
    assert result["missing"] == [] and result["frames"] == 12
    stamps = result["timestamps_ns"]
    assert all(stamps[2 * k] == stamps[2 * k + 1] for k in range(6))
    assert np.allclose(np.diff(stamps[::2]), 200_000_000, atol=1)


def test_paced_poses_play_forth_and_back():
    assert [node.paced_index(i, 4) for i in range(9)] == [0, 1, 2, 3, 2, 1, 0, 1, 2]
    assert [node.paced_index(i, 1) for i in range(3)] == [0, 0, 0]


def test_paced_motion_slows_to_turn_and_never_jumps():
    _, poses = node.paced_motion(61)
    for end, s in ((poses[0], 0.0), (poses[-1], 1.0)):
        r, t = synthetic.sweep_pose(s)
        assert np.allclose(end[0], r) and np.allclose(end[1], t)
    centres = np.array([-r.T @ t for r, t in poses])
    played = centres[[node.paced_index(i, 61) for i in range(250)]]
    steps = np.linalg.norm(np.diff(played, axis=0), axis=1)
    assert steps.max() < 0.02
    assert steps[0] < 0.001 and steps[59] < 0.001 and steps[60] < 0.001
    _, short = node.paced_motion(7)
    r, t = synthetic.sweep_pose(1.0, 2.5, 0.8)
    assert np.allclose(short[-1][0], r) and np.allclose(short[-1][1], t)
    _, slide = node.paced_motion(61, panel=True)
    assert np.allclose(slide[-1][1], [-0.15, 0.0, 0.0])
    assert abs(slide[1][1][0]) < abs(slide[31][1][0] - slide[30][1][0])


def test_paced_stream_returns_through_the_poses_it_came_by(tmp_path):
    server = node.StandIn(("127.0.0.1", 0), paced_poses=3)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address
        out = tmp_path / "paced"
        result = node.capture(host, port, "stereo", str(out), frames=5, rate_hz=5.0)
    finally:
        server.shutdown()
        server.server_close()
    assert result["missing"] == [] and result["frames"] == 10
    left = [cv2.imread(str(out / f"frame_{k:03d}" / "left.png"), cv2.IMREAD_GRAYSCALE)
            for k in range(5)]
    assert np.array_equal(left[0], left[4]) and np.array_equal(left[1], left[3])
    assert not np.array_equal(left[1], left[2]) and not np.array_equal(left[0], left[1])


def test_paced_stream_goes_on_after_a_client_leaves(stand_in, tmp_path):
    host, port = stand_in
    sock, _ = node.connect(host, port)
    request = {"set": "stereo", "frames": 6, "rate_hz": 40.0}
    node.send(sock, node.SEQUENCE, json.dumps(request).encode())
    sock.close()
    result = node.capture(host, port, "stereo", str(tmp_path / "next"), frames=2, rate_hz=5.0)
    assert result["missing"] == [] and result["frames"] == 4


def test_stereo_sequence_takes_light_off_pairs(stand_in, tmp_path):
    host, port = stand_in
    out = tmp_path / "dark"
    result = node.capture(host, port, "stereo", str(out), frames=8, dark_every=4)
    assert not result["missing"]
    names = node.pattern_names("stereo", 1024, 768, 4, 8, 4)
    assert names[0:2] == ["dark_000/left", "dark_000/right"]
    assert names[8:10] == ["dark_004/left", "dark_004/right"]
    lit = cv2.imread(str(out / "frame_002" / "left.png"), cv2.IMREAD_GRAYSCALE)
    dark = cv2.imread(str(out / "dark_004" / "left.png"), cv2.IMREAD_GRAYSCALE)
    assert np.std(dark[dark > 0]) < np.std(lit[lit > 0])


def test_paced_stream_takes_light_off_pairs(stand_in, tmp_path):
    host, port = stand_in
    out = tmp_path / "paced-dark"
    result = node.capture(host, port, "stereo", str(out), frames=4, rate_hz=5.0, dark_every=2)
    assert not result["missing"]
    dark = cv2.imread(str(out / "dark_002" / "left.png"), cv2.IMREAD_GRAYSCALE)
    lit = cv2.imread(str(out / "frame_001" / "left.png"), cv2.IMREAD_GRAYSCALE)
    assert np.std(dark[dark > 0]) < np.std(lit[lit > 0])


def test_scan_with_color_fields_over_the_protocol(tmp_path, rig, spec):
    from slscan import colors

    server = node.StandIn(("127.0.0.1", 0), scene="colors")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host, port = server.server_address
        result = node.capture(host, port, "scan", str(tmp_path / "colors"), pitch=16, color=True)
    finally:
        server.shutdown()
        server.server_close()
    assert result["missing"] == [] and result["in_order"]
    view = cli.read_scan(str(tmp_path / "colors"))
    assert set(view["color"]) == {"black", "red", "green", "blue"}
    columns, mask = decode.decode_columns(view["gray"], view["phase"], rig.projector.width, 16)
    points, pixels = triangulate.triangulate(columns, mask, synthetic.true_calibration(rig, spec))
    white = colors.white_balance(synthetic.color_captures(rig, synthetic.color_scenes()["white"]))
    rgb = colors.point_colors(view["color"], pixels, white)
    axes = synthetic.rotation(10, -15)
    along = (points - np.array([0.0, 0.0, 1.0])) @ axes[:, 0]
    for k, channel in ((-1, 0), (0, 1), (1, 2)):
        strip = np.median(rgb[np.abs(along - k * 0.5 / 3.0) < 0.05], axis=0)
        expected = np.where(np.arange(3) == channel, 204, 51)
        assert np.abs(strip - expected).max() <= 2
