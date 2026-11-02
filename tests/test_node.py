import json
import socket
import threading

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
