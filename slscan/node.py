"""Capture node protocol, a client that saves sequences, and a stand-in node.

Messages over one TCP connection start with the magic "SLS1", a type byte and a payload
length. Control payloads are JSON. A frame payload is a fixed header followed by the
pixels. The stand-in serves synthetic frames from slscan.synthetic, so the application
is developed against the same interface as the capture node.
"""

import json
import os
import socket
import socketserver
import struct

import numpy as np

from . import calibrate, patterns, synthetic

MAGIC = b"SLS1"
HEADER = struct.Struct("<4sBI")
FRAME_HEADER = struct.Struct("<IIHqHHB")
HELLO, SEQUENCE, FRAME, END, ERROR = 1, 2, 3, 4, 5
MONO8 = 0
FRAME_PERIOD_NS = 16_666_667


def send(sock: socket.socket, kind: int, payload: bytes) -> None:
    sock.sendall(HEADER.pack(MAGIC, kind, len(payload)) + payload)


def _exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(min(n - len(data), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed inside a message")
        data.extend(chunk)
    return bytes(data)


def receive(sock: socket.socket) -> tuple[int, bytes]:
    magic, kind, length = HEADER.unpack(_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise ValueError("message without the SLS1 magic")
    return kind, _exact(sock, length)


def frame_payload(sequence: int, index: int, exposure: int, timestamp_ns: int,
                  image: np.ndarray) -> bytes:
    height, width = image.shape
    return FRAME_HEADER.pack(sequence, index, exposure, timestamp_ns, width, height, MONO8) \
        + np.ascontiguousarray(image, dtype=np.uint8).tobytes()


def parse_frame(payload: bytes) -> tuple[dict, np.ndarray]:
    sequence, index, exposure, timestamp_ns, width, height, fmt = \
        FRAME_HEADER.unpack_from(payload)
    if fmt != MONO8:
        raise ValueError(f"unsupported pixel format {fmt}")
    pixels = np.frombuffer(payload, dtype=np.uint8, offset=FRAME_HEADER.size,
                           count=width * height).reshape(height, width)
    return {"sequence": sequence, "index": index, "exposure": exposure,
            "timestamp_ns": timestamp_ns}, pixels


def pattern_names(pattern_set: str, width: int, height: int, steps: int) -> list[str]:
    """Return the capture file names of a pattern set, in projection order."""
    if pattern_set == "calibration":
        return (["white"] + [f"columns_{i:03d}" for i in range(2 * patterns.bit_planes(width))]
                + [f"rows_{i:03d}" for i in range(2 * patterns.bit_planes(height))])
    if pattern_set == "scan":
        return ([f"gray_{i:03d}" for i in range(2 * patterns.bit_planes(width))]
                + [f"phase_{i:03d}" for i in range(steps)])
    raise ValueError(f"unknown pattern set {pattern_set}")


def connect(host: str, port: int, wait: float = 0.0,
            timeout: float = 60.0) -> tuple[socket.socket, dict]:
    """Connect and read HELLO, retrying for up to wait seconds while the node starts."""
    import time

    deadline = time.monotonic() + wait
    while True:
        sock = None
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            kind, payload = receive(sock)
            if kind != HELLO:
                raise ValueError("node did not open with HELLO")
            return sock, json.loads(payload)
        except (ConnectionError, OSError):
            if sock is not None:
                sock.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


def capture(host: str, port: int, pattern_set: str, out: str, pitch: int = 32,
            steps: int = 4, timeout: float = 60.0, wait: float = 0.0) -> dict:
    """Run one sequence on a node and save its frames as PNG files under out."""
    import cv2

    sock, hello = connect(host, port, wait, timeout)
    with sock:
        projector = hello["projector"]
        names = pattern_names(pattern_set, projector["width"], projector["height"], steps)
        request = {"set": pattern_set, "pitch": pitch, "steps": steps}
        send(sock, SEQUENCE, json.dumps(request).encode())
        os.makedirs(out, exist_ok=True)
        indices, stamps = [], []
        while True:
            kind, payload = receive(sock)
            if kind == ERROR:
                raise RuntimeError(json.loads(payload)["message"])
            if kind == END:
                break
            if kind != FRAME:
                raise ValueError(f"unexpected message type {kind}")
            header, image = parse_frame(payload)
            indices.append(header["index"])
            stamps.append(header["timestamp_ns"])
            cv2.imwrite(os.path.join(out, names[header["index"]] + ".png"), image)
    missing = sorted(set(range(len(names))) - set(indices))
    return {"node": hello, "frames": len(indices), "missing": missing,
            "in_order": indices == sorted(indices), "timestamps_ns": stamps}


class StandIn(socketserver.ThreadingTCPServer):
    """A capture node that renders synthetic frames of one scene and of board poses."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, scene: str = "plate", seed: int = 0):
        self.rig = synthetic.reference_rig()
        self.spec = calibrate.BoardSpec(9, 6, 0.04, 0.03)
        self.scene = synthetic.acceptance_scenes()[scene]
        self.poses = synthetic.board_poses(self.spec)
        self.next_pose = 0
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        super().__init__(address, _Handler)

    def hello(self) -> dict:
        r = self.rig
        return {"node": "stand-in", "projector": {"width": r.projector.width,
                                                  "height": r.projector.height,
                                                  "refresh_hz": 60},
                "cameras": [{"name": "final", "width": r.camera.width,
                             "height": r.camera.height, "format": "mono8"}]}

    def frames(self, request: dict) -> list:
        if request["set"] == "calibration":
            r, t = self.poses[self.next_pose % len(self.poses)]
            self.next_pose += 1
            view = synthetic.calibration_view(self.rig, self.spec, r, t, self.rng)
            return [view["white"]] + list(view["columns"]) + list(view["rows"])
        if request["set"] == "scan":
            captures, _ = synthetic.scan_captures(self.rig, self.scene, int(request["pitch"]),
                                                  int(request["steps"]), self.seed)
            return list(captures["gray"]) + list(captures["phase"])
        raise ValueError(f"unknown pattern set {request['set']}")


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        server: StandIn = self.server
        send(self.request, HELLO, json.dumps(server.hello()).encode())
        sequence = 0
        while True:
            try:
                kind, payload = receive(self.request)
            except (ConnectionError, OSError):
                return
            if kind != SEQUENCE:
                send(self.request, ERROR, json.dumps({"message": "expected SEQUENCE"}).encode())
                continue
            try:
                frames = server.frames(json.loads(payload))
            except (ValueError, KeyError) as e:
                send(self.request, ERROR, json.dumps({"message": str(e)}).encode())
                continue
            for index, image in enumerate(frames):
                stamp = index * FRAME_PERIOD_NS
                send(self.request, FRAME, frame_payload(sequence, index, 0, stamp, image))
            send(self.request, END, json.dumps({"sequence": sequence, "frames": len(frames)}).encode())
            sequence += 1
