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
import time

import numpy as np

from . import calibrate, patterns, synthetic

MAGIC = b"SLS1"
HEADER = struct.Struct("<4sBI")
FRAME_HEADER = struct.Struct("<IIHqHHB")
HELLO, SEQUENCE, FRAME, END, ERROR = 1, 2, 3, 4, 5
MONO8 = 0
FRAME_PERIOD_NS = 16_666_667
STEREO_PERIOD_NS = 33_333_333
PACED_CYCLE = 8


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


def pattern_names(pattern_set: str, width: int, height: int, steps: int,
                  frames: int = 1) -> list[str]:
    """Return the capture file names of a pattern set, in sequence order.

    A stereo set is frames pairs, left then right, each pair from one trigger. A
    stereo-calibration set is one board pair."""
    if pattern_set == "stereo":
        return [f"frame_{i:03d}/{side}" for i in range(frames) for side in ("left", "right")]
    if pattern_set == "stereo-calibration":
        return ["left", "right"]
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
            steps: int = 4, timeout: float = 60.0, wait: float = 0.0, frames: int = 1,
            rate_hz: float = 0.0) -> dict:
    """Run one sequence on a node and save its frames as PNG files under out. A rate asks
    a stereo set to be triggered at that many pairs per second."""
    import cv2

    sock, hello = connect(host, port, wait, timeout)
    with sock:
        projector = hello["projector"]
        names = pattern_names(pattern_set, projector["width"], projector["height"], steps, frames)
        request = {"set": pattern_set, "pitch": pitch, "steps": steps, "frames": frames,
                   "rate_hz": rate_hz}
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
            path = os.path.join(out, names[header["index"]] + ".png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, image)
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
        self.stereo = synthetic.stereo_head()
        self.stereo_spec = synthetic.SMALL_BOARD
        self.stereo_poses = synthetic.board_poses(self.stereo_spec, 0.5)
        self.next_stereo_pose = 0
        self.rng = np.random.default_rng(seed)
        self.dots = synthetic.random_dots(self.stereo.light.width, self.stereo.light.height, 3.0,
                                          1.0, np.random.default_rng(seed))
        self.seed = seed
        self.cycle = None
        super().__init__(address, _Handler)

    def hello(self) -> dict:
        r = self.rig
        return {"node": "stand-in", "projector": {"width": r.projector.width,
                                                  "height": r.projector.height,
                                                  "refresh_hz": 60},
                "cameras": [{"name": "final", "width": r.camera.width,
                             "height": r.camera.height, "format": "mono8"},
                            {"name": "left", "width": self.stereo.left.width,
                             "height": self.stereo.left.height, "format": "mono8"},
                            {"name": "right", "width": self.stereo.right.width,
                             "height": self.stereo.right.height, "format": "mono8"}]}

    def frames(self, request: dict) -> tuple[list, list]:
        """Return the images of a sequence and the timestamp of each, in nanoseconds."""
        if request["set"] == "stereo":
            count = int(request.get("frames", 1))
            poses = synthetic.sweep_poses(count, yaw_deg=min(25.0, float(count)),
                                          pitch_deg=min(8.0, 0.3 * count))
            images, stamps = [], []
            captures = synthetic.sweep_captures(self.stereo, synthetic.sweep_scene(), poses,
                                                self.dots, self.seed)
            for i, pair in enumerate(captures):
                images += [pair["left"], pair["right"]]
                stamps += [i * STEREO_PERIOD_NS] * 2
            return images, stamps
        if request["set"] == "stereo-calibration":
            r, t = self.stereo_poses[self.next_stereo_pose % len(self.stereo_poses)]
            self.next_stereo_pose += 1
            pair = synthetic.stereo_board_view(self.stereo, self.stereo_spec, r, t, self.rng)
            return [pair["left"], pair["right"]], [0, 0]
        images = self._pattern_frames(request)
        return images, [i * FRAME_PERIOD_NS for i in range(len(images))]

    def paced_pairs(self, count: int, rate_hz: float):
        """Yield (pair index, left, right, timestamp in nanoseconds) for count pairs on a
        trigger clock at rate_hz, playing a short pre-rendered sweep in a loop. A pair whose
        trigger time passed more than one period before the node reaches it is dropped, as
        a node with a full queue drops it."""
        if self.cycle is None:
            poses = synthetic.sweep_poses(PACED_CYCLE, yaw_deg=float(PACED_CYCLE),
                                          pitch_deg=0.3 * PACED_CYCLE)
            self.cycle = [(pair["left"], pair["right"]) for pair in synthetic.sweep_captures(
                self.stereo, synthetic.sweep_scene(), poses, self.dots, self.seed)]
        period = 1.0 / rate_hz
        start = time.monotonic() + period
        for i in range(count):
            due = start + i * period
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            elif now - due > period:
                continue
            left, right = self.cycle[i % len(self.cycle)]
            yield i, left, right, int(round(i * period * 1e9))

    def _pattern_frames(self, request: dict) -> list:
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
            request = json.loads(payload)
            rate = float(request.get("rate_hz", 0.0))
            if request.get("set") == "stereo" and rate > 0.0:
                sent = 0
                for i, left, right, stamp in server.paced_pairs(int(request.get("frames", 1)),
                                                                rate):
                    send(self.request, FRAME, frame_payload(sequence, 2 * i, 0, stamp, left))
                    send(self.request, FRAME, frame_payload(sequence, 2 * i + 1, 0, stamp, right))
                    sent += 2
                send(self.request, END, json.dumps({"sequence": sequence, "frames": sent}).encode())
                sequence += 1
                continue
            try:
                frames, stamps = server.frames(request)
            except (ValueError, KeyError) as e:
                send(self.request, ERROR, json.dumps({"message": str(e)}).encode())
                continue
            for index, (image, stamp) in enumerate(zip(frames, stamps)):
                send(self.request, FRAME, frame_payload(sequence, index, 0, stamp, image))
            send(self.request, END, json.dumps({"sequence": sequence, "frames": len(frames)}).encode())
            sequence += 1
