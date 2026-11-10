"""Capture node protocol, a client that saves sequences, and a stand-in node.

Messages over one TCP connection start with the magic "SLS1", a type byte and a payload
length. Control payloads are JSON. A frame payload is a fixed header followed by the
pixels. The stand-in serves synthetic frames from slscan.synthetic, so the application
is developed against the same interface as the capture node.
"""

import json
import math
import os
import socket
import socketserver
import struct
import threading
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
PACED_POSES = 8
PACED_ARC_POSES = 61


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


def is_dark(pair: int, dark_every: int) -> bool:
    """Return whether a stereo pair is taken with the pattern light off: every
    dark_every-th pair from the first, none when dark_every is 0."""
    return dark_every > 0 and pair % dark_every == 0


def paced_index(pair: int, poses: int) -> int:
    """Return which of a paced stream's head poses a pair shows. The poses play forth and
    back, so the one before the last follows the last."""
    if poses < 2:
        return 0
    k = pair % (2 * (poses - 1))
    return k if k < poses else 2 * (poses - 1) - k


def paced_motion(poses: int, panel: bool = False) -> tuple[dict, list]:
    """The scene and the head poses a paced stereo stream plays forth and back: an arc
    around a part on a mat, or with panel a slide along a dotted panel. PACED_ARC_POSES
    poses or more span 25° of yaw to either side, or 150 mm of slide, and fewer span less
    at the same spacing. Poses lie closer together toward both ends, so the head slows
    before it turns."""
    eased = [(1.0 - math.cos(math.pi * k / max(poses - 1, 1))) / 2.0 for k in range(poses)]
    reach = min(1.0, (poses - 1) / (PACED_ARC_POSES - 1))
    if panel:
        return synthetic.panel_scene(), [(np.eye(3), np.array([-0.15 * reach * s, 0.0, 0.0]))
                                         for s in eased]
    return synthetic.sweep_scene(), [synthetic.sweep_pose(s, 25.0 * reach, 8.0 * reach)
                                     for s in eased]


def pattern_names(pattern_set: str, width: int, height: int, steps: int,
                  frames: int = 1, dark_every: int = 0, exposures: int = 1,
                  color: bool = False) -> list[str]:
    """Return the capture file names of a pattern set, in sequence order.

    A stereo set is frames pairs, left then right, each pair from one trigger, and a pair
    taken with the pattern light off is named dark_ in place of frame_. A
    stereo-calibration set is one board pair. A scan set at several exposures takes each
    pattern at every exposure before the next pattern, under exposure_000/ onward, and a
    scan set with color ends with the black, red, green and blue fields under color/."""
    if pattern_set == "stereo":
        return [f"{'dark' if is_dark(i, dark_every) else 'frame'}_{i:03d}/{side}"
                for i in range(frames) for side in ("left", "right")]
    if pattern_set == "stereo-calibration":
        return ["left", "right"]
    if pattern_set == "calibration":
        return (["white"] + [f"columns_{i:03d}" for i in range(2 * patterns.bit_planes(width))]
                + [f"rows_{i:03d}" for i in range(2 * patterns.bit_planes(height))])
    if pattern_set == "scan":
        names = ([f"gray_{i:03d}" for i in range(2 * patterns.bit_planes(width))]
                 + [f"phase_{i:03d}" for i in range(steps)])
        if exposures > 1:
            names = [f"exposure_{e:03d}/{name}" for name in names for e in range(exposures)]
        if color:
            names += [f"color/{field}" for field in ("black", "red", "green", "blue")]
        return names
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
            rate_hz: float = 0.0, dark_every: int = 0, exposures=None,
            color: bool = False) -> dict:
    """Run one sequence on a node and save its frames as PNG files under out. A rate asks
    a stereo set to be triggered at that many pairs per second, dark_every asks for
    every dark_every-th pair with the pattern light off, exposures asks for a scan set at
    each of those exposures relative to the nominal one, and color for a scan set's black,
    red, green and blue fields."""
    import cv2

    sock, hello = connect(host, port, wait, timeout)
    with sock:
        projector = hello["projector"]
        names = pattern_names(pattern_set, projector["width"], projector["height"], steps, frames,
                              dark_every, len(exposures) if exposures else 1, color)
        request = {"set": pattern_set, "pitch": pitch, "steps": steps, "frames": frames,
                   "rate_hz": rate_hz, "dark_every": dark_every}
        if exposures:
            request["exposures"] = [float(x) for x in exposures]
        if color:
            request["color"] = True
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

    def __init__(self, address, scene: str = "plate", seed: int = 0,
                 paced_poses: int = PACED_POSES):
        self.rig = synthetic.reference_rig()
        self.spec = calibrate.BoardSpec(9, 6, 0.04, 0.03)
        self.scene = {**synthetic.acceptance_scenes(), **synthetic.strip_scenes(),
                      **synthetic.color_scenes()}[scene]
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
        self.paced_poses = max(1, paced_poses)
        self.paced = {}
        self.paced_lock = threading.Lock()
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

    def frames(self, request: dict) -> tuple[list, list, list]:
        """Return the images of a sequence, the timestamp of each in nanoseconds and the
        exposure index of each."""
        if request["set"] == "stereo":
            count = int(request.get("frames", 1))
            dark_every = int(request.get("dark_every", 0))
            scene, poses = self.stereo_sequence(count, dark_every)
            images, stamps = [], []
            captures = synthetic.sweep_captures(self.stereo, scene, poses, self.dots, self.seed,
                                                dark_every)
            for i, pair in enumerate(captures):
                images += [pair["left"], pair["right"]]
                stamps += [i * STEREO_PERIOD_NS] * 2
            return images, stamps, [0] * len(images)
        if request["set"] == "stereo-calibration":
            r, t = self.stereo_poses[self.next_stereo_pose % len(self.stereo_poses)]
            self.next_stereo_pose += 1
            pair = synthetic.stereo_board_view(self.stereo, self.stereo_spec, r, t, self.rng)
            return [pair["left"], pair["right"]], [0, 0], [0, 0]
        images, exposures = self._pattern_frames(request)
        return images, [i * FRAME_PERIOD_NS for i in range(len(images))], exposures

    @staticmethod
    def stereo_sequence(count: int, dark_every: int) -> tuple[dict, list]:
        """The scene and head poses of a stereo set: a sweep around a part on a mat, or with
        light-off pairs a slide along a dotted panel, which reference dots serve."""
        if dark_every > 0:
            return synthetic.panel_scene(), synthetic.slide_poses(count)
        poses = synthetic.sweep_poses(count, yaw_deg=min(25.0, float(count)),
                                      pitch_deg=min(8.0, 0.3 * count))
        return synthetic.sweep_scene(), poses

    def paced_pair(self, panel: bool, index: int, dark: bool) -> tuple:
        """Return the left and right images of one paced head pose, rendered on first use."""
        key = (panel, index, dark)
        with self.paced_lock:
            if key not in self.paced:
                scene, poses = paced_motion(self.paced_poses, panel)
                rng = np.random.default_rng([self.seed, int(panel), index, int(dark)])
                pair = synthetic.sweep_capture(self.stereo, scene, poses[index], self.dots, rng,
                                               dark)
                self.paced[key] = (pair["left"], pair["right"])
            return self.paced[key]

    def paced_pairs(self, count: int, rate_hz: float, dark_every: int = 0):
        """Yield (pair index, left, right, timestamp in nanoseconds) for count pairs on a
        trigger clock at rate_hz, playing its head poses forth and back. Poses are rendered
        before the clock starts. A pair whose trigger time passed more than one period
        before the node reaches it is dropped, as a node with a full queue drops it."""
        panel = dark_every > 0
        shown = [(paced_index(i, self.paced_poses), is_dark(i, dark_every))
                 for i in range(count)]
        rendered = {key: self.paced_pair(panel, *key) for key in dict.fromkeys(shown)}
        period = 1.0 / rate_hz
        start = time.monotonic() + period
        for i, (index, dark) in enumerate(shown):
            due = start + i * period
            now = time.monotonic()
            if now < due:
                time.sleep(due - now)
            elif now - due > period:
                continue
            left, right = rendered[index, dark]
            yield i, left, right, int(round(i * period * 1e9))

    def _pattern_frames(self, request: dict) -> tuple[list, list]:
        if request["set"] == "calibration":
            r, t = self.poses[self.next_pose % len(self.poses)]
            self.next_pose += 1
            view = synthetic.calibration_view(self.rig, self.spec, r, t, self.rng)
            images = [view["white"]] + list(view["columns"]) + list(view["rows"])
            return images, [0] * len(images)
        if request["set"] == "scan":
            exposures = [float(x) for x in request.get("exposures", [1.0])] or [1.0]
            views = synthetic.exposure_captures(self.rig, self.scene, exposures,
                                                int(request["pitch"]), int(request["steps"]),
                                                self.seed)
            stacks = [list(v["gray"]) + list(v["phase"]) for v in views]
            images = [stack[k] for k in range(len(stacks[0])) for stack in stacks]
            levels = [e for _ in range(len(stacks[0])) for e in range(len(stacks))]
            if request.get("color"):
                fields = synthetic.color_captures(self.rig, self.scene, self.seed + len(views))
                images += [fields[name] for name in ("black", "red", "green", "blue")]
                levels += [0] * 4
            return images, levels
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
                                                                rate,
                                                                int(request.get("dark_every", 0))):
                    send(self.request, FRAME, frame_payload(sequence, 2 * i, 0, stamp, left))
                    send(self.request, FRAME, frame_payload(sequence, 2 * i + 1, 0, stamp, right))
                    sent += 2
                send(self.request, END, json.dumps({"sequence": sequence, "frames": sent}).encode())
                sequence += 1
                continue
            try:
                frames, stamps, exposures = server.frames(request)
            except (ValueError, KeyError) as e:
                send(self.request, ERROR, json.dumps({"message": str(e)}).encode())
                continue
            for index, (image, stamp, exposure) in enumerate(zip(frames, stamps, exposures)):
                send(self.request, FRAME, frame_payload(sequence, index, exposure, stamp, image))
            send(self.request, END, json.dumps({"sequence": sequence, "frames": len(frames)}).encode())
            sequence += 1
