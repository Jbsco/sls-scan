"""Futhark kernels from Python: fut/decode.fut built into a shared library for one Futhark
backend, its decode entry points called through ctypes over the C interface an application
would link."""

import ctypes
import os
import subprocess
import time

import numpy as np

BACKENDS = ("c", "multicore", "opencl", "hip")
PRECISIONS = ("f64", "f32")
SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fut",
                      "decode.fut")
_TUPLE = "opaque_tup2_arr1d_f32_arr1d_u8"
_P = ctypes.c_void_p
_libc = ctypes.CDLL(None)
_libc.free.argtypes = [_P]


def rocm() -> str:
    """Return the ROCm root the HIP backend builds and runs against."""
    return os.environ.get("ROCM_PATH", "/opt/rocm")


def link_flags(backend: str) -> list[str]:
    """Return the compiler and linker flags a library of backend needs."""
    if backend == "c":
        return ["-lm"]
    if backend == "multicore":
        return ["-pthread", "-lm"]
    if backend == "opencl":
        return ["-lOpenCL", "-lm"]
    if backend == "hip":
        root = rocm()
        return [f"-I{root}/include", f"-L{root}/lib", f"-Wl,-rpath,{root}/lib",
                "-lamdhip64", "-lhiprtc", "-lm"]
    raise ValueError(f"unknown Futhark backend {backend!r}")


def build(backend: str, out: str, source: str = SOURCE) -> str:
    """Compile source with a Futhark backend into a shared library in out and return the
    library's path."""
    flags = link_flags(backend)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"decode_{backend}")
    subprocess.run(["futhark", backend, "--library", "-o", stem, source], check=True)
    library = os.path.join(out, f"libdecode_{backend}.so")
    subprocess.run([os.environ.get("CC", "cc"), "-O3", "-fPIC", "-shared", f"{stem}.c", "-o",
                    library, *flags], check=True)
    return library


class Decoder:
    """The decode entry points of fut/decode.fut in a library built for one backend, on one
    Futhark context."""

    def __init__(self, library: str, backend: str = "c"):
        lib = ctypes.CDLL(library)
        status = ctypes.c_int
        signatures = {
            "futhark_context_config_new": (_P, []),
            "futhark_context_config_free": (None, [_P]),
            "futhark_context_new": (_P, [_P]),
            "futhark_context_free": (None, [_P]),
            "futhark_context_get_error": (_P, [_P]),
            "futhark_context_sync": (status, [_P]),
            "futhark_new_u8_2d": (_P, [_P, _P, ctypes.c_int64, ctypes.c_int64]),
            "futhark_free_u8_2d": (status, [_P, _P]),
            "futhark_values_f32_1d": (status, [_P, _P, _P]),
            "futhark_values_u8_1d": (status, [_P, _P, _P]),
            "futhark_free_f32_1d": (status, [_P, _P]),
            "futhark_free_u8_1d": (status, [_P, _P]),
            f"futhark_project_{_TUPLE}_0": (status, [_P, ctypes.POINTER(_P), _P]),
            f"futhark_project_{_TUPLE}_1": (status, [_P, ctypes.POINTER(_P), _P]),
            f"futhark_free_{_TUPLE}": (status, [_P, _P]),
        }
        for precision, real in (("f64", ctypes.c_double), ("f32", ctypes.c_float)):
            signatures[f"futhark_entry_decode_{precision}"] = (
                status, [_P, ctypes.POINTER(_P), _P, _P, ctypes.c_int64] + [real] * 4)
        if backend == "hip":
            signatures["futhark_context_config_add_build_option"] = (None, [_P, ctypes.c_char_p])
        for name, (result, args) in signatures.items():
            function = getattr(lib, name)
            function.restype, function.argtypes = result, args
        self.backend = backend
        self._lib = lib
        self._config = lib.futhark_context_config_new()
        if backend == "hip":
            lib.futhark_context_config_add_build_option(self._config,
                                                        f"-I{rocm()}/include".encode())
        self._context = lib.futhark_context_new(self._config)
        error = self._error()
        if error:
            self.close()
            raise RuntimeError(error)

    def __enter__(self) -> "Decoder":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Free the context and its configuration."""
        if self._context:
            self._lib.futhark_context_free(self._context)
            self._context = None
        if self._config:
            self._lib.futhark_context_config_free(self._config)
            self._config = None

    def _error(self) -> str | None:
        pointer = self._lib.futhark_context_get_error(self._context)
        if not pointer:
            return None
        message = ctypes.string_at(pointer).decode(errors="replace")
        _libc.free(pointer)
        return message

    def _check(self, status: int) -> None:
        if status:
            raise RuntimeError(self._error() or f"Futhark call failed with status {status}")

    def _upload(self, stacks: list) -> list:
        arrays = []
        for stack in stacks:
            array = self._lib.futhark_new_u8_2d(self._context, stack.ctypes.data, *stack.shape)
            if not array:
                self._free(arrays)
                raise RuntimeError(self._error() or "Futhark upload failed")
            arrays.append(array)
        return arrays

    def _free(self, arrays: list) -> None:
        for array in arrays:
            self._lib.futhark_free_u8_2d(self._context, array)

    def _run(self, arrays: list, width: int, thresholds: tuple, precision: str):
        if precision not in PRECISIONS:
            raise ValueError(f"unknown precision {precision!r}")
        entry = getattr(self._lib, f"futhark_entry_decode_{precision}")
        result = _P()
        self._check(entry(self._context, ctypes.byref(result), *arrays, int(width),
                          *(float(t) for t in thresholds)))
        return result

    def decode(self, gray: np.ndarray, phase: np.ndarray, width: int, pitch: int,
               min_contrast: float = 10.0, min_modulation: float = 10.0,
               max_disagreement: float = 1.0,
               precision: str = "f64") -> tuple[np.ndarray, np.ndarray]:
        """Return the projector column per camera pixel and its mask, as
        decode.decode_columns does, computed in precision ("f64" or "f32")."""
        shape = np.shape(gray)[1:]
        stacks = [np.ascontiguousarray(s, dtype=np.uint8).reshape(len(s), -1)
                  for s in (gray, phase)]
        n = stacks[0].shape[1]
        columns, mask = np.empty(n, np.float32), np.empty(n, np.uint8)
        arrays = self._upload(stacks)
        try:
            result = self._run(arrays, width, (pitch, min_contrast, min_modulation,
                                               max_disagreement), precision)
            parts = [_P(), _P()]
            try:
                for i, part in enumerate(parts):
                    self._check(getattr(self._lib, f"futhark_project_{_TUPLE}_{i}")(
                        self._context, ctypes.byref(part), result))
                self._check(self._lib.futhark_values_f32_1d(self._context, parts[0],
                                                            columns.ctypes.data))
                self._check(self._lib.futhark_values_u8_1d(self._context, parts[1],
                                                           mask.ctypes.data))
                self._check(self._lib.futhark_context_sync(self._context))
            finally:
                for free, part in zip((self._lib.futhark_free_f32_1d,
                                       self._lib.futhark_free_u8_1d), parts):
                    if part:
                        free(self._context, part)
                getattr(self._lib, f"futhark_free_{_TUPLE}")(self._context, result)
        finally:
            self._free(arrays)
        return columns.reshape(shape), mask.reshape(shape).astype(bool)

    def time_kernel(self, gray: np.ndarray, phase: np.ndarray, width: int, pitch: int,
                    min_contrast: float = 10.0, min_modulation: float = 10.0,
                    max_disagreement: float = 1.0, precision: str = "f64",
                    runs: int = 10) -> float:
        """Return the fastest of runs decodes in milliseconds, on stacks already on the
        device and with no readback."""
        stacks = [np.ascontiguousarray(s, dtype=np.uint8).reshape(len(s), -1)
                  for s in (gray, phase)]
        arrays = self._upload(stacks)
        thresholds = (pitch, min_contrast, min_modulation, max_disagreement)
        fastest = float("inf")
        try:
            self._check(self._lib.futhark_context_sync(self._context))
            for _ in range(runs):
                start = time.perf_counter()
                result = self._run(arrays, width, thresholds, precision)
                self._check(self._lib.futhark_context_sync(self._context))
                fastest = min(fastest, 1000.0 * (time.perf_counter() - start))
                getattr(self._lib, f"futhark_free_{_TUPLE}")(self._context, result)
        finally:
            self._free(arrays)
        return fastest
