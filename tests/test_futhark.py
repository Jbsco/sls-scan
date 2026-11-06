import shutil

import numpy as np
import pytest

from slscan import decode, futhark, synthetic

pytestmark = pytest.mark.skipif(shutil.which("futhark") is None,
                                reason="futhark not installed")

PITCH = 16


@pytest.fixture(scope="module")
def decoder(tmp_path_factory):
    library = futhark.build("c", str(tmp_path_factory.mktemp("futhark")))
    with futhark.Decoder(library, "c") as d:
        yield d


def plate_scan(rig, exposure=1.0):
    captures, _ = synthetic.scan_captures(rig, [synthetic.acceptance_scenes()["plate"][0]],
                                          pitch=PITCH, exposure=exposure)
    expected = decode.decode_columns(captures["gray"], captures["phase"],
                                     rig.projector.width, PITCH)
    return captures, expected


def test_double_precision_matches_numpy_bit_for_bit(rig, decoder):
    captures, (expected, expected_mask) = plate_scan(rig)
    columns, mask = decoder.decode(captures["gray"], captures["phase"], rig.projector.width,
                                   PITCH)
    assert mask.shape == expected_mask.shape and mask.any()
    assert np.array_equal(mask, expected_mask)
    assert np.array_equal(columns[mask], expected.astype(np.float32)[mask])


def test_single_precision_stays_within_a_thousandth_of_a_pixel(rig, decoder):
    captures, (expected, expected_mask) = plate_scan(rig)
    columns, mask = decoder.decode(captures["gray"], captures["phase"], rig.projector.width,
                                   PITCH, precision="f32")
    assert np.count_nonzero(mask != expected_mask) <= 1e-4 * mask.size
    shared = mask & expected_mask
    assert np.abs(columns[shared] - expected[shared]).max() < 1e-3


def test_saturated_phase_samples_are_masked(rig, decoder):
    captures, (_, expected_mask) = plate_scan(rig, exposure=4.0)
    _, mask = decoder.decode(captures["gray"], captures["phase"], rig.projector.width, PITCH)
    saturated = captures["phase"].max(axis=0) >= decode.SATURATED
    assert saturated.any() and not (mask & saturated).any()
    assert np.array_equal(mask, expected_mask)


def test_kernel_time_is_measured(decoder):
    gray, phase = np.zeros((12, 4, 8), np.uint8), np.zeros((4, 4, 8), np.uint8)
    ms = decoder.time_kernel(gray, phase, 64, 8, runs=2)
    assert np.isfinite(ms) and ms >= 0.0


def test_unknown_precision_is_refused(decoder):
    gray, phase = np.zeros((12, 4, 8), np.uint8), np.zeros((4, 4, 8), np.uint8)
    with pytest.raises(ValueError):
        decoder.decode(gray, phase, 64, 8, precision="f16")


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError):
        futhark.link_flags("vulkan")

