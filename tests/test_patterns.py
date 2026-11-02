import math

import numpy as np

from slscan import patterns


def test_gray_code_round_trip():
    for n in range(4096):
        assert patterns.gray_decode(patterns.gray_encode(n)) == n


def test_gray_code_neighbours_differ_in_one_bit():
    for n in range(4095):
        a = patterns.gray_encode(n)
        b = patterns.gray_encode(n + 1)
        assert bin(a ^ b).count("1") == 1


def test_bit_planes_cover_the_width():
    assert patterns.bit_planes(1) == 1
    assert patterns.bit_planes(2) == 1
    assert patterns.bit_planes(1024) == 10
    assert patterns.bit_planes(1920) == 11
    assert 2 ** patterns.bit_planes(1920) >= 1920


def test_gray_code_images_decode_to_their_columns():
    width, height = 1920, 4
    stack = patterns.gray_code_images(width, height)
    planes = patterns.bit_planes(width)
    assert stack.shape == (2 * planes, height, width)
    assert np.array_equal(stack[0::2], 255 - stack[1::2])
    bits = (stack[0::2] > 127).astype(np.uint8)
    columns = patterns.decode_gray_bits(bits)
    expected = np.broadcast_to(np.arange(width), (height, width))
    assert np.array_equal(columns, expected)


def test_row_gray_code_images_decode_to_their_rows():
    width, height = 4, 768
    stack = patterns.gray_code_images(width, height, axis="rows")
    planes = patterns.bit_planes(height)
    assert stack.shape == (2 * planes, height, width)
    bits = (stack[0::2] > 127).astype(np.uint8)
    rows = patterns.decode_gray_bits(bits)
    expected = np.broadcast_to(np.arange(height)[:, np.newaxis], (height, width))
    assert np.array_equal(rows, expected)


def test_binarize_pairs_masks_low_contrast():
    stack = patterns.gray_code_images(64, 2).astype(np.float64)
    captured = 20.0 + 0.5 * stack
    captured[:, 1, :10] = 40.0
    bits, mask = patterns.binarize_pairs(captured, min_contrast=10.0)
    assert np.array_equal(patterns.decode_gray_bits(bits)[0], np.arange(64))
    assert mask[0].all()
    assert not mask[1, :10].any() and mask[1, 10:].all()


def test_phase_shift_images_recover_the_phase():
    width, height, pitch, steps = 256, 2, 32, 4
    stack = patterns.phase_shift_images(width, height, pitch, steps)
    assert stack.shape == (steps, height, width)
    phase = patterns.wrapped_phase(stack.astype(np.float64))
    expected = np.mod(2.0 * math.pi * np.arange(width) / pitch, 2.0 * math.pi)
    error = np.abs(np.angle(np.exp(1j * (phase[0] - expected))))
    assert np.max(error) < 0.02  # 8-bit quantization of the sinusoids


def test_unwrap_columns_combines_period_and_phase():
    pitch = 32
    columns = np.arange(0, 256, dtype=np.float64)
    period_index = np.floor(columns / pitch).astype(np.int64)
    phase = 2.0 * math.pi * np.mod(columns, pitch) / pitch
    recovered = patterns.unwrap_columns(phase, period_index, pitch)
    assert np.allclose(recovered, columns)
