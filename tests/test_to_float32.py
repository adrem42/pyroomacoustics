"""
Tests for ``pyroomacoustics.utilities.to_float32``.

Unsigned integer PCM (e.g. 8-bit WAV files, which ``scipy.io.wavfile.read``
returns as ``uint8``) has ``np.iinfo(dtype).min == 0``.  The old
implementation computed ``max_val = abs(np.iinfo(dtype).min) == 0`` and
divided by it, silently producing ``nan``/``inf`` and corrupting the signal.
"""

import os
import tempfile

import numpy as np
from scipy.io import wavfile

from pyroomacoustics import create_noisy_signal
from pyroomacoustics.utilities import to_float32


def test_uint8_is_finite_and_in_range():
    # Full uint8 range; libsndfile/soundfile map 8-bit unsigned PCM to
    # [-1, 1) via (x - 128) / 128.
    data = np.arange(256, dtype=np.uint8)
    out = to_float32(data)

    assert out.dtype == np.float32
    assert np.all(np.isfinite(out)), "8-bit unsigned WAV data produced nan/inf"
    assert out.min() >= -1.0
    assert out.max() < 1.0

    expected = (data.astype(np.float32) - 128.0) / 128.0
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)
    # Midpoint of the unsigned range maps to silence (0.0).
    assert to_float32(np.array([128], dtype=np.uint8))[0] == 0.0


def test_uint16_is_finite_and_in_range():
    data = np.array([0, 32768, 65535], dtype=np.uint16)
    out = to_float32(data)

    assert np.all(np.isfinite(out))
    expected = (data.astype(np.float32) - 32768.0) / 32768.0
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_int16_unchanged():
    # Signed integer behaviour must be preserved: min -> -1.0, max -> ~1.0.
    data = np.array([-32768, 0, 32767], dtype=np.int16)
    out = to_float32(data)
    np.testing.assert_allclose(out, [-1.0, 0.0, 32767.0 / 32768.0], atol=1e-7)


def test_float_passthrough():
    data = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    out = to_float32(data)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, data, atol=1e-7)


def test_create_noisy_signal_with_8bit_wav():
    # End-to-end: an 8-bit WAV read by scipy is uint8 and must not corrupt
    # the loaded signal.
    fs = 16000
    t = np.arange(fs) / fs
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    pcm_u8 = np.round(tone * 127 + 128).astype(np.uint8)

    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, "tone_u8.wav")
        wavfile.write(fp, fs, pcm_u8)
        assert wavfile.read(fp)[1].dtype == np.uint8  # 8-bit WAV -> uint8

        noisy, clean, noise, fs_out = create_noisy_signal(fp, snr=10)

    assert np.all(np.isfinite(clean)), "8-bit WAV corrupted the clean signal"
    assert np.all(np.isfinite(noisy))
    assert fs_out == fs


if __name__ == "__main__":
    test_uint8_is_finite_and_in_range()
    test_uint16_is_finite_and_in_range()
    test_int16_unchanged()
    test_float_passthrough()
    test_create_noisy_signal_with_8bit_wav()
    print("ok")
