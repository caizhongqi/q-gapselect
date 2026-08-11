from __future__ import annotations

import wave

import numpy as np
import pytest

pytest.importorskip("torch")

from qgapselect.qcollide.audio_pcm import load_pcm16_waveform


def test_load_pcm16_waveform_matches_pcm_amplitudes(tmp_path):
    path = tmp_path / "sample.wav"
    samples = np.asarray([0, 16384, -16384, 32767, -32768], dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())

    waveform, sample_rate = load_pcm16_waveform(path)
    assert sample_rate == 16000
    assert tuple(waveform.shape) == (1, len(samples))
    expected = samples.astype(np.float32) / 32768.0
    np.testing.assert_allclose(waveform.numpy()[0], expected, atol=0.0, rtol=0.0)


def test_load_pcm16_waveform_respects_frame_window(tmp_path):
    path = tmp_path / "sample.wav"
    samples = np.arange(16, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())

    waveform, _ = load_pcm16_waveform(path, frame_offset=4, num_frames=5)
    assert tuple(waveform.shape) == (1, 5)
    expected = samples[4:9].astype(np.float32) / 32768.0
    np.testing.assert_allclose(waveform.numpy()[0], expected, atol=0.0, rtol=0.0)
