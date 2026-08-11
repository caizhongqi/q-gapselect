"""Deterministic standard-library PCM decoder for Speech Commands experiments."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np


def load_pcm16_waveform(
    path: str | Path,
    *,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    **_: Any,
):
    """Load uncompressed 16-bit PCM WAV without TorchCodec/FFmpeg.

    The return contract matches the subset of ``torchaudio.load`` used by the
    Speech Commands dataset: ``(waveform_tensor, sample_rate)``.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError("audio PCM loading requires PyTorch") from exc

    if frame_offset < 0:
        raise ValueError("frame_offset must be non-negative")
    if num_frames == 0 or num_frames < -1:
        raise ValueError("num_frames must be -1 or a positive integer")

    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError("Speech Commands WAV must use uncompressed PCM")
        if handle.getsampwidth() != 2:
            raise ValueError("Speech Commands WAV must use 16-bit PCM samples")
        channels = int(handle.getnchannels())
        sample_rate = int(handle.getframerate())
        available = int(handle.getnframes())
        if frame_offset > available:
            raise ValueError("frame_offset exceeds available WAV frames")
        handle.setpos(frame_offset)
        requested = available - frame_offset if num_frames < 0 else min(num_frames, available - frame_offset)
        payload = handle.readframes(requested)

    samples = np.frombuffer(payload, dtype="<i2")
    if samples.size % channels != 0:
        raise ValueError("malformed interleaved PCM payload")
    samples = samples.reshape(-1, channels)
    if normalize:
        values = samples.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(values.copy())
    else:
        tensor = torch.from_numpy(samples.copy())
    if channels_first:
        tensor = tensor.transpose(0, 1).contiguous()
    return tensor, sample_rate


def patch_torchaudio_pcm_loader() -> None:
    """Patch ``torchaudio.load`` for the PCM-only Speech Commands workflow."""

    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - executable dependency guard
        raise RuntimeError("audio PCM patch requires torchaudio") from exc
    torchaudio.load = load_pcm16_waveform


__all__ = ["load_pcm16_waveform", "patch_torchaudio_pcm_loader"]
