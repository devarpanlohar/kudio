"""
crossfade.py -- Track Crossfade Engine
========================================
Overlaps the tail of track A with the intro of track B.

Fade curves:
  cosine     -- equal-power (no perceived loudness dip at midpoint)
  exponential -- faster initial fade, used for abrupt transitions

Usage (from streaming.py):
    cf = CrossfadeEngine(duration_s=4.0, curve='cosine', sr=44100)
    blended = cf.mix(tail_a, head_b)   # returns overlap_samples frames

Album mode: CrossfadeEngine respects stored track gain metadata so
album dynamics are preserved across the crossfade.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackMeta:
    """Loudness metadata stored per track for album normalization."""
    title:          str   = ""
    track_loudness: float = -14.0   # LUFS
    track_gain:     float = 0.0     # dB adjustment
    album_loudness: float = -14.0   # LUFS (shared across album)
    album_gain:     float = 0.0     # dB adjustment
    true_peak:      float = -1.0    # dBTP
    duration_s:     float = 0.0


class CrossfadeEngine:
    """
    Real-time crossfade between two consecutive tracks.

    Parameters
    ----------
    duration_s  : overlap length in seconds (1–12)
    curve       : 'cosine' (equal-power) or 'exponential'
    sr          : sample rate
    """

    def __init__(self, duration_s: float = 4.0,
                 curve: str = 'cosine', sr: int = 44100):
        self.sr         = sr
        self.curve      = curve
        self.set_duration(duration_s)

    def set_duration(self, duration_s: float):
        duration_s      = max(1.0, min(12.0, duration_s))
        self.n_samples  = int(self.sr * duration_s)
        self._fade_out  = self._make_curve(self.n_samples, out=True)
        self._fade_in   = self._make_curve(self.n_samples, out=False)

    def _make_curve(self, n: int, out: bool) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        if self.curve == 'cosine':
            # Equal-power: sum of squares = 1 at every point
            fade = np.cos(t * np.pi * 0.5) if out else np.sin(t * np.pi * 0.5)
        else:  # exponential
            if out:
                fade = np.exp(-4.0 * t).astype(np.float32)
                fade /= fade[0]
            else:
                fade = np.exp(-4.0 * (1.0 - t)).astype(np.float32)
                fade /= fade[-1]
        return fade.reshape(-1, 1)   # (N,1) for broadcasting with (N,2)

    def mix(self, tail_a: np.ndarray, head_b: np.ndarray,
            meta_a: Optional[TrackMeta] = None,
            meta_b: Optional[TrackMeta] = None,
            album_mode: bool = False) -> np.ndarray:
        """
        Blend the tail of track A into the head of track B.

        Parameters
        ----------
        tail_a  : (N, 2) float32 -- last N frames of track A
        head_b  : (N, 2) float32 -- first N frames of track B
        meta_a  : optional TrackMeta for track A
        meta_b  : optional TrackMeta for track B
        album_mode : if True use album gain instead of track gain

        Returns
        -------
        (N, 2) float32 -- blended output
        """
        n = min(len(tail_a), len(head_b), self.n_samples)
        fa = self._fade_out[:n]
        fb = self._fade_in[:n]

        # Apply gain adjustment if metadata provided
        gain_a = gain_b = 1.0
        if meta_a:
            db = meta_a.album_gain if album_mode else meta_a.track_gain
            gain_a = 10.0 ** (db / 20.0)
        if meta_b:
            db = meta_b.album_gain if album_mode else meta_b.track_gain
            gain_b = 10.0 ** (db / 20.0)

        blended = (tail_a[:n] * fa * gain_a +
                   head_b[:n] * fb * gain_b).astype(np.float32)
        return np.clip(blended, -1.0, 1.0, out=blended)

    def pre_roll(self, n_frames: int = None) -> int:
        """How many frames before end of track A to start capturing tail."""
        return n_frames or self.n_samples


class AlbumNormalizer:
    """
    Stores per-track and album loudness metadata.
    Computes ReplayGain-style gains for track and album normalization.

    Usage:
        an = AlbumNormalizer(target_lufs=-14.0)
        an.add_track("Song A", measured_lufs=-16.2, true_peak=-0.8)
        an.add_track("Song B", measured_lufs=-12.0, true_peak=-0.1)
        an.finalize()   # computes album loudness and all gains
        meta = an.get_meta("Song A")
        print(meta.track_gain, meta.album_gain)
    """

    def __init__(self, target_lufs: float = -14.0):
        self.target = target_lufs
        self._tracks: list[TrackMeta] = []

    def add_track(self, title: str, measured_lufs: float,
                  true_peak: float = -1.0, duration_s: float = 0.0):
        meta = TrackMeta(
            title          = title,
            track_loudness = measured_lufs,
            track_gain     = self.target - measured_lufs,
            true_peak      = true_peak,
            duration_s     = duration_s,
        )
        self._tracks.append(meta)

    def finalize(self):
        """Compute album loudness (energy-weighted mean) and album gains."""
        if not self._tracks:
            return
        # Energy-weighted average loudness
        powers = [10.0 ** (t.track_loudness / 10.0) for t in self._tracks]
        album_lufs = 10.0 * np.log10(np.mean(powers))
        album_gain = self.target - album_lufs
        for t in self._tracks:
            t.album_loudness = album_lufs
            t.album_gain     = album_gain

    def get_meta(self, title: str) -> Optional[TrackMeta]:
        for t in self._tracks:
            if t.title == title:
                return t
        return None

    def all_tracks(self) -> list:
        return list(self._tracks)