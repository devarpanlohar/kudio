"""
meters.py -- ITU-R BS.1770-4 K-Weighted LUFS Loudness Meter
============================================================
Non-blocking: push() feeds audio from the DSP thread;
print_meter() is called by a background timer thread.
"""

import sys
import threading
import numpy as np
from scipy import signal as sp
from filters import _design_sos

SR = 44100


class LUFSMeter:
    """
    Integrated short-term loudness meter.

    K-weighting (ITU-R BS.1770-4):
      Stage 1: +4 dB high-shelf pre-filter @ 1681 Hz
      Stage 2: 100 Hz highpass (RLB weighting)

    LUFS = -0.691 + 10*log10(mean(filtered2)) over a 400 ms sliding window.
    """

    def __init__(self, sr: int = SR, window_ms: float = 400.0):
        self.win  = int(sr * window_ms / 1000)

        sos1      = _design_sos('hs', 1681.0, gain_db=4.0, q=0.707, sr=sr)
        sos2      = _design_sos('hp', 100.0,  q=0.5,        sr=sr)
        zi1       = sp.sosfilt_zi(sos1)
        zi2       = sp.sosfilt_zi(sos2)
        self.sos1 = sos1;  self.zi1 = np.stack([zi1, zi1], axis=2)
        self.sos2 = sos2;  self.zi2 = np.stack([zi2, zi2], axis=2)

        self._ring  = np.zeros(self.win, dtype=np.float64)
        self._rptr  = 0
        self._lock  = threading.Lock()
        self._lufs  = -70.0

    def push(self, x: np.ndarray) -> None:
        f1, self.zi1 = sp.sosfilt(
            self.sos1, np.ascontiguousarray(x.T, dtype=np.float64), zi=self.zi1
        )
        f2, self.zi2 = sp.sosfilt(self.sos2, f1, zi=self.zi2)
        ms   = np.mean(f2 ** 2, axis=0)
        N    = len(ms);  win = self.win;  rptr = self._rptr
        space = win - rptr
        if N <= space:
            self._ring[rptr:rptr+N] = ms;  rptr += N
        else:
            self._ring[rptr:]   = ms[:space]
            self._ring[:N-space] = ms[space:];  rptr = N - space
        self._rptr = rptr % win
        lufs = -0.691 + 10.0 * np.log10(max(float(np.mean(self._ring)), 1e-10))
        with self._lock:
            self._lufs = lufs

    @property
    def lufs(self) -> float:
        with self._lock:
            return self._lufs

    def print_meter(self) -> None:
        lv   = max(self._lufs, -70.0)
        bars = int((lv + 70) / 70 * 30)
        bar  = '#' * bars + '-' * (30 - bars)
        sys.stdout.write(f"\r  LUFS {lv:6.1f} [{bar}]  ")
        sys.stdout.flush()