"""
filters.py -- Stateful Biquad Filters and Linkwitz-Riley Crossovers
===================================================================
These are kept in Python because scipy.sosfilt is already highly optimised
(C under the hood) and the filter count is small enough that Python overhead
is negligible compared to the C++ reverb / limiter.
"""

import numpy as np
from scipy import signal as sp


SR = 44100


def _design_sos(ftype: str, freq: float, gain_db: float = 0.0,
                q: float = 0.707, sr: int = SR) -> np.ndarray:
    """Design a single biquad returned as (1,6) SOS row."""
    A   = 10 ** (gain_db / 40.0)
    w0  = 2 * np.pi * freq / sr
    cw  = np.cos(w0);  sw = np.sin(w0);  alp = sw / (2 * q)

    if ftype == 'lp':
        b = [(1-cw)/2,  1-cw,     (1-cw)/2]
        a = [1+alp,    -2*cw,     1-alp]
    elif ftype == 'hp':
        b = [(1+cw)/2, -(1+cw),   (1+cw)/2]
        a = [1+alp,    -2*cw,     1-alp]
    elif ftype == 'peak':
        b = [1+alp*A,  -2*cw,     1-alp*A]
        a = [1+alp/A,  -2*cw,     1-alp/A]
    elif ftype == 'ls':
        b = [A*((A+1)-(A-1)*cw+2*np.sqrt(A)*alp),
              2*A*((A-1)-(A+1)*cw),
              A*((A+1)-(A-1)*cw-2*np.sqrt(A)*alp)]
        a = [(A+1)+(A-1)*cw+2*np.sqrt(A)*alp,
             -2*((A-1)+(A+1)*cw),
              (A+1)+(A-1)*cw-2*np.sqrt(A)*alp]
    elif ftype == 'hs':
        b = [A*((A+1)+(A-1)*cw+2*np.sqrt(A)*alp),
             -2*A*((A-1)+(A+1)*cw),
              A*((A+1)+(A-1)*cw-2*np.sqrt(A)*alp)]
        a = [(A+1)-(A-1)*cw+2*np.sqrt(A)*alp,
              2*((A-1)-(A+1)*cw),
              (A+1)-(A-1)*cw-2*np.sqrt(A)*alp]
    else:
        raise ValueError(f"Unknown filter type: {ftype!r}")

    b = np.array(b, dtype=np.float64) / a[0]
    a = np.array(a, dtype=np.float64) / a[0]
    return np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])


def sanitize(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in-place -- prevents IIR state corruption."""
    if not np.isfinite(x).all():
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


class Biquad:
    """Stateful stereo biquad using scipy.sosfilt with persistent zi."""

    def __init__(self, ftype: str, freq: float,
                 gain_db: float = 0.0, q: float = 0.707):
        self.sos = _design_sos(ftype, freq, gain_db, q)
        zi1      = sp.sosfilt_zi(self.sos)
        self.zi  = np.stack([zi1, zi1], axis=2)   # (1,2,2) -- secsxregsxchans

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out, self.zi = sp.sosfilt(
            self.sos, np.ascontiguousarray(x.T, dtype=np.float64), zi=self.zi
        )
        return sanitize(np.ascontiguousarray(out.T).astype(np.float32))


class DCBlocker:
    """5 Hz highpass -- removes DC offset before any nonlinear stage."""

    def __init__(self, sr: int = SR):
        sos      = _design_sos('hp', 5.0, q=0.5, sr=sr)
        zi       = sp.sosfilt_zi(sos)
        self.sos = sos
        self.zi  = np.stack([zi, zi], axis=2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out, self.zi = sp.sosfilt(
            self.sos, np.ascontiguousarray(x.T, dtype=np.float64), zi=self.zi
        )
        return sanitize(np.ascontiguousarray(out.T).astype(np.float32))


class LinkwitzRileyCrossover:
    """
    LR-4 crossover pair (two cascaded 2nd-order Butterworth = -24 dB/oct).
    Sums flat (no magnitude ripple at crossover frequency).
    Returns (low, high) band signals.
    """

    def __init__(self, crossover_hz: float, sr: int = SR):
        # LR-4 = two cascaded Butterworth LP/HP (q=0.5 for Butterworth 2nd order)
        self.lp = [Biquad('lp', crossover_hz, q=0.5),
                   Biquad('lp', crossover_hz, q=0.5)]
        self.hp = [Biquad('hp', crossover_hz, q=0.5),
                   Biquad('hp', crossover_hz, q=0.5)]

    def __call__(self, x: np.ndarray):
        lo = hi = x
        for f in self.lp: lo = f(lo)
        for f in self.hp: hi = f(hi)
        return lo, hi


class CinemaDelay:
    """
    Two-tap slapback delay (cinema early reflections).
    Tap 1 ~= 42 ms (rear wall), Tap 2 ~= 135 ms (ceiling).
    L/R taps offset by a few ms for natural stereo spread.
    """

    def __init__(self, t1_ms: float = 42, t2_ms: float = 135,
                 g1: float = 0.15,  g2: float = 0.10,
                 lp_hz: float = 2600, sr: int = SR):
        max_ms   = max(t1_ms, t2_ms) + 20
        self.sz  = int(sr * max_ms / 1000) + 1
        self.buf = np.zeros((self.sz, 2), dtype=np.float32)
        self.ptr = 0

        self.d1L = int(sr * t1_ms        / 1000)
        self.d1R = int(sr * (t1_ms + 5)  / 1000)
        self.d2L = int(sr * t2_ms        / 1000)
        self.d2R = int(sr * (t2_ms + 11) / 1000)
        self.g1, self.g2 = g1, g2

        lp = _design_sos('lp', lp_hz, q=0.6, sr=sr)
        zi = sp.sosfilt_zi(lp)
        self.lp_sos = lp
        self.lp_zi  = np.stack([zi, zi], axis=2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        N   = len(x);  sz = self.sz;  p = self.ptr
        idx   = (p + np.arange(N)) % sz
        idx1L = (p - self.d1L + np.arange(N)) % sz
        idx1R = (p - self.d1R + np.arange(N)) % sz
        idx2L = (p - self.d2L + np.arange(N)) % sz
        idx2R = (p - self.d2R + np.arange(N)) % sz

        t1L = self.buf[idx1L, 0];  t1R = self.buf[idx1R, 1]
        t2L = self.buf[idx2L, 0];  t2R = self.buf[idx2R, 1]

        self.buf[idx, 0] = x[:, 0];  self.buf[idx, 1] = x[:, 1]
        self.ptr = (p + N) % sz

        out        = np.empty_like(x)
        out[:, 0]  = x[:, 0] + self.g1 * t1L + self.g2 * t2L
        out[:, 1]  = x[:, 1] + self.g1 * t1R + self.g2 * t2R

        out_f, self.lp_zi = sp.sosfilt(
            self.lp_sos, np.ascontiguousarray(out.T, dtype=np.float64),
            zi=self.lp_zi
        )
        return sanitize(np.ascontiguousarray(out_f.T).astype(np.float32))