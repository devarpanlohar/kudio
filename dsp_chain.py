"""
dsp_chain.py -- Kudio Cinema DSP Chain v3.1
=============================================

SIGNAL PATH
-----------
  Raw PCM
    |
  [DC Blocker]
    |
  [6-Band Parametric EQ]          <-- NEW: user-adjustable
    |
  [LR-4 Multiband Split x3]       -- 4 bands: SUB/BASS/MID/HIGH
  [Per-band Compressor]
  [Per-band EQ]
  [Recombine + safety clip]
    |
  [Bass Spatial Enhancer]         <-- NEW: sub-bass width + punch
    |
  [SpectralShaper]     C++
    |
  [PsychoStereo]       C++
    |
  [CinemaDelay]
    |
  [FDNReverb 32-line]  C++
    |
  [HarmonicExciter]    C++
    |
  [Loudness Normalizer]           <-- NEW: Quiet/Normal/Loud modes
    |
  [TruePeakLimiter]    C++
    |
  [LUFS Meter]
    |
  Output
"""

import numpy as np
import warnings
from filters   import DCBlocker, Biquad, LinkwitzRileyCrossover, CinemaDelay, sanitize
from compressor import BandComp
from meters    import LUFSMeter

SR    = 44100
CHUNK = 1024

# Loudness normalization targets (LUFS)
NORM_TARGETS = {
    'quiet':  -23.0,   # preserve dynamics
    'normal': -14.0,   # balanced (default)
    'loud':   -11.0,   # noisy environments
}

try:
    import kudio_dsp as _cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    warnings.warn(
        "\n  kudio_dsp C++ engine not found.\n"
        "  Audio will play but without C++ acceleration.\n"
        "  To build: run build.sh",
        stacklevel=2
    )


# ── Pure-Python fallbacks ──────────────────────────────────────────────────────

def _py_soft_clip(x, drive=1.1, dry=0.70):
    wet = (np.tanh(x * drive) / np.tanh(drive)).astype(np.float32)
    return (dry * x + (1.0 - dry) * wet).astype(np.float32)


class _PyTruePeakLimiter:
    def __init__(self, ceil_db=-1.0, la_ms=3.0, rel_ms=80, sr=SR):
        self.ceil    = 10 ** (ceil_db / 20.0)
        self.la      = int(2 * sr * la_ms / 1000)
        self._la_buf = np.zeros((max(self.la, 1), 2), dtype=np.float32)
        self.gain    = 1.0
        self.rel     = float(np.exp(-1.0 / (sr * rel_ms / 1000.0)))
        from scipy import signal as sp
        self._os_h  = sp.firwin(64, 0.45)
        self._os_zi = np.zeros((63, 2))

    def process(self, x):
        from scipy import signal as sp
        N  = len(x); la = self.la; ceil = self.ceil; rel = self.rel
        up = np.zeros((N*2, 2), dtype=np.float32); up[::2] = x * 2.0
        upL, zL = sp.lfilter(self._os_h, [1.0], up[:,0], zi=self._os_zi[:,0])
        upR, zR = sp.lfilter(self._os_h, [1.0], up[:,1], zi=self._os_zi[:,1])
        self._os_zi[:,0] = zL; self._os_zi[:,1] = zR
        up[:,0] = upL.astype(np.float32); up[:,1] = upR.astype(np.float32)
        delayed = np.concatenate([self._la_buf, up], axis=0)[:N*2]
        self._la_buf = up[-la:].copy() if la > 0 else self._la_buf
        peaks  = np.maximum(np.abs(up[:,0]), np.abs(up[:,1]))
        needed = np.where(peaks > ceil, ceil / np.maximum(peaks, 1e-12), 1.0)
        needed = np.minimum(needed, 1.0)
        _rel = rel; _omr = 1.0 - rel
        def _gs(g, n): ng = n if n < g else g * _rel + _omr; return ng if ng < 1.0 else 1.0
        seed = np.concatenate([[self.gain], needed])
        garr = np.frompyfunc(_gs, 2, 1).accumulate(seed, dtype=np.object_).astype(np.float64)[1:]
        self.gain = float(garr[-1])
        return sanitize((delayed * garr[:, np.newaxis]).astype(np.float32)[::2])


# ── New: Bass Spatial Enhancer ─────────────────────────────────────────────────

class BassSpatialEnhancer:
    """
    Adds spatial width and punch to sub-bass.
    Uses a single stereo Biquad LP (Biquad expects (N,2) input).
    """
    def __init__(self, sr=SR, width=0.4, punch=0.3):
        self.width = width
        self.punch = punch
        self._lp   = Biquad('lp', 120, q=0.6)   # stereo LP -- correct shape

    def process(self, x):
        # x is (N,2) -- pass full stereo to LP
        sub  = self._lp(x)                        # (N,2) low band
        mid  = (sub[:, 0] + sub[:, 1]) * 0.5     # mono sum
        side = (sub[:, 0] - sub[:, 1]) * 0.5     # stereo diff
        mid_punchy = np.tanh(mid * (1.0 + self.punch)).astype(np.float32)
        side_wide  = (side * (1.0 + self.width)).astype(np.float32)
        out = x.copy()
        out[:, 0] += (mid_punchy + side_wide) * 0.35
        out[:, 1] += (mid_punchy - side_wide) * 0.35
        return sanitize(out)


# ── New: 6-Band Parametric EQ ─────────────────────────────────────────────────

class ParametricEQ6:
    """
    6-band user EQ. Gains in dB, default 0 (flat).

    Bands: Bass(60), LowMid(150), Mid(400), UpperMid(1k), Presence(2.4k), Treble(15k)
    """
    BANDS = [
        ('bass',      'peak', 60,    1.4),
        ('low_mid',   'peak', 150,   1.2),
        ('mid',       'peak', 400,   1.0),
        ('upper_mid', 'peak', 1000,  1.0),
        ('presence',  'peak', 2400,  1.1),
        ('treble',    'hs',   15000, 0.7),
    ]

    def __init__(self, sr=SR, **gains_db):
        """
        gains_db: keyword args like bass=3.0, treble=-2.0
        Any unspecified band defaults to 0.0 dB (flat).
        """
        self._filters = {}
        for name, ftype, freq, q in self.BANDS:
            gain = gains_db.get(name, 0.0)
            if abs(gain) > 0.01:
                self._filters[name] = Biquad(ftype, freq, gain_db=gain, q=q)
            else:
                self._filters[name] = None

    def set_gain(self, band: str, gain_db: float, sr=SR):
        """Hot-swap a band filter at runtime."""
        for name, ftype, freq, q in self.BANDS:
            if name == band:
                if abs(gain_db) > 0.01:
                    self._filters[name] = Biquad(ftype, freq, gain_db=gain_db, q=q)
                else:
                    self._filters[name] = None
                return

    def process(self, x):
        for name, f in self._filters.items():
            if f is not None:
                x = f(x)
        return x


# ── New: Loudness Normalizer ───────────────────────────────────────────────────

class LoudnessNormalizer:
    """
    Real-time gain rider that targets a loudness set-point.

    - Measures short-term LUFS in a 3-second sliding window
    - Smoothly adjusts gain toward target (time constant ~4s)
    - Hard ceiling so it can never clip into the limiter
    - 'loud' mode adds a gentle additional compression pass
    """
    def __init__(self, mode='normal', sr=SR):
        self.target  = NORM_TARGETS.get(mode, -14.0)
        self.mode    = mode
        self._gain   = 1.0          # current linear gain
        self._smooth = float(np.exp(-1.0 / (sr * 4.0)))   # 4s time constant
        self._max_gain = 10.0 ** (12.0 / 20.0)            # +12 dB ceiling
        self._min_gain = 10.0 ** (-24.0 / 20.0)           # -24 dB floor
        # Short-term LUFS window (3s)
        self._win   = sr * 3
        self._ring  = np.zeros(self._win, dtype=np.float64)
        self._ptr   = 0
        self._n     = 0             # chunks processed (skip first few)

    def process(self, x: np.ndarray) -> np.ndarray:
        self._n += 1
        # Update ring buffer with RMS power
        pwr = float(np.mean(x.astype(np.float64) ** 2))
        N   = len(x)
        idx = np.arange(self._ptr, self._ptr + N) % self._win
        self._ring[idx] = pwr
        self._ptr = (self._ptr + N) % self._win

        # Only adjust gain after 2s of audio (ring partially filled)
        if self._n * CHUNK > SR * 2:
            mean_pwr = float(np.mean(self._ring))
            if mean_pwr > 1e-10:
                current_lufs = -0.691 + 10.0 * np.log10(mean_pwr)
                deficit_db   = self.target - current_lufs
                # Gentle approach: move 25% toward target per update
                target_gain  = self._gain * (10.0 ** (deficit_db * 0.25 / 20.0))
                target_gain  = np.clip(target_gain, self._min_gain, self._max_gain)
                # Smooth the gain with IIR
                self._gain   = self._smooth * self._gain + (1 - self._smooth) * target_gain

        out = (x * self._gain).astype(np.float32)

        # Extra gentle compression for 'loud' mode
        if self.mode == 'loud':
            ceiling = 0.85
            over    = np.abs(out) > ceiling
            out[over] = np.sign(out[over]) * (
                ceiling + (np.abs(out[over]) - ceiling) * 0.3
            )

        return sanitize(out)

    @property
    def current_gain_db(self):
        return 20.0 * np.log10(max(self._gain, 1e-12))


# ── CinemaDSP ─────────────────────────────────────────────────────────────────

class CinemaDSP:
    """
    Full cinema DSP chain.

    Parameters
    ----------
    reverb_mode  : "hall" | "cathedral" | "plate" | "chamber"
    sr           : sample rate
    norm_mode    : "quiet" | "normal" | "loud"
    eq_gains     : dict of band->dB, e.g. {"bass": 3.0, "treble": -2.0}
    """

    def __init__(self, reverb_mode='hall', sr=SR,
                 norm_mode='normal', eq_gains=None):
        self.sr = sr

        # Python layers
        self.dc  = DCBlocker(sr)

        # 6-band user EQ (flat by default)
        self.user_eq = ParametricEQ6(sr, **(eq_gains or {}))

        # LR-4 crossovers
        self.xo_sub_bas = LinkwitzRileyCrossover(80,   sr)
        self.xo_bas_mid = LinkwitzRileyCrossover(300,  sr)
        self.xo_mid_hi  = LinkwitzRileyCrossover(4000, sr)

        # Per-band compressors
        self.c_sub = BandComp(-8,  4.0,  5,  80,  0.0, sr)
        self.c_bas = BandComp(-12, 2.5, 12, 120,  1.5, sr)
        self.c_mid = BandComp(-18, 1.5, 30, 200,  0.5, sr)
        self.c_hi  = BandComp(-15, 2.0,  8, 100,  0.0, sr)

        # Per-band EQ
        self.eq_sub = [Biquad('hp',   28, q=0.6),
                       Biquad('peak', 52, gain_db=3.0, q=0.75)]
        self.eq_bas = [Biquad('peak', 120, gain_db=3.5, q=0.85),
                       Biquad('peak', 240, gain_db=-3.0, q=1.0)]
        self.eq_mid = [Biquad('peak', 350,  gain_db=-3.5, q=1.1),
                       Biquad('peak', 800,  gain_db=0.5,  q=1.5),
                       Biquad('peak', 3200, gain_db=2.0,  q=1.0)]
        self.eq_hi  = [Biquad('hs',   8000,  gain_db=3.5, q=0.7),
                       Biquad('peak', 14000, gain_db=2.5, q=0.9)]

        # Bass spatial
        self.bass_spatial = BassSpatialEnhancer(sr, width=0.4, punch=0.3)

        # Cinema delay
        self.delay = CinemaDelay(t1_ms=42, t2_ms=135,
                                  g1=0.15, g2=0.10, lp_hz=2600, sr=sr)

        # Loudness normalizer
        self.normalizer = LoudnessNormalizer(mode=norm_mode, sr=sr)

        # C++ layers
        if _CPP_AVAILABLE:
            self.spectral = _cpp.SpectralShaper("cinema", sr)
            self.psycho   = _cpp.PsychoStereo(
                                haas_ms=1.2, crossfeed=0.18,
                                cf_delay_ms=0.35, ild_cutoff=1400, sr=sr)
            self.reverb   = _cpp.FDNReverb(reverb_mode, 0.25, 1.0, sr)
            self.harmonic = _cpp.HarmonicExciter(
                                drive=0.35, even_mix=0.60, odd_mix=0.40,
                                wet=0.22, excite_hz=2000, sr=sr)
        else:
            self.spectral = None
            self.psycho   = None
            self.reverb   = None
            self.harmonic = None
        if _CPP_AVAILABLE:
            self.limiter = _cpp.TruePeakLimiter(-1.0, 3.0, 80.0, sr, CHUNK)
        else:
            self.limiter = _PyTruePeakLimiter(-1.0, 3.0, 80, sr)

        self.meter = LUFSMeter(sr)

    @staticmethod
    def _chain(x, filters):
        for f in filters:
            x = f(x)
        return x

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32)

        # 1. DC block
        x = self.dc(x)

        # 2. 6-band user EQ
        x = self.user_eq.process(x)

        # 3. Multiband split
        sub, tmp  = self.xo_sub_bas(x)
        bas, tmp2 = self.xo_bas_mid(tmp)
        mid, hi   = self.xo_mid_hi(tmp2)

        # 4. Per-band compress
        sub = self.c_sub(sub); bas = self.c_bas(bas)
        mid = self.c_mid(mid); hi  = self.c_hi(hi)

        # 5. Per-band EQ
        sub = self._chain(sub, self.eq_sub)
        bas = self._chain(bas, self.eq_bas)
        mid = self._chain(mid, self.eq_mid)
        hi  = self._chain(hi,  self.eq_hi)

        # 6. Recombine
        mix = sub + bas + mid + hi
        np.clip(mix, -1.5, 1.5, out=mix)

        # 7. Bass spatial enhancement
        mix = self.bass_spatial.process(mix)

        # 8. Spectral shaping (C++)
        if self.spectral is not None:
            mix = sanitize(self.spectral.process(mix))

        # 9. Psychoacoustic stereo (C++)
        if self.psycho is not None:
            mix = sanitize(self.psycho.process(mix))

        # 10. Cinema slapback delay
        mix = self.delay(mix)

        # 11. FDN reverb (C++) -- dry=1.0, wet=0.25
        if self.reverb is not None:
            mix = sanitize(self.reverb.process(mix))
        else:
            mix = _py_soft_clip(mix)

        # 12. Harmonic exciter (C++)
        if self.harmonic is not None:
            mix = sanitize(self.harmonic.process(mix))

        # 13. Loudness normalization
        mix = self.normalizer.process(mix)

        # 14. True-peak limiter
        mix = sanitize(self.limiter.process(mix))

        # 15. LUFS meter (side-chain)
        self.meter.push(mix)

        return mix.astype(np.float32)