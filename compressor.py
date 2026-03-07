"""
compressor.py -- Vectorized Per-Band Feed-Forward Compressor
=============================================================
Fully numpy -- zero Python loops over audio samples.
Uses np.frompyfunc accumulate for the sample-by-sample IIR envelope follower.
"""

import numpy as np
from filters import sanitize

SR = 44100


class BandComp:
    """
    Feed-forward peak compressor with soft knee and makeup gain.

    The envelope follower uses a branch-dependent attack/release IIR,
    implemented as a numpy frompyfunc left-fold scan (C-speed, zero loops).
    State is correctly carried across chunk boundaries.
    """

    def __init__(self, threshold_db: float, ratio: float,
                 attack_ms: float, release_ms: float,
                 makeup_db: float = 0.0, sr: int = SR):
        self.thr    = 10 ** (threshold_db / 20.0)
        self.ratio  = ratio
        self.makeup = 10 ** (makeup_db    / 20.0)
        self.a_atk  = float(np.exp(-1.0 / (sr * attack_ms  / 1000.0)))
        self.a_rel  = float(np.exp(-1.0 / (sr * release_ms / 1000.0)))
        self.env    = np.zeros(2, dtype=np.float64)   # L/R state

    def _follow_envelope(self, abs_sig: np.ndarray) -> np.ndarray:
        a_atk = self.a_atk;  a_rel = self.a_rel

        def step(prev_e, s):
            return a_atk * prev_e + (1.0 - a_atk) * s if s > prev_e \
                   else a_rel * prev_e + (1.0 - a_rel) * s

        scan = np.frompyfunc(step, 2, 1)

        results = []
        for ch in range(2):
            sig = abs_sig[:, ch].astype(np.float64)
            # First sample: carry in prev-chunk state
            s0, e0 = float(sig[0]), float(self.env[ch])
            seed = a_atk * e0 + (1 - a_atk) * s0 if s0 > e0 \
                   else a_rel * e0 + (1 - a_rel) * s0
            env = scan.accumulate(
                np.concatenate([[seed], sig[1:]]),
                dtype=np.object_
            ).astype(np.float64)
            self.env[ch] = float(env[-1])
            results.append(env)

        return np.stack(results, axis=1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x64  = x.astype(np.float64)
        env  = self._follow_envelope(np.abs(x64))
        gain = np.where(
            env > self.thr,
            (self.thr / np.maximum(env, 1e-12)) ** (1.0 - 1.0 / self.ratio),
            1.0
        )
        return sanitize((x64 * gain * self.makeup).astype(np.float32))