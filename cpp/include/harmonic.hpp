#pragma once
#include "common.hpp"
#include <cmath>
/**
 * harmonic.hpp — Parallel Harmonic Exciter
 * ═════════════════════════════════════════
 *
 * WHAT IS A HARMONIC EXCITER?
 * ───────────────────────────
 *   A harmonic exciter (like the Aphex Aural Exciter or SPL Vitalizer) adds
 *   subtle harmonic distortion products — overtones that weren't in the original
 *   signal but are musically related to it.  This creates a sense of "air",
 *   "presence", and "brightness" without actually boosting the high frequencies
 *   in a linear way.
 *
 * TUBE MODEL
 * ──────────
 *   Valve / tube amplifiers produce characteristic even-order harmonics (2nd, 4th)
 *   because their transfer function is asymmetric.  This adds "warmth".
 *   Odd harmonics (3rd) add "presence" and "bite" (like a pushed transistor amp).
 *
 *   Even harmonics:  x_even = drive * (x^2 + 0.2 * x^4)   (always positive)
 *   Odd harmonics:   x_odd  = drive * tanh(x^3)            (always odd symmetry)
 *
 * ARCHITECTURE  (fully parallel, no series distortion)
 * ─────────────────────────────────────────────────────
 *   Input ──┬──────────────────────────────────── dry (1.0)
 *            │
 *            └─▶ HP biquad (excite_hz) ──▶ waveshaper ──▶ LP biquad ──▶ wet blend
 *
 *   The highpass isolates high-midrange and treble (>2 kHz by default) so
 *   the harmonics only appear in the sparkly top end — not in bass where
 *   distortion sounds muddy.  The LP after the waveshaper removes ultrasonic
 *   content above ~18 kHz that could cause inter-modulation artifacts.
 *
 * Reference:
 *   Zölzer (2011) "DAFX: Digital Audio Effects", Ch. 5 (Nonlinear Processing)
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

// ─── Simple second-order highpass / lowpass biquad ──────────────────────────
struct Biquad2 {
    float b0 = 1, b1 = 0, b2 = 0;
    float a1 = 0, a2 = 0;
    // state per channel [L, R]
    float x1[2] = {}, x2[2] = {}, y1[2] = {}, y2[2] = {};

    void set_hp(float freq, float q, int sr) {
        const float w0  = 2.0f * kPI_F * freq / sr;
        const float cw  = std::cos(w0);
        const float sw  = std::sin(w0);
        const float alp = sw / (2.0f * q);
        const float a0  = 1.0f + alp;
        b0 = ((1.0f + cw) * 0.5f) / a0;
        b1 = (-(1.0f + cw))       / a0;
        b2 = b0;
        a1 = (-2.0f * cw)         / a0;
        a2 = (1.0f - alp)         / a0;
    }

    void set_lp(float freq, float q, int sr) {
        const float w0  = 2.0f * kPI_F * freq / sr;
        const float cw  = std::cos(w0);
        const float sw  = std::sin(w0);
        const float alp = sw / (2.0f * q);
        const float a0  = 1.0f + alp;
        b0 = ((1.0f - cw) * 0.5f) / a0;
        b1 = (1.0f - cw)           / a0;
        b2 = b0;
        a1 = (-2.0f * cw)          / a0;
        a2 = (1.0f - alp)          / a0;
    }

    inline float tick(float x, int ch) noexcept {
        const float y = b0*x + b1*x1[ch] + b2*x2[ch]
                             - a1*y1[ch] - a2*y2[ch];
        x2[ch] = x1[ch]; x1[ch] = x;
        y2[ch] = y1[ch]; y1[ch] = y;
        return y;
    }
};


class HarmonicExciter {
public:
    /**
     * @param drive        nonlinear drive amount  (0.0 – 1.0, default 0.35)
     *                     Controls how hard the waveshaper is pushed.
     * @param even_mix     even-harmonic weight  (0.0 – 1.0, default 0.6)
     *                     Higher → warmer, more "tube" character.
     * @param odd_mix      odd-harmonic weight   (0.0 – 1.0, default 0.4)
     *                     Higher → more "presence" and edge.
     * @param wet          exciter wet blend      (0.0 – 1.0, default 0.22)
     *                     Final level of added harmonics in the output.
     * @param excite_hz    high-pass knee of the exciter path  (default 2000 Hz)
     *                     Only frequencies above this are excited.
     * @param sr           sample rate
     */
    explicit HarmonicExciter(float drive     = 0.35f,
                             float even_mix  = 0.60f,
                             float odd_mix   = 0.40f,
                             float wet       = 0.22f,
                             float excite_hz = 2000.0f,
                             int   sr        = 44100)
        : drive_(drive), even_mix_(even_mix), odd_mix_(odd_mix),
          wet_(std::clamp(wet, 0.0f, 1.0f))
    {
        hp_.set_hp(excite_hz,  0.707f, sr);
        lp_.set_lp(18000.0f,   0.707f, sr);
    }

    /**
     * Process one stereo block in-place.
     * @param io  interleaved stereo float32 (L,R,L,R,...), N*2 values
     * @param N   stereo frame count
     */
    void process(float* RESTRICT io, int N) noexcept {
        for (int n = 0; n < N; ++n) {
            for (int ch = 0; ch < 2; ++ch) {
                const float x   = io[n*2 + ch];

                // High-pass to isolate excitation band
                const float hp  = hp_.tick(x, ch);

                // Waveshaper: tube-style even + odd harmonics
                const float hp2 = hp * hp;
                const float hp3 = hp2 * hp;
                const float hp4 = hp2 * hp2;

                const float even = drive_ * (hp2 + 0.18f * hp4);
                const float odd  = drive_ * std::tanh(hp3 * 2.0f) * 0.5f;

                const float harm = even_mix_ * even + odd_mix_ * odd;

                // Low-pass to remove ultrasonic content
                const float harm_lp = lp_.tick(harm, ch);

                io[n*2 + ch] = x + wet_ * harm_lp;
            }
        }
    }

    void set_drive(float d) noexcept { drive_ = std::clamp(d, 0.0f, 1.0f); }
    void set_wet  (float w) noexcept { wet_   = std::clamp(w, 0.0f, 1.0f); }

private:
    float drive_, even_mix_, odd_mix_, wet_;
    Biquad2 hp_, lp_;
};
