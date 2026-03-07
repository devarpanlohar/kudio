#pragma once
#include "common.hpp"
#include <cmath>
/**
 * spectral.hpp — Cinematic Spectral Shaper
 * ═════════════════════════════════════════
 *
 * WHAT IT DOES
 * ────────────
 *   A parallel bank of 6 parametric biquad sections, each targeting a
 *   perceptually critical frequency range.  Together they impose the
 *   characteristic tonal balance of large-format cinema mixing:
 *
 *   ┌──────────┬────────────┬──────────────────────────────────────────┐
 *   │ Band     │ Frequency  │ Purpose                                  │
 *   ├──────────┼────────────┼──────────────────────────────────────────┤
 *   │ Sub weight│ 35–50 Hz  │ Deep physical body, felt in chest        │
 *   │ Warmth   │ 200–280 Hz │ Tonal richness, "analogue" density       │
 *   │ Mud cut  │ 380–450 Hz │ Remove boxiness / congestion             │
 *   │ Presence │ 2.5–4.5 kHz│ Vocal clarity, snare crack, definition  │
 *   │ Brilliance│ 7–10 kHz  │ Cymbal shimmer, bow attack, air detail   │
 *   │ Air      │ 14–18 kHz  │ Ultra-high shelf, "open ceiling" feel    │
 *   └──────────┴────────────┴──────────────────────────────────────────┘
 *
 * PRESETS
 * ───────
 *   "cinema"    : Loud, wide, deep bass, clear mids (default)
 *   "warm"      : Lifted sub/warmth, pulled back presence — melancholic
 *   "bright"    : More presence + air — detail forward
 *   "neutral"   : Subtle, reference-flat tonal touches only
 *
 * IMPLEMENTATION
 * ──────────────
 *   Each section is a Direct Form II transposed biquad (numerically stable).
 *   Stereo state maintained as parallel L/R registers.  All 6 sections are
 *   applied IN SERIES so their gains ADD in dB (not multiply).
 *
 *   Using series biquad EQ (not FFT-based) avoids frame-boundary artefacts
 *   and maintains phase continuity across chunks — essential for real-time use.
 */

#include <vector>
#include <cmath>
#include <string>
#include <array>
#include <stdexcept>
#include <algorithm>

struct BiquadSection {
    // Coefficients
    float b0 = 1, b1 = 0, b2 = 0;
    float a1 = 0, a2 = 0;
    // DF-II transposed state [L, R]
    float s1[2] = {}, s2[2] = {};

    // Peak/bell EQ
    void set_peak(float freq, float gain_db, float q, int sr) noexcept {
        const float A   = std::pow(10.0f, gain_db / 40.0f);
        const float w0  = 2.0f * kPI_F * freq / sr;
        const float cw  = std::cos(w0);
        const float sw  = std::sin(w0);
        const float alp = sw / (2.0f * q);
        const float a0  = 1.0f + alp / A;
        b0 = (1.0f + alp * A) / a0;
        b1 = (-2.0f * cw)     / a0;
        b2 = (1.0f - alp * A) / a0;
        a1 = (-2.0f * cw)     / a0;
        a2 = (1.0f - alp / A) / a0;
    }

    // Low shelf
    void set_ls(float freq, float gain_db, float q, int sr) noexcept {
        const float A  = std::pow(10.0f, gain_db / 40.0f);
        const float w0 = 2.0f * kPI_F * freq / sr;
        const float cw = std::cos(w0);
        const float sw = std::sin(w0);
        const float ap = sw / (2.0f * q);
        const float sq = 2.0f * std::sqrt(A) * ap;
        const float a0 = (A+1) + (A-1)*cw + sq;
        b0 = A*((A+1) - (A-1)*cw + sq)         / a0;
        b1 = 2.0f*A*((A-1) - (A+1)*cw)         / a0;
        b2 = A*((A+1) - (A-1)*cw - sq)          / a0;
        a1 = -2.0f*((A-1) + (A+1)*cw)           / a0;
        a2 = ((A+1) + (A-1)*cw - sq)             / a0;
    }

    // High shelf
    void set_hs(float freq, float gain_db, float q, int sr) noexcept {
        const float A  = std::pow(10.0f, gain_db / 40.0f);
        const float w0 = 2.0f * kPI_F * freq / sr;
        const float cw = std::cos(w0);
        const float sw = std::sin(w0);
        const float ap = sw / (2.0f * q);
        const float sq = 2.0f * std::sqrt(A) * ap;
        const float a0 = (A+1) - (A-1)*cw + sq;
        b0 =  A*((A+1) + (A-1)*cw + sq)         / a0;
        b1 = -2.0f*A*((A-1) + (A+1)*cw)         / a0;
        b2 =  A*((A+1) + (A-1)*cw - sq)          / a0;
        a1 =  2.0f*((A-1) - (A+1)*cw)            / a0;
        a2 = ((A+1) - (A-1)*cw - sq)              / a0;
    }

    inline float tick(float x, int ch) noexcept {
        const float y = b0*x + s1[ch];
        s1[ch] = b1*x - a1*y + s2[ch];
        s2[ch] = b2*x - a2*y;
        return y;
    }
};


class SpectralShaper {
public:
    struct Params {
        float sub_db;        // sub weight  (< 80 Hz shelf)
        float warmth_db;     // warmth      (~240 Hz peak)
        float mud_db;        // mud cut     (~420 Hz peak, negative)
        float presence_db;   // presence    (~3.2 kHz peak)
        float brilliance_db; // brilliance  (~8 kHz shelf)
        float air_db;        // air         (~15 kHz shelf)
    };

    static Params preset(const std::string& name) {
        if (name == "warm")    return { 2.5f,  3.5f, -2.5f,  1.0f,  1.5f,  1.5f };
        if (name == "bright")  return { 1.5f,  1.0f, -2.0f,  3.5f,  3.5f,  3.0f };
        if (name == "neutral") return { 1.0f,  1.5f, -1.5f,  1.5f,  1.0f,  1.0f };
        // default: "cinema"
        return                  { 2.0f,  2.5f, -2.5f,  2.5f,  2.5f,  2.0f };
    }

    /**
     * @param p     spectral parameter set  (use preset() or custom)
     * @param sr    sample rate
     */
    explicit SpectralShaper(const Params& p = preset("cinema"), int sr = 44100)
    {
        sections_[0].set_ls   ( 65.0f,  p.sub_db,        0.60f, sr); // sub weight
        sections_[1].set_peak (240.0f,  p.warmth_db,     0.80f, sr); // warmth
        sections_[2].set_peak (420.0f,  p.mud_db,        1.20f, sr); // mud cut
        sections_[3].set_peak (3200.0f, p.presence_db,   0.90f, sr); // presence
        sections_[4].set_hs   (7500.0f, p.brilliance_db, 0.70f, sr); // brilliance
        sections_[5].set_hs  (14000.0f, p.air_db,        0.60f, sr); // air
    }

    /**
     * Process one stereo block in-place.
     * @param io  interleaved (L,R,...) float32, N*2 values
     * @param N   stereo frame count
     */
    void process(float* RESTRICT io, int N) noexcept {
        for (int n = 0; n < N; ++n) {
            float L = io[n*2];
            float R = io[n*2+1];
            for (auto& s : sections_) {
                L = s.tick(L, 0);
                R = s.tick(R, 1);
            }
            io[n*2]   = L;
            io[n*2+1] = R;
        }
    }

    /** Re-configure one band at runtime (0=sub,1=warmth,2=mud,3=pres,4=brill,5=air) */
    void set_band(int band, float gain_db, int sr = 44100) {
        if (band < 0 || band > 5) return;
        const float freqs[6] = {65.0f, 240.0f, 420.0f, 3200.0f, 7500.0f, 14000.0f};
        const float qs[6]    = {0.60f,  0.80f,  1.20f,   0.90f,   0.70f,   0.60f};
        if (band == 0) sections_[0].set_ls  (freqs[0], gain_db, qs[0], sr);
        else if (band == 4) sections_[4].set_hs(freqs[4], gain_db, qs[4], sr);
        else if (band == 5) sections_[5].set_hs(freqs[5], gain_db, qs[5], sr);
        else sections_[band].set_peak(freqs[band], gain_db, qs[band], sr);
    }

private:
    std::array<BiquadSection, 6> sections_;
};
