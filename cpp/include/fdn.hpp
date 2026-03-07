#pragma once
#include "common.hpp"
#include <cmath>
/**
 * fdn.hpp — 32-Line Schroeder + FDN Hybrid Reverb
 * ═══════════════════════════════════════════════════
 *
 * ARCHITECTURE
 * ────────────
 *   Pre-delay (mode-dependent)
 *       │
 *       ├──▶  [4 Schroeder feedback combs]  ──▶  sum  ──▶  early_out
 *       │
 *       └──▶  [4 allpass diffusers]
 *                     │
 *                     ▼
 *            [FDN-16 left  (lines 0-15)]  ──▶  late_L
 *            [FDN-16 right (lines 16-31)] ──▶  late_R
 *
 *   output = dry * in  +  wet_early * early_out  +  wet_late * (late_L / late_R)
 *
 * MIXING MATRIX  (FDN-16 per channel)
 * ────────────────────────────────────
 *   Uses the Fast Walsh–Hadamard Transform (FWHT).  For N=16 (a power of 2)
 *   this is exactly equivalent to multiplying by the 16×16 Hadamard matrix
 *   but requires only N·log₂(N) = 64 adds instead of N² = 256 mults.
 *   The transform is orthogonal, energy-preserving, and lossless — the ideal
 *   property for a reverb feedback network (no unwanted spectral colouration).
 *
 * SCHROEDER COMBS
 * ───────────────
 *   4 parallel feedback comb filters (Moorer style — damping LP inside loop).
 *   RT60 of the combs is set shorter than the FDN so they provide dense early
 *   reflections and then decay, leaving only the FDN tail audible.
 *
 * MODE PRESETS
 * ────────────
 *   "hall"      RT60 2.6s, pre-delay 22ms — orchestral concert hall
 *   "cathedral" RT60 5.0s, pre-delay 35ms — large stone cathedral
 *   "plate"     RT60 1.8s, pre-delay  4ms — vintage EMT-style plate
 *   "chamber"   RT60 1.1s, pre-delay 12ms — small live chamber / wood room
 *
 * References:
 *   Schroeder (1962) "Natural Sounding Artificial Reverberation"
 *   Jot & Chaigne (1991) "Digital Delay Networks for Designing Artificial Reverberators"
 *   Smith (2010) "Physical Audio Signal Processing" §2.3 FDN Reverb
 *   Zölzer (2011) "DAFX: Digital Audio Effects" Ch.7
 */

#include <vector>
#include <cmath>
#include <string>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include "allpass.hpp"

// ─── Fast Walsh–Hadamard Transform  (in-place, N must be power of 2) ──────────
// After the transform, divide each element by sqrt(N) to normalise.
// This is equivalent to left-multiplying by the N×N normalised Hadamard matrix.
static inline void fwht_normalised(float* RESTRICT v, int N) noexcept {
    for (int len = 1; len < N; len <<= 1) {
        for (int i = 0; i < N; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                float a = v[i + j];
                float b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
    const float norm = 1.0f / std::sqrt(static_cast<float>(N));
    for (int i = 0; i < N; ++i) v[i] *= norm;
}


// ─── 1-pole lowpass (used for FDN line damping & comb damping) ────────────────
struct OnePoleLP {
    float coeff = 0.0f;   // b0 = 1-a1
    float state = 0.0f;   // y[n-1]

    void set(float cutoff_hz, int sr) noexcept {
        // bilinear-ish 1-pole: a1 = exp(-2π·fc/sr)
        coeff = 1.0f - std::exp(-2.0f * kPI_F * cutoff_hz / sr);
    }

    inline float tick(float x) noexcept {
        state += coeff * (x - state);
        return state;
    }
};


// ─── Schroeder Feedback Comb Filter ──────────────────────────────────────────
// y[n] = x[n] + g * damp_lp(y[n - d])
struct SchroederComb {
    std::vector<float> buf;
    int   d = 0, sz = 0, ptr = 0;
    float g = 0.0f;
    OnePoleLP damp;

    void init(float delay_ms, float rt60_s, float damp_hz, int sr) {
        d   = static_cast<int>(delay_ms * sr / 1000.0f);
        sz  = d + 4;
        buf.assign(sz, 0.0f);
        ptr = 0;
        // RT60 gain: g = 10^(-3·d / (RT60·SR))
        g   = std::pow(10.0f, -3.0f * d / (rt60_s * sr));
        damp.set(damp_hz, sr);
    }

    inline float tick(float x) noexcept {
        const int read_idx = ((ptr - d) % sz + sz) % sz;
        const float y      = x + g * damp.tick(buf[read_idx]);
        buf[ptr % sz]      = y;
        ptr                = (ptr + 1) % sz;
        return y;
    }

    void reset() noexcept {
        std::fill(buf.begin(), buf.end(), 0.0f);
        damp.state = 0.0f;
        ptr = 0;
    }
};


// ─── FDN-16 (single channel) ─────────────────────────────────────────────────
// 16 delay lines with FWHT mixing + per-line damping LP + RT60 gain.
struct FDN16 {
    static constexpr int NLINES = 16;

    struct Line {
        std::vector<float> buf;
        int   d = 0, sz = 0, ptr = 0;
        float g = 0.0f;
        OnePoleLP damp;
    };

    std::array<Line, NLINES> lines;

    void init(const float delays_ms[NLINES], float rt60_s,
              float damp_hz, int sr) {
        for (int i = 0; i < NLINES; ++i) {
            auto& L = lines[i];
            L.d     = static_cast<int>(delays_ms[i] * sr / 1000.0f);
            L.sz    = L.d + 4;
            L.buf.assign(L.sz, 0.0f);
            L.ptr   = 0;
            L.g     = std::pow(10.0f, -3.0f * L.d / (rt60_s * sr));
            L.damp.set(damp_hz, sr);
        }
    }

    /**
     * Process one mono block.
     * @param in   input  (N,) samples
     * @param out  output (N,) samples  (accumulates — caller zero-inits)
     * @param tmp  scratch (NLINES,) float buffer
     */
    void process_block(const float* RESTRICT in,
                             float* RESTRICT out,
                             int N) noexcept {
        float taps[NLINES];

        for (int n = 0; n < N; ++n) {
            // 1. Read each delay tap
            for (int i = 0; i < NLINES; ++i) {
                auto& L     = lines[i];
                int read_i  = ((L.ptr - L.d) % L.sz + L.sz) % L.sz;
                taps[i]     = L.buf[read_i];
            }

            // 2. FWHT mix (lossless Hadamard rotation, in-place)
            fwht_normalised(taps, NLINES);

            // 3. Add input + apply RT60 gain + damping → write back
            for (int i = 0; i < NLINES; ++i) {
                auto& L        = lines[i];
                float fed      = taps[i] + in[n];
                // per-line LP damping BEFORE feedback → natural HF roll-off
                L.buf[L.ptr % L.sz] = L.g * L.damp.tick(fed);
                L.ptr = (L.ptr + 1) % L.sz;
            }

            // 4. Output = mean of all taps (sum pre-mix taps)
            float acc = 0.0f;
            for (int i = 0; i < NLINES; ++i) acc += taps[i];
            out[n] = acc * (1.0f / NLINES);
        }
    }

    void reset() noexcept {
        for (auto& L : lines) {
            std::fill(L.buf.begin(), L.buf.end(), 0.0f);
            L.damp.state = 0.0f;
            L.ptr = 0;
        }
    }
};


// ═════════════════════════════════════════════════════════════════════════════
//  HybridReverb32 — public API class
// ═════════════════════════════════════════════════════════════════════════════
class HybridReverb32 {
public:
    // ── Mode preset table ────────────────────────────────────────────────────
    struct Preset {
        float rt60;          // FDN late tail  (seconds)
        float early_rt60;    // Schroeder comb decay  (seconds)
        float pre_delay_ms;  // pre-delay before all reverb
        float damp_hz;       // FDN per-line damping LP cutoff
        float early_damp_hz; // comb damping LP cutoff
        float wet_early;     // early reflection gain (linear)
        float wet_late;      // late FDN gain (linear)
        float dry;           // dry gain
    };

    static Preset make_preset(const std::string& mode) {
        if (mode == "cathedral")
            return {5.0f, 1.5f, 35.0f, 3800.0f, 4500.0f, 0.18f, 0.24f, 0.78f};
        if (mode == "plate")
            return {1.8f, 0.9f,  4.0f, 6500.0f, 7000.0f, 0.22f, 0.20f, 0.78f};
        if (mode == "chamber")
            return {1.1f, 0.6f, 12.0f, 5500.0f, 6000.0f, 0.16f, 0.18f, 0.78f};
        // default: "hall"
        return    {2.6f, 1.0f, 22.0f, 5200.0f, 5800.0f, 0.15f, 0.22f, 0.78f};
    }

    /**
     * @param mode     "hall" | "cathedral" | "plate" | "chamber"
     * @param wet      overall wet mix gain override  (-1 = use preset)
     * @param dry      overall dry mix gain override  (-1 = use preset)
     * @param sr       sample rate
     * @param chunk    max chunk size (for scratch buffer allocation)
     */
    HybridReverb32(const std::string& mode = "hall",
                   float wet = -1.0f,
                   float dry = -1.0f,
                   int sr = 44100, int chunk = 1024)
        : sr_(sr), chunk_(chunk)
    {
        preset_ = make_preset(mode);
        if (wet >= 0.0f) { preset_.wet_early *= wet / 0.22f;
                           preset_.wet_late  *= wet / 0.22f; }
        if (dry >= 0.0f)   preset_.dry = dry;

        _init_predelay();
        _init_combs();
        _init_allpass();
        _init_fdn();

        // Scratch buffers
        tmpL_.resize(chunk_);
        tmpR_.resize(chunk_);
        mono_.resize(chunk_);
        comb_out_.resize(chunk_);
    }

    /**
     * Process one stereo block (N × 2 interleaved — L,R,L,R,...).
     * @param io   pointer to N*2 float32 interleaved samples (in/out)
     * @param N    number of stereo frames
     */
    void process(float* RESTRICT io, int N) noexcept {
        // ── 1. Pre-delay ───────────────────────────────────────────────
        _predelay(io, N);

        // ── 2. Sum to mono ─────────────────────────────────────────────
        for (int n = 0; n < N; ++n)
            mono_[n] = (io[n*2] + io[n*2+1]) * 0.5f;

        // ── 3. Allpass diffusion (4 stages, in-place on mono) ──────────
        for (auto& ap : ap_) ap.process_inplace(mono_.data(), N);

        // ── 4. Schroeder combs (early reflections) ─────────────────────
        std::fill(comb_out_.begin(), comb_out_.begin() + N, 0.0f);
        for (auto& c : combs_) {
            for (int n = 0; n < N; ++n)
                comb_out_[n] += c.tick(mono_[n]);
        }
        const float comb_norm = 1.0f / static_cast<float>(combs_.size());
        for (int n = 0; n < N; ++n) comb_out_[n] *= comb_norm;

        // ── 5. FDN late field (L and R independently tuned) ───────────
        std::fill(tmpL_.begin(), tmpL_.begin() + N, 0.0f);
        std::fill(tmpR_.begin(), tmpR_.begin() + N, 0.0f);
        fdnL_.process_block(mono_.data(), tmpL_.data(), N);
        fdnR_.process_block(mono_.data(), tmpR_.data(), N);

        // ── 6. Mix: dry + early + late ─────────────────────────────────
        const float we = preset_.wet_early;
        const float wl = preset_.wet_late;
        const float d  = preset_.dry;
        for (int n = 0; n < N; ++n) {
            const float L = io[n*2];
            const float R = io[n*2+1];
            const float e = comb_out_[n];
            io[n*2]   = d * L  +  we * e  +  wl * tmpL_[n];
            io[n*2+1] = d * R  +  we * e  +  wl * tmpR_[n];
        }
    }

    void reset() noexcept {
        std::fill(pd_buf_.begin(), pd_buf_.end(), 0.0f);
        pd_ptr_ = 0;
        for (auto& c : combs_) c.reset();
        for (auto& a : ap_)    a.reset();
        fdnL_.reset();
        fdnR_.reset();
    }

    // Accessors for runtime tweaking
    void set_wet(float w) noexcept {
        preset_.wet_early = w * 0.40f;
        preset_.wet_late  = w * 0.60f;
    }
    void set_dry(float d) noexcept { preset_.dry = d; }

private:
    int   sr_, chunk_;
    Preset preset_;

    // Pre-delay
    std::vector<float> pd_buf_;
    int pd_ptr_ = 0, pd_sz_ = 0;

    // 4 Schroeder feedback combs
    std::array<SchroederComb, 4> combs_;

    // 4 allpass diffusers (in series, before FDN)
    std::array<AllpassDelay, 4> ap_;

    // FDN-16 × 2 (independent L/R tuning for stereo decorrelation)
    FDN16 fdnL_, fdnR_;

    // scratch
    std::vector<float> tmpL_, tmpR_, mono_, comb_out_;

    // ── Delay line tables ────────────────────────────────────────────────────
    // Mutually prime ms values — no integer ratios → no metallic resonance.
    // L and R sets are staggered by ~1.5 ms each to decorrelate stereo image.
    static constexpr float FDN_L_MS[16] = {
        29.7f, 37.1f, 41.1f, 43.7f, 47.3f, 53.9f, 61.1f, 67.3f,
        71.3f, 79.7f, 83.1f, 89.3f, 97.1f,103.3f,107.7f,113.9f
    };
    static constexpr float FDN_R_MS[16] = {
        31.3f, 38.9f, 42.7f, 45.1f, 49.7f, 55.3f, 63.7f, 69.1f,
        73.7f, 81.3f, 85.7f, 91.1f, 99.7f,101.9f,109.3f,117.1f
    };
    // Schroeder comb delays (Moorer classic values, scaled to prime-ish ms)
    static constexpr float COMB_MS[4] = { 36.8f, 40.2f, 44.9f, 52.7f };
    // Allpass diffuser delays (short, for echo density buildup)
    static constexpr float AP_MS[4]   = { 12.7f,  6.3f,  3.1f,  1.7f };

    void _init_predelay() {
        pd_sz_ = static_cast<int>(preset_.pre_delay_ms * sr_ / 1000.0f) + 4;
        pd_buf_.assign(pd_sz_ * 2, 0.0f);  // stereo interleaved
        pd_ptr_ = 0;
    }

    void _predelay(float* RESTRICT io, int N) noexcept {
        if (pd_sz_ <= 4) return;   // pre-delay too short to matter
        const int d   = pd_sz_ - 4;
        const int sz  = pd_sz_;
        for (int n = 0; n < N; ++n) {
            const int wp   =  pd_ptr_           % sz;
            const int rp   = ((pd_ptr_ - d) % sz + sz) % sz;
            const float iL = io[n*2];
            const float iR = io[n*2+1];
            io[n*2]   = pd_buf_[rp*2];
            io[n*2+1] = pd_buf_[rp*2+1];
            pd_buf_[wp*2]   = iL;
            pd_buf_[wp*2+1] = iR;
            ++pd_ptr_;
        }
    }

    void _init_combs() {
        for (int i = 0; i < 4; ++i)
            combs_[i].init(COMB_MS[i], preset_.early_rt60,
                           preset_.early_damp_hz, sr_);
    }

    void _init_allpass() {
        for (int i = 0; i < 4; ++i)
            ap_[i] = AllpassDelay(AP_MS[i], 0.5f, sr_);
    }

    void _init_fdn() {
        fdnL_.init(FDN_L_MS, preset_.rt60, preset_.damp_hz, sr_);
        fdnR_.init(FDN_R_MS, preset_.rt60, preset_.damp_hz, sr_);
    }
};

// Required by ODR for constexpr static members (C++14 compat)
constexpr float HybridReverb32::FDN_L_MS[16];
constexpr float HybridReverb32::FDN_R_MS[16];
constexpr float HybridReverb32::COMB_MS[4];
constexpr float HybridReverb32::AP_MS[4];
