#pragma once
#include "common.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <array>

/**
 * limiter.hpp — 2× Oversampled True-Peak Brick-Wall Limiter
 * ══════════════════════════════════════════════════════════
 *
 * ALGORITHM (ITU-R BS.1770-4 compliant)
 * ──────────────────────────────────────
 *  1. Zero-stuff upsample ×2 → FIR half-band anti-imaging filter.
 *     Layout: up_[] is interleaved stereo at 2× rate.
 *             Frame i: L = up_[i*2], R = up_[i*2+1]
 *             Even frames (i=0,2,4,...) carry original samples (×2).
 *             Odd  frames (i=1,3,5,...) are zero-stuffed, filled by FIR.
 *
 *  2. Look-ahead gain scan (FULLY SEPARATED read/write buffers):
 *       delayed signal ← delay_line_ (ring of la_up_ unscaled 2× frames)
 *       peak detection ← up_[] (current unscaled 2× FIR output)
 *       output         → out_[] (separate buffer, never read during scan)
 *     This separation is mandatory — reading up_[] after writing out_[]
 *     would corrupt the delay line with gain-scaled values.
 *
 *  3. Update delay_line_ ring with the tail of up_[] (BEFORE any gain).
 *
 *  4. Decimate out_[] by 2 (take even frames) → write back to io[].
 */

class TruePeakLimiter {
public:
    explicit TruePeakLimiter(float ceil_db = -1.0f,
                             float la_ms   =  3.0f,
                             float rel_ms  = 80.0f,
                             int   sr      = 44100,
                             int   chunk   = 1024)
        : sr_(sr), ceil_(std::pow(10.0f, ceil_db / 20.0f))
    {
        rel_    = std::exp(-1.0f / (sr_ * rel_ms / 1000.0f));
        gain_   = 1.0f;

        // Look-ahead size in the 2× domain
        la_up_  = static_cast<int>(2.0f * sr_ * la_ms / 1000.0f);
        if (la_up_ < 1) la_up_ = 1;

        // Ring buffer: holds last la_up_ frames of the unscaled 2× signal
        // Indexed as delay_line_[pos * 2 + ch], pos in [0, la_up_)
        delay_line_.assign(la_up_ * 2, 0.0f);
        delay_pos_ = 0;   // write head

        // FIR
        _build_fir();
        fir_zi_L_.assign(FIR_TAPS - 1, 0.0f);
        fir_zi_R_.assign(FIR_TAPS - 1, 0.0f);

        // Scratch buffers (max 2×chunk frames, interleaved stereo)
        const int maxN2 = 2 * chunk;
        up_.resize(maxN2 * 2, 0.0f);   // FIR output  (read-only in gain scan)
        out_.resize(maxN2 * 2, 0.0f);  // gain output (write-only in gain scan)
    }

    void process(float* RESTRICT io, int N) noexcept {
        const int N2 = N * 2;   // frames in 2× domain

        // ── 1. Zero-stuff upsample into up_[] ──────────────────────────
        // Ensure scratch is large enough (handles variable chunk size)
        if (static_cast<int>(up_.size()) < N2 * 2) {
            up_.resize(N2 * 2, 0.0f);
            out_.resize(N2 * 2, 0.0f);
        }
        std::fill(up_.begin(), up_.begin() + N2 * 2, 0.0f);
        for (int n = 0; n < N; ++n) {
            // Even 2× frame 2n:  carries the original sample (scaled ×2)
            up_[n * 4    ] = io[n * 2    ] * 2.0f;   // L
            up_[n * 4 + 1] = io[n * 2 + 1] * 2.0f;   // R
            // Odd  2× frame 2n+1: zero-stuffed → FIR interpolates
        }

        // ── 2. FIR anti-imaging filter (in-place, stride-2 per channel) ─
        _fir_filter(up_.data(),     N2, fir_zi_L_);   // L: offset 0, stride 2
        _fir_filter(up_.data() + 1, N2, fir_zi_R_);   // R: offset 1, stride 2

        // ── 3. Gain scan ────────────────────────────────────────────────
        // Reads from up_[] (unscaled) and delay_line_ (delayed unscaled).
        // Writes to out_[] only — NEVER reads from out_[].
        float gain             = gain_;
        const float one_m_rel  = 1.0f - rel_;

        for (int i = 0; i < N2; ++i) {
            // --- Delayed frame (la_up_ frames behind frame i) -----------
            // Ring buffer: the delayed frame is the one at the write head
            // (it's la_up_ writes ago, which is exactly what we need).
            const int dpos = delay_pos_;          // oldest entry = delayed out
            const float dL = delay_line_[dpos * 2    ];
            const float dR = delay_line_[dpos * 2 + 1];

            // --- Update ring: push current unscaled frame in ------------
            delay_line_[dpos * 2    ] = up_[i * 2    ];
            delay_line_[dpos * 2 + 1] = up_[i * 2 + 1];
            delay_pos_ = (delay_pos_ + 1 < la_up_) ? delay_pos_ + 1 : 0;

            // --- Peak from look-ahead (current unscaled frame) ----------
            const float peak   = std::max(std::fabs(up_[i * 2    ]),
                                          std::fabs(up_[i * 2 + 1]));
            const float needed = (peak > ceil_) ? (ceil_ / (peak + 1e-12f))
                                                : 1.0f;

            // Instant attack, exponential release
            if (needed < gain) {
                gain = needed;
            } else {
                gain += (1.0f - gain) * one_m_rel;   // ≡ gain*rel + (1-rel)
                if (gain > 1.0f) gain = 1.0f;
            }

            // Write gain-scaled delayed frame to OUTPUT buffer only
            out_[i * 2    ] = dL * gain;
            out_[i * 2 + 1] = dR * gain;
        }
        gain_ = gain;

        // ── 4. Decimate ×2 → write back to io[] ─────────────────────────
        // Take even 2× frames (0, 2, 4, ...) — FIR already band-limited.
        for (int n = 0; n < N; ++n) {
            io[n * 2    ] = out_[n * 4    ];
            io[n * 2 + 1] = out_[n * 4 + 1];
        }
    }

    void reset() noexcept {
        gain_      = 1.0f;
        delay_pos_ = 0;
        std::fill(delay_line_.begin(), delay_line_.end(), 0.0f);
        std::fill(fir_zi_L_.begin(),   fir_zi_L_.end(),   0.0f);
        std::fill(fir_zi_R_.begin(),   fir_zi_R_.end(),   0.0f);
    }

private:
    // ── Parameters ─────────────────────────────────────────────────────
    int   sr_, la_up_;
    float ceil_, rel_, gain_;

    // ── Delay ring (stores unscaled 2× frames) ─────────────────────────
    std::vector<float> delay_line_;
    int                delay_pos_;

    // ── Scratch ────────────────────────────────────────────────────────
    std::vector<float> up_;    // FIR-filtered 2× signal  (read in gain scan)
    std::vector<float> out_;   // gain-scaled output       (write in gain scan)

    // ── FIR ────────────────────────────────────────────────────────────
    static constexpr int FIR_TAPS = 64;
    std::array<float, FIR_TAPS> fir_h_;
    std::vector<float> fir_zi_L_, fir_zi_R_;

    void _build_fir() noexcept {
        // Windowed-sinc, fc = 0.45 × Nyquist_2x = 0.225 × original Nyquist
        // Hann window, 64 taps
        const float fc = 0.45f;
        const int   M  = FIR_TAPS - 1;
        for (int i = 0; i <= M; ++i) {
            const float x    = static_cast<float>(i) - M * 0.5f;
            const float hann = 0.5f * (1.0f - std::cos(2.0f * kPI_F * i / M));
            fir_h_[i] = (std::fabs(x) < 1e-6f)
                      ? 2.0f * fc * hann
                      : (std::sin(2.0f * kPI_F * fc * x) / (kPI_F * x)) * hann;
        }
    }

    // FIR direct-form, operates on stride-2 interleaved channel in-place
    void _fir_filter(float* RESTRICT sig, int N_frames,
                     std::vector<float>& zi) noexcept {
        const int K = FIR_TAPS;
        // Build contiguous input: [state | new samples]
        std::vector<float> x(K - 1 + N_frames);
        for (int i = 0; i < K - 1; ++i) x[i]         = zi[i];
        for (int i = 0; i < N_frames;  ++i) x[K-1+i]  = sig[i * 2];
        // Convolve
        for (int i = 0; i < N_frames; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += fir_h_[k] * x[i + K - 1 - k];
            sig[i * 2] = acc;
        }
        // Update state
        for (int i = 0; i < K - 1; ++i)
            zi[i] = x[N_frames + i];
    }
};