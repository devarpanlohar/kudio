#pragma once
#include "common.hpp"
/**
 * allpass.hpp — First-order Schroeder Allpass Delay Unit
 * ═══════════════════════════════════════════════════════
 * Transfer function:  H(z) = (z^{-d} - g) / (1 - g·z^{-d})
 *
 * Magnitude response: FLAT (all-pass)
 * Phase response    : non-linear → smears / diffuses transients
 *
 * This is the core building block of Schroeder / Moorer reverb diffusers.
 * Used in chains of 3-6 stages to build up echo density before the
 * feedback delay network (FDN) takes over the late tail.
 *
 * Standard sample-by-sample recurrence:
 *   delayed  = buf[(ptr - d) % sz]
 *   y        = delayed - g * x
 *   buf[ptr] = x + g * y          ← stores sum so delayed read = x + g*y
 *   out      = y
 *
 * Reference: Schroeder (1962) "Natural Sounding Artificial Reverberation"
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>


class AllpassDelay {
public:
    AllpassDelay() : d_(0), sz_(1), g_(0.5f), ptr_(0) {
        buf_.assign(1, 0.0f);
    }

    /**
     * @param delay_ms  allpass delay time in milliseconds
     * @param gain      allpass coefficient  0 < g < 1  (0.5 = classic Schroeder)
     * @param sr        sample rate in Hz
     */
    AllpassDelay(float delay_ms, float gain = 0.5f, int sr = 44100)
        : g_(std::clamp(gain, 0.001f, 0.999f)), ptr_(0)
    {
        d_  = static_cast<int>(delay_ms * sr / 1000.0f);
        sz_ = d_ + 4;                    // small safety margin
        buf_.assign(sz_, 0.0f);
    }

    /** Tick one mono sample through the allpass. */
    inline float tick(float x) noexcept {
        const int read_idx  = ((ptr_ - d_) % sz_ + sz_) % sz_;
        const float delayed = buf_[read_idx];
        const float y       = delayed - g_ * x;
        buf_[ptr_]          = x + g_ * y;
        ptr_                = (ptr_ + 1) % sz_;
        return y;
    }

    /** Process a contiguous mono block in-place. */
    void process_inplace(float* RESTRICT mono, int N) noexcept {
        for (int n = 0; n < N; ++n)
            mono[n] = tick(mono[n]);
    }

    /** Process a contiguous mono block into a separate output buffer. */
    void process(const float* RESTRICT in,
                 float*       RESTRICT out, int N) noexcept {
        for (int n = 0; n < N; ++n)
            out[n] = tick(in[n]);
    }

    void reset() noexcept {
        std::fill(buf_.begin(), buf_.end(), 0.0f);
        ptr_ = 0;
    }

private:
    int               d_, sz_, ptr_;
    float             g_;
    std::vector<float> buf_;
};
