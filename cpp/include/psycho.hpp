#pragma once
#include "common.hpp"
#include <cmath>
/**
 * psycho.hpp — Psychoacoustic Stereo Processor
 * ══════════════════════════════════════════════
 *
 * THREE INDEPENDENT STAGES (all optional, independently configurable)
 * ───────────────────────────────────────────────────────────────────
 *
 *  1. HAAS EFFECT (inter-aural time difference — ITD)
 *     ─────────────────────────────────────────────────
 *     The Haas / precedence effect: a brief delay (0.6–30 ms) between L and R
 *     makes the sound feel MUCH wider than simple amplitude panning.  Delays
 *     < 1 ms sound phasey; 1–5 ms = classic "widening"; 5–30 ms = big room.
 *     We apply the delay to the SIDE channel in M/S domain so the mono mid
 *     is unaffected — preserving mono compatibility perfectly.
 *
 *  2. HRTF HEAD SHADOW (inter-aural level difference — ILD)
 *     ───────────────────────────────────────────────────────
 *     The human head absorbs high frequencies coming from the opposite side.
 *     Sounds to the left arrive at the right ear with:
 *       • ~0.6 ms of delay (ITD, handled above)
 *       • ~6 dB of HF attenuation above ~1.5 kHz (ILD — head shadow)
 *     Simulating this with a 1-pole LP on the cross-fed signal creates a
 *     convincing out-of-head, speaker-like impression through headphones.
 *
 *  3. CROSSFEED (headphone speaker simulation)
 *     ────────────────────────────────────────
 *     Conventional stereo through headphones places sounds "inside the head"
 *     because each ear hears ONLY its own channel.  In real speaker playback,
 *     each ear hears BOTH speakers (left ear hears the right speaker slightly
 *     delayed and attenuated).  Crossfeed simulates this by adding a small,
 *     delayed, low-passed replica of each channel into the opposite channel.
 *     Result: natural speaker-in-a-room sensation through wired headphones.
 *
 * References:
 *   Haas (1951) "The Influence of a Single Echo on the Audibility of Speech"
 *   Bregman (1990) "Auditory Scene Analysis"
 *   Glasberg & Moore (1990) psychoacoustic loudness models
 *   Linkwitz (2007) binaural crossfeed theory  headwize.com
 */

#include <vector>
#include <cmath>
#include <algorithm>

class PsychoStereo {
public:
    /**
     * @param haas_ms      ITD side-channel delay (ms, default 1.2)
     *                     Set 0 to disable Haas.
     * @param crossfeed    cross-channel blend coefficient (0–0.35, default 0.18)
     *                     Set 0 to disable crossfeed (use for speakers).
     * @param cf_delay_ms  crossfeed signal delay (ms, default 0.35)
     *                     Gives a slight comb but sounds more natural.
     * @param ild_cutoff   HRTF head-shadow LP cutoff Hz (default 1400)
     * @param sr           sample rate
     */
    explicit PsychoStereo(float haas_ms    =  1.2f,
                          float crossfeed  =  0.18f,
                          float cf_delay_ms=  0.35f,
                          float ild_cutoff = 1400.0f,
                          int   sr         = 44100)
        : crossfeed_(std::clamp(crossfeed, 0.0f, 0.35f)), sr_(sr)
    {
        // Haas delay on the SIDE channel (M/S domain)
        haas_d_  = static_cast<int>(haas_ms * sr / 1000.0f);
        haas_sz_ = haas_d_ + 4;
        haas_buf_.assign(haas_sz_, 0.0f);
        haas_ptr_ = 0;

        // Crossfeed delay
        cf_d_  = static_cast<int>(cf_delay_ms * sr / 1000.0f);
        cf_sz_ = cf_d_ + 4;
        cf_buf_L_.assign(cf_sz_, 0.0f);
        cf_buf_R_.assign(cf_sz_, 0.0f);
        cf_ptr_ = 0;

        // HRTF head-shadow LP (1-pole, per cross-fed channel)
        const float a1  = std::exp(-2.0f * kPI_F
                                   * ild_cutoff / sr);
        ild_coeff_      = 1.0f - a1;
        ild_state_L_    = 0.0f;
        ild_state_R_    = 0.0f;
    }

    /**
     * Process one stereo block in-place.
     * @param io  interleaved stereo (L,R,L,R,...), N*2 float32 values
     * @param N   stereo frame count
     */
    void process(float* RESTRICT io, int N) noexcept {
        for (int n = 0; n < N; ++n) {
            float L = io[n*2];
            float R = io[n*2+1];

            // ── Stage 1: Haas effect on side channel (M/S) ────────────
            if (haas_d_ > 0) {
                const float mid  = (L + R) * 0.5f;
                const float side = (L - R) * 0.5f;

                // Delay the side channel
                const int wp   = haas_ptr_ % haas_sz_;
                const int rp   = ((haas_ptr_ - haas_d_) % haas_sz_
                                  + haas_sz_) % haas_sz_;
                const float side_delayed = haas_buf_[rp];
                haas_buf_[wp] = side;
                haas_ptr_ = (haas_ptr_ + 1) % haas_sz_;

                // Decode back to L/R
                L = mid + side_delayed;
                R = mid - side_delayed;
            }

            // ── Stage 2: Crossfeed + HRTF head shadow ─────────────────
            if (crossfeed_ > 0.0f) {
                // Write current L and R into crossfeed delay buffers
                const int wp  = cf_ptr_ % cf_sz_;
                const int rp  = ((cf_ptr_ - cf_d_) % cf_sz_ + cf_sz_) % cf_sz_;

                cf_buf_L_[wp] = L;
                cf_buf_R_[wp] = R;
                cf_ptr_ = (cf_ptr_ + 1) % cf_sz_;

                // Cross-signal: delayed and head-shadow LP filtered
                float cf_L = cf_buf_R_[rp];   // right-to-left cross
                float cf_R = cf_buf_L_[rp];   // left-to-right cross

                // HRTF ILD: 1-pole LP (high-frequency shadow of the head)
                ild_state_L_ += ild_coeff_ * (cf_L - ild_state_L_);
                ild_state_R_ += ild_coeff_ * (cf_R - ild_state_R_);
                cf_L = ild_state_L_;
                cf_R = ild_state_R_;

                L += crossfeed_ * cf_L;
                R += crossfeed_ * cf_R;
            }

            io[n*2]   = L;
            io[n*2+1] = R;
        }
    }

    void set_crossfeed(float v) noexcept {
        crossfeed_ = std::clamp(v, 0.0f, 0.35f);
    }

private:
    int   sr_;
    float crossfeed_;

    // Haas delay (side channel in M/S domain)
    int   haas_d_, haas_sz_, haas_ptr_;
    std::vector<float> haas_buf_;

    // Crossfeed delay
    int   cf_d_, cf_sz_, cf_ptr_;
    std::vector<float> cf_buf_L_, cf_buf_R_;

    // HRTF head-shadow LP state
    float ild_coeff_, ild_state_L_, ild_state_R_;
};
