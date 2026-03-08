# Kudio

**Cinema-grade audio DSP engine for streaming music — built in C++ and Python.**

Kudio streams music from YouTube / YouTube Music, runs every frame through a
professional mastering-grade signal chain (written in C++ with pybind11 bindings),
and plays it back in real time with configurable reverb, loudness normalization,
parametric EQ, and psychoacoustic stereo enhancement.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange.svg)](#)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#)

---

## Table of Contents

- [Why Kudio](#why-kudio)
- [Signal Chain](#signal-chain)
- [DSP Modules](#dsp-modules)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [License](#license)

---

## Why Kudio

Most music players apply no DSP — or apply cheap brickwall EQ and call it
"enhancement". Kudio treats every chunk of audio as if it were passing through
a professional mastering chain:

- **4-way Linkwitz-Riley crossover** splits the signal into sub, bass, mid, and
  high bands so each band is compressed and shaped independently — no
  inter-band pumping.
- **32-line FDN reverb** (Schroeder + Fast Walsh-Hadamard Transform) adds
  spacious room character that a simple convolution reverb cannot reproduce
  in real time.
- **Psychoacoustic stereo** (Haas ITD + HRTF head-shadow simulation) gives
  headphone listeners a convincing out-of-head presentation without the
  comb-filtering artifacts of naive stereo widening.
- **ITU-R BS.1770-4 loudness normalization** means every track lands at the
  same perceived loudness regardless of how it was mastered.
- **True-peak limiter** (2x oversampled) prevents inter-sample clipping that a
  conventional sample-peak limiter misses.

All the heavy DSP runs in native C++ with `-O3 -march=native` (or `/O2
/arch:AVX2` on MSVC) — no Python GIL in the hot path, no garbage collection
pauses, no NumPy overhead on the inner loop.

---

## Signal Chain

```
YouTube / YouTube Music
        |
   yt-dlp  (search + metadata)
        |
   ffmpeg  (decode -> PCM f32le 44100 Hz stereo)
        |
+-------------------------------------------+
|              Python Layer                 |
|                                           |
|  [DC Blocker]          5 Hz HP biquad     |
|       |                                   |
|  [6-Band Parametric EQ]  user-adjustable  |
|       |                                   |
|  [LR-4 Multiband Split]  3x crossovers    |
|   SUB(80Hz) BASS(300Hz) MID(4kHz) HIGH    |
|       |                                   |
|  [Per-Band Compressor]  feed-forward peak |
|  [Per-Band EQ]          biquad shelves    |
|  [Recombine + safety clip]                |
|       |                                   |
|  [Bass Spatial Enhancer]  M/S sub punch   |
|       |                                   |
+---------------+---------------------------+
                |
+---------------+---------------------------+
|              C++ Layer  (kudio_dsp)       |
|                                           |
|  [SpectralShaper]     6-band cinema EQ    |
|       |                                   |
|  [PsychoStereo]       Haas + HRTF         |
|       |                                   |
+---------------+---------------------------+
                |
+---------------+---------------------------+
|              Python Layer                 |
|  [CinemaDelay]  2-tap slapback 42/135 ms  |
+---------------+---------------------------+
                |
+---------------+---------------------------+
|              C++ Layer                    |
|  [FDNReverb 32-line]  Schroeder + FWHT    |
|       |                                   |
|  [HarmonicExciter]    tube even+odd       |
|       |                                   |
|  [TruePeakLimiter]    2x OS, ITU-R        |
+---------------+---------------------------+
                |
+---------------+---------------------------+
|              Python Layer                 |
|  [LoudnessNormalizer]  LUFS gain rider    |
|  [LUFS Meter]          BS.1770-4 display  |
+-------------------------------------------+
                |
         sounddevice output
```

---

## DSP Modules

### DC Blocker
5 Hz highpass biquad. Removes any DC offset before the first nonlinear stage
so that saturation and compression behave symmetrically.

### 6-Band Parametric EQ (user)
Interactive EQ with six bands adjustable at startup:

| Band       | Frequency | Type        |
|------------|-----------|-------------|
| Bass       | 60 Hz     | Peak        |
| Low Mid    | 150 Hz    | Peak        |
| Mid        | 400 Hz    | Peak        |
| Upper Mid  | 1 kHz     | Peak        |
| Presence   | 2.4 kHz   | Peak        |
| Treble     | 15 kHz    | High shelf  |

Each band is a stateful scipy biquad with persistent `zi` state across chunk
boundaries — no clicks or zipper noise at chunk edges.

### Linkwitz-Riley 4th-Order Crossover (LR-4)
Three cascaded LR-4 crossovers (two 2nd-order Butterworth stages each) split
the signal into four bands. LR-4 has a maximally flat magnitude response — the
four bands sum perfectly to the original with no ripple at the crossover
frequencies.

### Per-Band Feed-Forward Compressor
Independent compressor on each of the four bands:

| Band  | Threshold | Ratio | Attack | Release |
|-------|-----------|-------|--------|---------|
| Sub   | -8 dB     | 4.0:1 | 5 ms   | 80 ms   |
| Bass  | -12 dB    | 2.5:1 | 12 ms  | 120 ms  |
| Mid   | -18 dB    | 1.5:1 | 30 ms  | 200 ms  |
| High  | -15 dB    | 2.0:1 | 8 ms   | 100 ms  |

The envelope follower is implemented as a `numpy.frompyfunc` left-fold scan —
equivalent to a sample-by-sample IIR but executed at C speed with no Python
loop. State is carried across chunk boundaries.

### Bass Spatial Enhancer
Mid/Side split below 120 Hz:

- **Mid channel**: soft-saturated with `tanh` for added harmonic punch
- **Side channel**: gain-boosted for sub-bass stereo width

Enhanced sub mixed at 35% wet. Phase-aligned — no comb filtering.

### SpectralShaper (C++)
6-band cinematic tonal curve in C++. The `cinema` preset lifts sub-bass
presence, dips the muddy 350 Hz region, and adds a broad high-frequency air
shelf — matching the character of a professional film mix bus. Bands are
runtime-reconfigurable via `set_band()`.

### PsychoStereo (C++)
Three-stage psychoacoustic headphone processor:

1. **Haas ITD** — 1.2 ms delay on the side channel simulates the inter-aural
   time difference used to localize sound. 1-5 ms gives convincing width.

2. **HRTF Head Shadow (ILD)** — one-pole lowpass on the cross-fed signal above
   ~1.4 kHz simulates the 6 dB HF attenuation caused by your head blocking
   the contralateral ear (physical basis of headphone stereo localization).

3. **Crossfeed** — blends 18% of each channel into the other (0.35 ms delay)
   to reduce the unnatural inside-the-head presentation of close-mic stereo.
   Based on Linkwitz binaural crossfeed theory.

### CinemaDelay (Python)
Two-tap slapback: 42 ms (rear wall) + 135 ms (ceiling reflection), with L/R
tap offsets of 5/11 ms for natural stereo spread. 2600 Hz lowpass models HF
absorption of room surfaces.

### FDN Reverb 32-line (C++)
Schroeder + Feedback Delay Network hybrid:

- **4 parallel Moorer comb filters** — dense early reflections with
  mode-dependent RT60, damping LP inside the loop
- **4 allpass diffusers** — smear reflections before feeding the FDN
- **FDN-16 per channel** (32 lines total) — feedback matrix is the normalized
  16x16 Hadamard transform via **Fast Walsh-Hadamard Transform** (O(N log N)
  vs O(N²)). FWHT is orthogonal and energy-preserving — no spectral colouration.

Four room presets:

| Mode       | RT60   | Pre-delay | Character               |
|------------|--------|-----------|-------------------------|
| hall       | 2.6 s  | 22 ms     | Orchestral concert hall |
| cathedral  | 5.0 s  | 35 ms     | Large stone cathedral   |
| plate      | 1.8 s  | 4 ms      | Vintage EMT-140 plate   |
| chamber    | 1.1 s  | 12 ms     | Small wood live room    |

### HarmonicExciter (C++)
Models harmonic distortion of analog valve/tube equipment:

- **Even harmonics** (2nd, 4th): `x^2 + 0.2*x^4` — warm, rounded, triode-tube character
- **Odd harmonics** (3rd): `tanh(x^3)` — presence and bite, pushed transistor stage

Both paths are highpass-gated above 2 kHz — sub-bass is never hardened.
Wet blend 22% — audible as air and sparkle, not distortion.

### TruePeakLimiter (C++)
ITU-R BS.1770-4 compliant true-peak brick-wall limiter:

1. **2x zero-stuffing upsample** — exposes inter-sample peaks invisible to a
   conventional sample-peak meter
2. **64-tap windowed-sinc FIR** (fc = 0.45 x Nyquist, Hann window) — filters
   the upsampled signal per channel with persistent state
3. **Gain scan** — ring-buffer look-ahead delay (3 ms), instant attack,
   exponential release (80 ms). Delay line stores **unscaled** FIR output;
   gain scan reads from it and writes to a **separate** output buffer —
   no read-after-write aliasing possible
4. **Decimate x2** — take even frames (FIR already band-limited)

Ceiling: -1 dBTP (1 dB headroom for DA conversion).

### Loudness Normalizer (Python)
Real-time LUFS gain rider with 3-second measurement window and 4-second
smoothing. Gain clamped to +/-12 dB.

| Mode   | Target   | Use case                    |
|--------|----------|-----------------------------|
| Quiet  | -23 LUFS | Preserves full dynamics     |
| Normal | -14 LUFS | Balanced everyday listening |
| Loud   | -11 LUFS | Noisy environments          |

### LUFS Meter (Python)
ITU-R BS.1770-4 K-weighted integrated loudness: +4 dB high-shelf pre-filter at
1681 Hz, 100 Hz RLB highpass, 400 ms sliding RMS window. Non-blocking
push/read via threading.Lock.

### Crossfade Engine (Python)
Track-to-track crossfade for playlist use:

- 1-12 second overlap (configurable)
- Equal-power cosine curve (sum of squares = 1 — no loudness dip at midpoint)
- Exponential curve option for abrupt transitions
- `AlbumNormalizer` computes energy-weighted album LUFS so album dynamics are
  preserved across fades

---

## Features

- Stream any song from YouTube Music or YouTube by name — no API key required
- 4 reverb modes: hall, cathedral, plate, chamber
- 3 loudness targets: -23 / -14 / -11 LUFS
- 6-band parametric EQ configurable at startup
- Bass spatial enhancement (M/S sub-bass width + harmonic punch)
- Live LUFS meter in the terminal
- Pure-Python fallback — runs without C++ build, same audio quality
- Auto-retry streaming with exponential backoff on network failure
- Automatic CDN URL refresh on expiry
- Cross-platform: Windows, Linux, macOS

---

## Requirements

| Dependency    | Version  | Notes                                |
|---------------|----------|--------------------------------------|
| Python        | >= 3.10  | 3.12+ recommended                    |
| cmake         | >= 3.14  |                                      |
| C++ compiler  | C++17    | MSVC 2019+, GCC 10+, Clang 12+       |
| pybind11      | >= 2.11  | auto-installed by cmake if missing   |
| numpy         | any      |                                      |
| scipy         | any      |                                      |
| sounddevice   | any      | PortAudio backend                    |
| yt-dlp        | any      |                                      |
| ffmpeg        | >= 4.0   | must be on PATH                      |

---

## Installation

### Windows

**1. Install prerequisites**

- [Python 3.12+](https://www.python.org/downloads/windows/) — tick "Add to PATH"
- [CMake](https://cmake.org/download/) — tick "Add to PATH"
- [Visual Studio 2019+](https://visualstudio.microsoft.com/) — select
  "Desktop development with C++" workload
- [ffmpeg](https://ffmpeg.org/download.html) — extract and add `bin/` to PATH

**2. Install Python dependencies**

```cmd
pip install numpy scipy sounddevice yt-dlp pybind11
```

**3. Clone the repository**

```cmd
git clone https://github.com/devarpanlohar/kudio.git
cd kudio
```

**4. Build the C++ DSP engine**

Open a **Developer Command Prompt for VS** (search in Start menu):

```cmd
rmdir /s /q build & mkdir build & cd build
cmake ..
cmake --build . --config Release
copy Release\kudio_dsp*.pyd ..
cd ..
```

**5. Run**

```cmd
python main.py
```

---

### Linux

**1. Install prerequisites**

```bash
# Debian / Ubuntu
sudo apt install python3 python3-pip cmake build-essential ffmpeg

# Arch
sudo pacman -S python python-pip cmake base-devel ffmpeg

# Fedora
sudo dnf install python3 python3-pip cmake gcc-c++ ffmpeg
```

**2. Install Python dependencies**

```bash
pip install numpy scipy sounddevice yt-dlp pybind11
```

**3. Clone and build**

```bash
git clone https://github.com/devarpanlohar/kudio.git
cd kudio
chmod +x build.sh && ./build.sh
```

**4. Run**

```bash
python main.py
```

---

### macOS

**1. Install prerequisites**

```bash
brew install cmake ffmpeg python
```

**2. Install Python dependencies**

```bash
pip3 install numpy scipy sounddevice yt-dlp pybind11
```

**3. Clone and build**

```bash
git clone https://github.com/devarpanlohar/kudio.git
cd kudio
chmod +x build.sh && ./build.sh
```

**4. Run**

```bash
python3 main.py
```

> **Apple Silicon:** If you use a specific Python version (e.g. pyenv), set
> `KUDIO_PYTHON` before building:
> ```bash
> export KUDIO_PYTHON=$(which python3.12)
> ./build.sh
> ```

---

### Pure-Python mode (no C++ build required)

Skip the build step entirely. Kudio falls back to pure-Python equivalents for
every C++ stage automatically. Same audio quality, slightly higher CPU usage.

```bash
pip install numpy scipy sounddevice yt-dlp
python main.py
```

---

## Usage

```
  _  __         _ _
 | |/ /   _  __| (_) ___
 | ' / | | |/ _` | |/ _ \
 | . \ |_| | (_| | | (_) |
 |_|\_\__,_|\__,_|_|\___/

  Cinema DSP Engine v3.1  [C++ Accelerated]

  Select reverb character:
    [1] Concert Hall   -- RT60 2.6s, orchestral depth
    [2] Cathedral      -- RT60 5.0s, stone / ethereal
    [3] Vintage Plate  -- RT60 1.8s, dense & intimate
    [4] Live Chamber   -- RT60 1.1s, wood & warmth

  Select loudness target:
    [1] Quiet  -- -23 LUFS, full dynamics preserved
    [2] Normal -- -14 LUFS, balanced (default)
    [3] Loud   -- -11 LUFS, noisy environments

  Customize EQ? [y/N]: y

  6-Band EQ (press Enter to skip / keep flat):
  Band         Freq       Gain dB
  ------------------------------------
  bass         60 Hz      3
  low_mid      150 Hz
  mid          400 Hz     -2
  upper_mid    1 kHz
  presence     2.4 kHz    1.5
  treble       15 kHz

  Enter song name: dark knight rises hans zimmer
```

Press `Ctrl+C` to stop playback.

---

## Architecture

```
Python  (UI / search / streaming / control)
    |   numpy (N,2) float32 chunks
    v
C++ DSP Engine  (kudio_dsp.pyd / kudio_dsp.so)
    |   pybind11 -- zero-copy NumPy interface
    |   MSVC /O2 /arch:AVX2  OR  GCC -O3 -march=native
    v
sounddevice (PortAudio) --> speakers / headphones
```

**Streaming thread model**

| Thread      | Responsibility                                             |
|-------------|------------------------------------------------------------|
| pipe_reader | Blocks on ffmpeg stdout — isolated so blocking never stalls others |
| assembler   | Accumulates variable-size blobs into exact 1024-frame chunks |
| manager     | Monitors ffmpeg health, retries with exponential backoff, refreshes CDN URLs |
| player      | Dequeues chunks -> DSP chain -> sounddevice write          |

The DSP call (`dsp.process()`) runs entirely in the player thread with no GIL
contention — pybind11 releases the GIL for all C++ processing.

---

## Performance

Benchmarks on i7-12700H, Windows 11, Python 3.14, MSVC Release.
1024-frame chunks at 44100 Hz = 23.2 ms budget per chunk.

| Stage               | Time per chunk | Notes                         |
|---------------------|----------------|-------------------------------|
| LR-4 crossover      | ~0.08 ms       | 12 scipy biquad calls         |
| Per-band compressor | ~0.15 ms       | frompyfunc accumulate scan    |
| SpectralShaper C++  | ~0.02 ms       | 6 biquads, SIMD-vectorized    |
| PsychoStereo C++    | ~0.03 ms       | delay line + 1-pole LP        |
| FDNReverb C++       | ~0.35 ms       | 32 delay lines + FWHT         |
| HarmonicExciter C++ | ~0.04 ms       | 2 biquads + waveshaper        |
| TruePeakLimiter C++ | ~0.12 ms       | 64-tap FIR + gain scan        |
| **Total chain**     | **~0.80 ms**   | **96% headroom remaining**    |

---

## Project Structure

```
kudio/
├── main.py           Entry point -- UI, reverb/loudness/EQ selection
├── streaming.py      4-thread ffmpeg streaming engine
├── dsp_chain.py      Signal chain orchestrator
├── filters.py        DCBlocker, Biquad, LR-4, CinemaDelay, sanitize()
├── compressor.py     Vectorized per-band feed-forward compressor
├── meters.py         ITU-R BS.1770-4 LUFS meter
├── crossfade.py      Track crossfade + AlbumNormalizer
├── CMakeLists.txt    Auto-detects Python, auto-installs pybind11
├── build.sh          One-command build for Linux / macOS
└── cpp/
    ├── kudio_dsp.cpp       pybind11 module entry point
    └── include/
        ├── common.hpp      kPI_F, RESTRICT macro
        ├── allpass.hpp     Allpass diffuser (used by FDN)
        ├── fdn.hpp         32-line Schroeder+FDN reverb
        ├── limiter.hpp     2x oversampled true-peak limiter
        ├── harmonic.hpp    Parallel harmonic exciter
        ├── psycho.hpp      Haas ITD + HRTF + crossfeed
        └── spectral.hpp    6-band cinematic spectral shaper
```

---

## License

MIT License

```
Copyright (c) 2026 Kudio Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## References

- Schroeder, M. R. (1962). *Natural Sounding Artificial Reverberation*. JAES.
- Jot, J.-M. & Chaigne, A. (1991). *Digital Delay Networks for Artificial Reverberators*. AES 90th.
- Smith, J. O. (2010). *Physical Audio Signal Processing*, §2.3 FDN Reverb.
- Zölzer, U. (2011). *DAFX: Digital Audio Effects*, 2nd ed., Ch. 7.
- ITU-R BS.1770-4 (2015). *Algorithms to measure audio programme loudness*.
- Nielsen, S. & Lund, T. (2003). *0 dBFS+ Levels in Digital Mastering*. AES 115th.
- Linkwitz, S. (2007). *Headphone Crossfeed*. headwize.com.
- Moorer, J. A. (1979). *About This Reverberation Business*. Computer Music Journal.